# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-silico gene perturbation helpers for MaxToki.

Operates directly on rank-value-encoded token sequences so no counts or medians
are needed. A screen is built by pairing each "old" cell with a "young" anchor
and applying a set of perturbations to the old cell's rank-value sequence.

Two scoring readouts are supported:

* **Regression (TimeBetweenCells).** Use :func:`build_regression_prompt` and
  :func:`build_screen_dataset`; the prompts feed into
  ``python -m bionemo.maxtoki.predict`` in regression mode. Requires the
  temporal / fine-tuned MaxToki checkpoint (the one with the regression head,
  i.e. ``MaxTokiMultitaskFineTuneConfig``). Lower predicted time-gap =
  more rejuvenated.

* **Log-likelihood (NextCell log-prob).** Use :func:`build_scoring_prompt` and
  :func:`build_screen_dataset_scoring`; scores are computed by
  :func:`score_screen`, which takes a loaded HuggingFace-format causal LM
  (e.g. the published ``MaxToki-217M-HF`` checkpoint). Works with the
  pretraining-only checkpoint — no regression head required. Higher
  ``log P(young | perturbed_old)`` = the model thinks a young successor state
  is more plausible = more rejuvenated.

Example (log-likelihood readout, works with the pretrained checkpoint):

.. code-block :: python

    from transformers import AutoModelForCausalLM
    from bionemo.maxtoki.perturb import (
        CellPair, make_knockout_spec, make_noop_spec,
        build_screen_dataset_scoring, score_screen,
    )

    pairs = [CellPair(young_tokens=young_row["input_ids"],
                      old_tokens=old_row["input_ids"],
                      cell_id=f"{donor}:{barcode}") for ...]
    specs = [make_noop_spec()] + [make_knockout_spec(tokenizer, g)
                                  for g in ["TGFB1", "CCN2", "ACTA2"]]
    ds_path, manifest_path = build_screen_dataset_scoring(
        tokenizer, pairs, specs, "/data/screen_tgfb",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "/ptmp/$USER/models/maxtoki-hf/MaxToki-217M-HF", torch_dtype="auto",
    ).to("cuda").eval()
    df = score_screen(model, ds_path, manifest_path, device="cuda")
    # df has columns: spec_name, sum_logprob, mean_logprob, delta_vs_baseline, ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import datasets
import numpy as np
import pandas as pd

from bionemo.maxtoki.tokenizer import MaxTokiTokenizer


RankTokens = list[int]
"""Gene-only token list, without the <bos>/<eos> wrappers."""


# ---------------------------------------------------------------------------
# Low-level token ops
# ---------------------------------------------------------------------------


def strip_bos_eos(cell_tokens: Sequence[int], tokenizer: MaxTokiTokenizer) -> RankTokens:
    """Return the rank-value token list with leading <bos> and trailing <eos> removed."""
    tokens = list(cell_tokens)
    if tokens and tokens[0] == tokenizer.bos_id:
        tokens = tokens[1:]
    if tokens and tokens[-1] == tokenizer.eos_id:
        tokens = tokens[:-1]
    return tokens


def wrap_bos_eos(rank_tokens: Sequence[int], tokenizer: MaxTokiTokenizer) -> list[int]:
    """Wrap a rank-value token list in <bos>...<eos>."""
    return [tokenizer.bos_id, *rank_tokens, tokenizer.eos_id]


def knockout(rank_tokens: Sequence[int], gene_ids: int | Iterable[int]) -> RankTokens:
    """Remove one or more gene tokens. Genes not present in the cell are skipped silently."""
    drop = {gene_ids} if isinstance(gene_ids, int) else set(gene_ids)
    return [t for t in rank_tokens if t not in drop]


def overexpress(rank_tokens: Sequence[int], gene_ids: int | Iterable[int]) -> RankTokens:
    """Promote one or more gene tokens to the top of the rank.

    When multiple ids are given, the first id becomes rank 0, the second rank 1, etc.
    Genes already in the cell are moved; genes absent are inserted at the top.
    """
    ids = [gene_ids] if isinstance(gene_ids, int) else list(gene_ids)
    rest = [t for t in rank_tokens if t not in set(ids)]
    return [*ids, *rest]


def soft_overexpress(
    rank_tokens: Sequence[int],
    gene_id: int,
    boost_ranks: int = 50,
) -> RankTokens:
    """Move a gene up by ``boost_ranks`` positions (physiologically plausible OE).

    Unlike :func:`overexpress`, which pins the gene to rank 0 and creates an
    out-of-distribution profile for transcription factors (whose absolute mRNA
    counts are normally low), this operator shifts the gene's rank by a
    bounded amount:

    * **Gene already present at rank R** → new rank ``max(0, R - boost_ranks)``.
    * **Gene absent** → inserted at rank ``max(0, len(rank_tokens) - boost_ranks)``.
      For ``boost_ranks >= len(rank_tokens)`` this reduces to hard OE (rank 0).

    Args:
        rank_tokens: Rank-value token list (no <bos>/<eos> wrappers).
        gene_id: Token id of the gene to boost.
        boost_ranks: Number of positions to move the gene toward rank 0.
            Must be non-negative. Reasonable values: 25–200.

    Returns:
        Modified rank-value token list of length ``len(rank_tokens)`` (if the
        gene was already present) or ``len(rank_tokens) + 1`` (if inserted).
    """
    if boost_ranks < 0:
        raise ValueError(f"boost_ranks must be non-negative, got {boost_ranks}")
    tokens = list(rank_tokens)
    try:
        current_rank = tokens.index(gene_id)
        tokens.pop(current_rank)
        new_rank = max(0, current_rank - boost_ranks)
    except ValueError:
        new_rank = max(0, len(tokens) - boost_ranks)
    tokens.insert(new_rank, gene_id)
    return tokens


# ---------------------------------------------------------------------------
# Perturbation specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerturbationSpec:
    """A named transformation of a rank-value token list.

    Attributes:
        name: Short identifier used in the screen manifest (e.g. ``"KO:TGFB1"``).
        op: Callable that maps rank-value tokens to a modified rank-value list.
        metadata: Arbitrary extra fields written to the manifest.
    """

    name: str
    op: Callable[[RankTokens], RankTokens]
    metadata: dict[str, str] = field(default_factory=dict)


def _gene_id(tokenizer: MaxTokiTokenizer, gene_symbol: str) -> int:
    token_id = tokenizer.token_dictionary.get(gene_symbol)
    if token_id is None:
        raise KeyError(f"Gene {gene_symbol!r} not in tokenizer vocabulary")
    return token_id


def make_knockout_spec(tokenizer: MaxTokiTokenizer, gene_symbol: str) -> PerturbationSpec:
    """Build a knockout spec for a single gene symbol (Ensembl ID or HGNC symbol as vocabulary uses)."""
    token_id = _gene_id(tokenizer, gene_symbol)
    return PerturbationSpec(
        name=f"KO:{gene_symbol}",
        op=lambda rt, _id=token_id: knockout(rt, _id),
        metadata={"kind": "knockout", "gene": gene_symbol},
    )


def make_overexpression_spec(tokenizer: MaxTokiTokenizer, gene_symbol: str) -> PerturbationSpec:
    """Build an overexpression spec that moves the gene to rank 0 of the perturbed cell."""
    token_id = _gene_id(tokenizer, gene_symbol)
    return PerturbationSpec(
        name=f"OE:{gene_symbol}",
        op=lambda rt, _id=token_id: overexpress(rt, _id),
        metadata={"kind": "overexpression", "gene": gene_symbol},
    )


def make_soft_overexpression_spec(
    tokenizer: MaxTokiTokenizer,
    gene_symbol: str,
    boost_ranks: int = 50,
) -> PerturbationSpec:
    """Build a soft OE spec that boosts ``gene_symbol`` by ``boost_ranks`` positions.

    Physiologically more realistic than :func:`make_overexpression_spec` for
    transcription factors: TFs normally sit mid-rank, and pinning them to
    rank 0 makes the profile out-of-distribution for the model. See
    :func:`soft_overexpress` for the rank-shift semantics.
    """
    token_id = _gene_id(tokenizer, gene_symbol)
    return PerturbationSpec(
        name=f"sOE{boost_ranks}:{gene_symbol}",
        op=lambda rt, _id=token_id, _k=boost_ranks: soft_overexpress(rt, _id, _k),
        metadata={
            "kind": "soft_overexpression",
            "gene": gene_symbol,
            "boost_ranks": str(boost_ranks),
        },
    )


def make_soft_combo_spec(
    tokenizer: MaxTokiTokenizer,
    overexpressions: Sequence[str],
    boost_ranks: int = 50,
    name: str | None = None,
) -> PerturbationSpec:
    """Build a soft-OE combo spec (e.g. OSKM) that boosts each gene by ``boost_ranks``.

    Genes are boosted sequentially in the given order via :func:`soft_overexpress`.
    For absent genes in a cell with few tokens, boosts stack near the top because
    each insertion extends the list; for very large ``boost_ranks`` this reduces
    to hard multi-gene OE.
    """
    ids = [_gene_id(tokenizer, g) for g in overexpressions]

    def op(rt: RankTokens, _ids=ids, _k=boost_ranks) -> RankTokens:
        out = list(rt)
        for gid in _ids:
            out = soft_overexpress(out, gid, _k)
        return out

    if name is None:
        name = f"sOE{boost_ranks}_COMBO:{'+'.join(overexpressions)}"
    else:
        name = f"sOE{boost_ranks}_COMBO:{name}"
    return PerturbationSpec(
        name=name,
        op=op,
        metadata={
            "kind": "soft_overexpression_combo",
            "overexpressions": ",".join(overexpressions),
            "boost_ranks": str(boost_ranks),
        },
    )


def make_combo_spec(
    tokenizer: MaxTokiTokenizer,
    knockouts: Sequence[str] = (),
    overexpressions: Sequence[str] = (),
    name: str | None = None,
) -> PerturbationSpec:
    """Build a combinatorial spec. Knockouts are applied first, then overexpressions (promoted to top in order)."""
    ko_ids = [_gene_id(tokenizer, g) for g in knockouts]
    oe_ids = [_gene_id(tokenizer, g) for g in overexpressions]

    def op(rt: RankTokens, _ko=ko_ids, _oe=oe_ids) -> RankTokens:
        out = knockout(rt, _ko) if _ko else list(rt)
        if _oe:
            out = overexpress(out, _oe)
        return out

    if name is None:
        parts = [*(f"KO:{g}" for g in knockouts), *(f"OE:{g}" for g in overexpressions)]
        name = "+".join(parts) if parts else "noop"
    return PerturbationSpec(
        name=name,
        op=op,
        metadata={
            "kind": "combo",
            "knockouts": ",".join(knockouts),
            "overexpressions": ",".join(overexpressions),
        },
    )


def make_noop_spec() -> PerturbationSpec:
    """Identity transformation; used as the per-cell baseline."""
    return PerturbationSpec(name="baseline", op=lambda rt: list(rt), metadata={"kind": "baseline"})


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _numeric_token_id(tokenizer: MaxTokiTokenizer, value: int) -> int:
    """Look up the token id for an integer time value.

    The numeric keys in ``tokenizer.token_dictionary`` may be either ``int`` or
    ``str`` depending on how it was loaded (JSON → ``str`` keys, pickle →
    ``int`` keys). :class:`MaxTokiTokenizer` builds :attr:`numeric_tokens`
    keyed by the ``str`` form, so use that.
    """
    key = str(int(value))
    if key not in tokenizer.numeric_tokens:
        raise KeyError(
            f"Time value {value} has no numeric token in the tokenizer vocabulary. "
            "Ensure the tokenizer's time dictionary covers the required range."
        )
    return tokenizer.numeric_tokens[key]


def build_regression_prompt(
    tokenizer: MaxTokiTokenizer,
    context_cells: Sequence[RankTokens],
    context_timesteps: Sequence[int],
    question_cell: RankTokens,
    time_placeholder: int = 0,
) -> list[int]:
    """Build ``input_ids`` for a TimeBetweenCells regression inference sample.

    Mirrors :func:`bionemo.maxtoki.data_prep.dataset_utils.compose_example` with
    ``predict_time_or_cell="time"``. Layout::

        c_0 [t_0] c_1 [t_1] ... c_{N-2} <boq> c_{N-1} <eoq> [placeholder]

    where each ``c_i`` is ``<bos> ... <eos>`` wrapped. The trailing numeric
    placeholder is required so the multitask collator classifies the sample
    as ``TimeBetweenCells``; it does not influence the regression output,
    which is read at the ``<eoq>`` position.

    Args:
        tokenizer: MaxToki tokenizer instance (provides special-token and time-token ids).
        context_cells: ``N-1`` cells forming the prompt context, each as a
            rank-token list *without* <bos>/<eos>. Pass an empty list for the
            minimal 2-cell trajectory (only a young anchor + question cell).
        context_timesteps: ``N-2`` time intervals between adjacent context
            cells. Must satisfy ``len == max(0, len(context_cells) - 1)``.
            Ignored when ``context_cells`` is empty.
        question_cell: The final cell (``c_{N-1}``), rank tokens without
            <bos>/<eos>. This is the one that carries the perturbation.
        time_placeholder: Integer value used for the trailing numeric token.
            Any value with a token id works; ``0`` is a safe default.

    Returns:
        A single ``input_ids`` list ready to embed in a HuggingFace dataset.
    """
    if context_timesteps and len(context_timesteps) != max(0, len(context_cells) - 1):
        raise ValueError(
            f"context_timesteps must have length len(context_cells)-1; "
            f"got {len(context_timesteps)} timesteps for {len(context_cells)} context cells"
        )

    tokens: list[int] = []
    for i, cell in enumerate(context_cells):
        tokens.extend(wrap_bos_eos(cell, tokenizer))
        if i < len(context_cells) - 1:
            tokens.append(_numeric_token_id(tokenizer, context_timesteps[i]))

    tokens.append(tokenizer.boq_id)
    tokens.extend(wrap_bos_eos(question_cell, tokenizer))
    tokens.append(tokenizer.eoq_id)
    tokens.append(_numeric_token_id(tokenizer, time_placeholder))
    return tokens


# ---------------------------------------------------------------------------
# Screen dataset builder
# ---------------------------------------------------------------------------


@dataclass
class CellPair:
    """A (young anchor, old cell) pair, plus any metadata carried to the manifest.

    ``young_tokens`` and ``old_tokens`` are expected to be the ``input_ids`` as
    produced by :class:`bionemo.maxtoki.data_prep.TranscriptomeTokenizer`, i.e.
    wrapped in <bos> ... <eos>. They are stripped internally.
    """

    young_tokens: Sequence[int]
    old_tokens: Sequence[int]
    cell_id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def build_screen_dataset(
    tokenizer: MaxTokiTokenizer,
    pairs: Iterable[CellPair],
    specs: Iterable[PerturbationSpec],
    output_dir: str | Path,
    model_input_size: int = 16_384,
    time_placeholder: int = 0,
    skip_too_long: bool = True,
) -> tuple[Path, Path]:
    """Materialize a perturbation screen as a HuggingFace dataset on disk.

    For each (cell pair × spec) combination the perturbation is applied to the
    old cell's rank-value tokens, the regression prompt is built, and the
    sample is written to ``output_dir`` as a HF ``datasets`` directory
    (loadable by ``datasets.load_from_disk`` — the same path accepted by
    ``python -m bionemo.maxtoki.predict --data-path``). A sidecar
    ``manifest.csv`` is written alongside with one row per prompt, indexed
    in the same order as the dataset.

    Args:
        tokenizer: MaxToki tokenizer.
        pairs: Iterable of :class:`CellPair` (young anchor + old cell to perturb).
        specs: Iterable of :class:`PerturbationSpec` to apply to every pair.
        output_dir: Directory to write the HF dataset and ``manifest.csv`` to.
            Created if missing; existing contents may be overwritten.
        model_input_size: Max sequence length. Prompts longer than this are
            either skipped (``skip_too_long=True``) or raise (default True,
            mirrors the framework's hard limit — exceeding leads to
            truncation-before-<eos> in the paragraph assembler, which is
            lossy and inappropriate for scored perturbations).
        time_placeholder: Integer time placeholder at the end of each prompt.
            Must be in the tokenizer's time-token range.
        skip_too_long: Skip samples that exceed ``model_input_size`` instead
            of raising.

    Returns:
        ``(dataset_path, manifest_path)``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = list(specs)
    if not specs:
        raise ValueError("At least one PerturbationSpec is required")

    input_ids: list[list[int]] = []
    manifest_rows: list[dict[str, object]] = []
    skipped = 0

    for pair_idx, pair in enumerate(pairs):
        young_rt = strip_bos_eos(pair.young_tokens, tokenizer)
        old_rt = strip_bos_eos(pair.old_tokens, tokenizer)

        for spec in specs:
            perturbed = spec.op(old_rt)
            prompt = build_regression_prompt(
                tokenizer,
                context_cells=[young_rt],
                context_timesteps=[],
                question_cell=perturbed,
                time_placeholder=time_placeholder,
            )
            if len(prompt) > model_input_size:
                if skip_too_long:
                    skipped += 1
                    continue
                raise ValueError(
                    f"Prompt for pair={pair.cell_id} spec={spec.name} has length {len(prompt)} "
                    f"> model_input_size={model_input_size}"
                )

            row_idx = len(input_ids)
            input_ids.append(prompt)
            manifest_rows.append({
                "row": row_idx,
                "pair_idx": pair_idx,
                "cell_id": pair.cell_id,
                "spec_name": spec.name,
                "prompt_length": len(prompt),
                **spec.metadata,
                **pair.metadata,
            })

    if not input_ids:
        raise ValueError("No prompts produced — all pairs were too long or the input iterable was empty.")

    ds = datasets.Dataset.from_dict({"input_ids": input_ids})
    dataset_path = out / "dataset"
    ds.save_to_disk(str(dataset_path))

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    meta_path = out / "screen_meta.json"
    meta_path.write_text(json.dumps({
        "n_prompts": len(input_ids),
        "n_pairs": pair_idx + 1 if input_ids else 0,
        "n_specs": len(specs),
        "skipped_too_long": skipped,
        "model_input_size": model_input_size,
        "time_placeholder": time_placeholder,
    }, indent=2))
    return dataset_path, manifest_path


# ---------------------------------------------------------------------------
# Results parsing
# ---------------------------------------------------------------------------


def load_screen_predictions(
    predictions_dir: str | Path,
    manifest_path: str | Path,
) -> pd.DataFrame:
    """Join ``predictions__rank_*.pt`` outputs with the screen manifest.

    MaxToki writes one file per prediction rank. Each file contains a
    dictionary with a ``regression_preds`` tensor shaped ``[1, B]`` (one value
    per sample in rank order). Concatenating across ranks yields one value per
    row of the input dataset — which matches the manifest row order.

    Args:
        predictions_dir: Directory passed to ``predict --output-dir``.
        manifest_path: The ``manifest.csv`` produced by :func:`build_screen_dataset`.

    Returns:
        The manifest augmented with a ``predicted_time_gap`` column. Lower
        values suggest the perturbed old cell is closer in time to the young
        anchor — i.e. a more rejuvenated predicted state.
    """
    import torch

    pred_dir = Path(predictions_dir)
    files = sorted(pred_dir.glob("predictions__rank_*.pt"))
    if not files:
        raise FileNotFoundError(f"No predictions__rank_*.pt files in {pred_dir}")

    preds: list[np.ndarray] = []
    for f in files:
        payload = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "regression_preds" in payload:
            arr = payload["regression_preds"]
        else:
            raise ValueError(f"Unexpected prediction payload in {f}: keys={list(payload)}")
        preds.append(np.asarray(arr).reshape(-1))

    flat = np.concatenate(preds)
    manifest = pd.read_csv(manifest_path)
    if len(flat) != len(manifest):
        raise ValueError(
            f"Predictions ({len(flat)}) and manifest rows ({len(manifest)}) disagree; "
            "check that --limit-predict-batches-to-n was not used."
        )

    manifest = manifest.copy()
    manifest["predicted_time_gap"] = flat
    baseline = manifest[manifest["spec_name"] == "baseline"].set_index("pair_idx")["predicted_time_gap"]
    if not baseline.empty:
        manifest["delta_vs_baseline"] = manifest["predicted_time_gap"] - manifest["pair_idx"].map(baseline)
    return manifest


# ---------------------------------------------------------------------------
# Log-likelihood readout (works with pretraining-only checkpoint)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoringPrompt:
    """A prompt for log-likelihood-of-young-given-perturbed-old scoring.

    Attributes:
        input_ids: Full token sequence ``<bos> old_perturbed <eos> <bos> young <eos>``.
        target_start: Index of the first target token to score (inclusive).
            The target span is the young cell body only — we condition on
            ``<bos> old_perturbed <eos> <bos>`` and score from the first
            young gene token.
        target_end: One past the last target token to score (exclusive).
            Points at ``<eos>`` of the young cell, which is not scored.
    """

    input_ids: list[int]
    target_start: int
    target_end: int


def build_scoring_prompt(
    tokenizer: MaxTokiTokenizer,
    context_cell: RankTokens,
    target_cell: RankTokens,
) -> ScoringPrompt:
    """Build a prompt that scores the log-probability of a target cell given a context cell.

    Layout: ``<bos> context <eos> <bos> target <eos>``. The *target* cell body
    (gene tokens only, excluding both <bos> and <eos>) is what gets scored;
    the model conditions on everything up to and including the target cell's
    leading ``<bos>``.

    This uses only the autoregressive LM head, so it works with the published
    pretraining-only MaxToki checkpoint.

    Args:
        tokenizer: MaxToki tokenizer (for <bos>/<eos> ids).
        context_cell: Rank-value tokens of the context cell (usually the
            perturbed old cell), without <bos>/<eos>.
        target_cell: Rank-value tokens of the target cell (usually the young
            anchor), without <bos>/<eos>.

    Returns:
        :class:`ScoringPrompt` with ``input_ids`` and the ``[target_start, target_end)``
        span to score log-probs over.
    """
    context_wrapped = wrap_bos_eos(context_cell, tokenizer)
    target_wrapped = wrap_bos_eos(target_cell, tokenizer)

    input_ids = [*context_wrapped, *target_wrapped]
    target_start = len(context_wrapped) + 1  # skip the target's leading <bos>
    target_end = len(input_ids) - 1  # exclude the target's trailing <eos>
    return ScoringPrompt(input_ids=input_ids, target_start=target_start, target_end=target_end)


def build_screen_dataset_scoring(
    tokenizer: MaxTokiTokenizer,
    pairs: Iterable[CellPair],
    specs: Iterable[PerturbationSpec],
    output_dir: str | Path,
    model_input_size: int = 16_384,
    skip_too_long: bool = True,
) -> tuple[Path, Path]:
    """Materialize a log-likelihood perturbation screen as a HF dataset on disk.

    For each ``(cell pair × spec)`` combination, the perturbation is applied to
    the old cell's rank-value tokens, and the prompt
    ``<bos> perturbed_old <eos> <bos> young <eos>`` is written to a HF dataset.
    A ``manifest.csv`` records per-row ``target_start`` / ``target_end``
    indices for :func:`score_screen` to gather log-probs over.

    Args:
        tokenizer: MaxToki tokenizer.
        pairs: Iterable of :class:`CellPair` (young anchor + old cell to perturb).
        specs: Iterable of :class:`PerturbationSpec` to apply to every pair.
        output_dir: Directory to write the HF dataset and ``manifest.csv`` to.
        model_input_size: Max sequence length (prompts over this are skipped or raise).
        skip_too_long: If True, silently drop oversized prompts; otherwise raise.

    Returns:
        ``(dataset_path, manifest_path)``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = list(specs)
    if not specs:
        raise ValueError("At least one PerturbationSpec is required")

    input_ids: list[list[int]] = []
    manifest_rows: list[dict[str, object]] = []
    skipped = 0
    pair_idx = -1

    for pair_idx, pair in enumerate(pairs):
        young_rt = strip_bos_eos(pair.young_tokens, tokenizer)
        old_rt = strip_bos_eos(pair.old_tokens, tokenizer)

        for spec in specs:
            perturbed = spec.op(old_rt)
            sp = build_scoring_prompt(tokenizer, context_cell=perturbed, target_cell=young_rt)
            if len(sp.input_ids) > model_input_size:
                if skip_too_long:
                    skipped += 1
                    continue
                raise ValueError(
                    f"Prompt for pair={pair.cell_id} spec={spec.name} has length "
                    f"{len(sp.input_ids)} > model_input_size={model_input_size}"
                )

            row_idx = len(input_ids)
            input_ids.append(sp.input_ids)
            manifest_rows.append({
                "row": row_idx,
                "pair_idx": pair_idx,
                "cell_id": pair.cell_id,
                "spec_name": spec.name,
                "prompt_length": len(sp.input_ids),
                "target_start": sp.target_start,
                "target_end": sp.target_end,
                **spec.metadata,
                **pair.metadata,
            })

    if not input_ids:
        raise ValueError("No prompts produced — all pairs were too long or the input iterable was empty.")

    ds = datasets.Dataset.from_dict({"input_ids": input_ids})
    dataset_path = out / "dataset"
    ds.save_to_disk(str(dataset_path))

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    meta_path = out / "screen_meta.json"
    meta_path.write_text(json.dumps({
        "readout": "log_likelihood",
        "n_prompts": len(input_ids),
        "n_pairs": pair_idx + 1 if input_ids else 0,
        "n_specs": len(specs),
        "skipped_too_long": skipped,
        "model_input_size": model_input_size,
    }, indent=2))
    return dataset_path, manifest_path


def score_logprob(
    logits,  # torch.Tensor of shape [seq_len, vocab_size] or [1, seq_len, vocab_size]
    input_ids,  # 1D list/tensor of token ids, length seq_len
    target_start: int,
    target_end: int,
) -> dict[str, float]:
    """Sum log-probabilities of the target span under an autoregressive LM.

    Given logits from a full forward pass over ``input_ids``, compute
    ``sum_{i in [target_start, target_end)} log_softmax(logits[i-1])[input_ids[i]]``.

    Args:
        logits: ``[seq_len, vocab_size]`` or ``[1, seq_len, vocab_size]`` logits.
        input_ids: The token sequence that produced ``logits``.
        target_start: First position to score (must be >= 1).
        target_end: One past last position to score.

    Returns:
        Dict with ``sum_logprob``, ``mean_logprob``, ``n_tokens``.
    """
    import torch

    if target_start < 1:
        raise ValueError("target_start must be >= 1 (position 0 has no preceding context)")
    if target_end <= target_start:
        raise ValueError(f"target_end ({target_end}) must be > target_start ({target_start})")

    logits_t = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    if logits_t.dim() == 3:
        logits_t = logits_t.squeeze(0)
    ids_t = input_ids if isinstance(input_ids, torch.Tensor) else torch.as_tensor(input_ids)

    # Predict token i from logits at position i-1.
    shifted = logits_t[target_start - 1 : target_end - 1]  # [n, V]
    targets = ids_t[target_start:target_end]  # [n]
    log_probs = torch.log_softmax(shifted.float(), dim=-1)
    gathered = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [n]

    n = int(targets.numel())
    s = float(gathered.sum().item())
    return {"sum_logprob": s, "mean_logprob": s / n, "n_tokens": n}


def score_screen(
    model,  # any HF-style causal LM: model(input_ids=...).logits
    dataset_path: str | Path,
    manifest_path: str | Path,
    device: str = "cuda",
    dtype=None,
) -> pd.DataFrame:
    """Score every prompt in a screen and return the manifest augmented with log-probs.

    Runs one forward pass per prompt (batch_size=1; prompts are variable-length
    and cheap to score one at a time at 217M scale). Computes
    ``log P(young | perturbed_old)`` on the target span defined in the manifest.

    Args:
        model: A HuggingFace-style causal LM that, when called with
            ``input_ids=<LongTensor[1, L]>``, returns an object with a
            ``.logits`` attribute of shape ``[1, L, V]``. The published
            ``MaxToki-217M-HF`` checkpoint loaded via
            ``transformers.AutoModelForCausalLM.from_pretrained`` fits this.
        dataset_path: Path passed to :func:`build_screen_dataset_scoring`
            (will be loaded via ``datasets.load_from_disk``).
        manifest_path: The ``manifest.csv`` produced by
            :func:`build_screen_dataset_scoring`.
        device: Device to run forwards on.
        dtype: Optional torch dtype to cast inputs to; usually unnecessary.

    Returns:
        Manifest DataFrame augmented with ``sum_logprob``, ``mean_logprob``,
        ``n_scored_tokens`` columns, plus ``delta_vs_baseline`` computed per
        ``pair_idx`` against the ``baseline`` spec.
    """
    import torch

    ds = datasets.load_from_disk(str(dataset_path))
    manifest = pd.read_csv(manifest_path)
    if len(ds) != len(manifest):
        raise ValueError(f"Dataset ({len(ds)}) and manifest ({len(manifest)}) lengths disagree")

    rows = []
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            for i, sample in enumerate(ds):
                ids = torch.as_tensor(sample["input_ids"], device=device).unsqueeze(0)
                if dtype is not None:
                    ids = ids.to(dtype=torch.long)
                out = model(input_ids=ids)
                logits = out.logits if hasattr(out, "logits") else out[0]
                m_row = manifest.iloc[i]
                rows.append(score_logprob(
                    logits=logits[0],
                    input_ids=ids[0],
                    target_start=int(m_row["target_start"]),
                    target_end=int(m_row["target_end"]),
                ))
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    scored = manifest.copy()
    scored["sum_logprob"] = [r["sum_logprob"] for r in rows]
    scored["mean_logprob"] = [r["mean_logprob"] for r in rows]
    scored["n_scored_tokens"] = [r["n_tokens"] for r in rows]
    baseline = scored[scored["spec_name"] == "baseline"].set_index("pair_idx")["mean_logprob"]
    if not baseline.empty:
        scored["delta_vs_baseline"] = scored["mean_logprob"] - scored["pair_idx"].map(baseline)
    return scored
