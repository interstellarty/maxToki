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

import datasets
import numpy as np
import pandas as pd
import pytest

from bionemo.maxtoki.data_prep.dataset_utils import compose_example
from bionemo.maxtoki.perturb import (
    CellPair,
    ScoringPrompt,
    build_regression_prompt,
    build_scoring_prompt,
    build_screen_dataset,
    build_screen_dataset_scoring,
    knockout,
    load_screen_predictions,
    make_combo_spec,
    make_knockout_spec,
    make_noop_spec,
    make_overexpression_spec,
    overexpress,
    score_logprob,
    score_screen,
    strip_bos_eos,
    wrap_bos_eos,
)


# ---------------------------------------------------------------------------
# Low-level token ops
# ---------------------------------------------------------------------------


def test_strip_and_wrap_roundtrip(tokenizer):
    cell = [tokenizer.bos_id, 5, 6, 7, tokenizer.eos_id]
    rank = strip_bos_eos(cell, tokenizer)
    assert rank == [5, 6, 7]
    assert wrap_bos_eos(rank, tokenizer) == cell


def test_strip_without_markers_is_noop(tokenizer):
    assert strip_bos_eos([5, 6, 7], tokenizer) == [5, 6, 7]


def test_knockout_single(tokenizer):
    assert knockout([5, 6, 7], 6) == [5, 7]


def test_knockout_multiple_preserves_order(tokenizer):
    assert knockout([5, 6, 7, 8, 9], [6, 8]) == [5, 7, 9]


def test_knockout_missing_gene_is_silent(tokenizer):
    assert knockout([5, 6, 7], 99) == [5, 6, 7]


def test_overexpress_moves_to_rank_0(tokenizer):
    assert overexpress([5, 6, 7], 6) == [6, 5, 7]


def test_overexpress_inserts_absent_gene(tokenizer):
    assert overexpress([5, 6, 7], 99) == [99, 5, 6, 7]


def test_overexpress_multiple_preserves_given_order(tokenizer):
    assert overexpress([5, 6, 7, 8], [8, 5]) == [8, 5, 6, 7]


# ---------------------------------------------------------------------------
# Perturbation specs
# ---------------------------------------------------------------------------


def test_knockout_spec_uses_token_dict(tokenizer, synthetic_token_dictionary):
    spec = make_knockout_spec(tokenizer, "GENE3")
    g3 = synthetic_token_dictionary["GENE3"]
    assert spec.op([g3, 99, 100]) == [99, 100]
    assert spec.name == "KO:GENE3"
    assert spec.metadata["kind"] == "knockout"


def test_overexpression_spec_promotes_to_top(tokenizer, synthetic_token_dictionary):
    spec = make_overexpression_spec(tokenizer, "GENE4")
    g4 = synthetic_token_dictionary["GENE4"]
    assert spec.op([1, 2, g4, 3])[0] == g4


def test_combo_spec_applies_ko_before_oe(tokenizer, synthetic_token_dictionary):
    spec = make_combo_spec(tokenizer, knockouts=["GENE3"], overexpressions=["GENE7"])
    g3 = synthetic_token_dictionary["GENE3"]
    g7 = synthetic_token_dictionary["GENE7"]
    out = spec.op([g3, 20, g7, 30])
    assert g3 not in out
    assert out[0] == g7


def test_missing_gene_raises(tokenizer):
    with pytest.raises(KeyError):
        make_knockout_spec(tokenizer, "NOT_A_GENE")


def test_noop_spec(tokenizer):
    spec = make_noop_spec()
    assert spec.op([1, 2, 3]) == [1, 2, 3]
    assert spec.name == "baseline"


# ---------------------------------------------------------------------------
# Prompt format
# ---------------------------------------------------------------------------


def test_regression_prompt_minimal_2cell_structure(tokenizer):
    """For N=2 cells, prompt = <bos>young<eos> <boq> <bos>old<eos> <eoq> <time0>."""
    young = [5, 6]
    old = [7, 8, 9]
    prompt = build_regression_prompt(
        tokenizer, context_cells=[young], context_timesteps=[], question_cell=old
    )
    expected = [
        tokenizer.bos_id, 5, 6, tokenizer.eos_id,
        tokenizer.boq_id,
        tokenizer.bos_id, 7, 8, 9, tokenizer.eos_id,
        tokenizer.eoq_id,
        tokenizer.numeric_tokens["0"],
    ]
    assert prompt == expected


def test_regression_prompt_3cell_includes_internal_timestep(tokenizer):
    c0, c1, c2 = [4], [5], [6]
    prompt = build_regression_prompt(
        tokenizer, context_cells=[c0, c1], context_timesteps=[3], question_cell=c2
    )
    assert tokenizer.numeric_tokens["3"] in prompt
    # Time token appears between c0 and c1 specifically.
    eos_positions = [i for i, t in enumerate(prompt) if t == tokenizer.eos_id]
    first_eos = eos_positions[0]
    assert prompt[first_eos + 1] == tokenizer.numeric_tokens["3"]


def test_regression_prompt_timestep_length_mismatch_raises(tokenizer):
    with pytest.raises(ValueError, match="context_timesteps"):
        build_regression_prompt(
            tokenizer, context_cells=[[1], [2]], context_timesteps=[3, 4], question_cell=[5]
        )


def test_regression_prompt_unknown_time_value_raises(tokenizer):
    # Synthetic dictionary only has time tokens 0..9
    with pytest.raises(KeyError, match="numeric token"):
        build_regression_prompt(
            tokenizer, context_cells=[[1]], context_timesteps=[], question_cell=[2],
            time_placeholder=42,
        )


def test_regression_prompt_matches_compose_example(tokenizer, synthetic_token_dictionary):
    """Our prompt must match what the training pipeline would build for the same input."""
    cell_a = [tokenizer.bos_id, 5, 6, tokenizer.eos_id]
    cell_b = [tokenizer.bos_id, 7, 8, 9, tokenizer.eos_id]
    fake_dataset = datasets.Dataset.from_list([
        {"input_ids": cell_a, "unfiltered_cell_indices": 0, "unfiltered_dataset_indices": 0},
        {"input_ids": cell_b, "unfiltered_cell_indices": 1, "unfiltered_dataset_indices": 0},
    ])

    # compose_example expects numeric token keys as ints (after convert_token_dictionary_keys).
    training_dict = {
        k: v for k, v in synthetic_token_dictionary.items() if not (isinstance(k, str) and k.isdigit())
    }
    for i in range(10):
        training_dict[i] = synthetic_token_dictionary[str(i)]

    # For N=2, is_train=False returns a 9-tuple; prompt = context + <boq> cell_b <eoq>.
    prompt_train, _ctx, _q, _resp, _rlen, _mrlen, _last_t, _ufi, _ufd = compose_example(
        dataset_i=fake_dataset,
        cell_indices_i=[0, 1],
        timesteps_i=[3],
        time_choices_i=[0, 3],
        token_dictionary=training_dict,
        predict_time_or_cell="time",
        is_train=False,
        model_input_size=4096,
        clip_first_cell=False,
    )

    our_prompt = build_regression_prompt(
        tokenizer,
        context_cells=[[5, 6]],
        context_timesteps=[],
        question_cell=[7, 8, 9],
        time_placeholder=0,
    )
    # compose_example's prompt stops at <eoq>; ours appends a numeric placeholder so the
    # multitask collator can classify the sample as TimeBetweenCells at inference time.
    assert our_prompt[:-1] == prompt_train


# ---------------------------------------------------------------------------
# Screen dataset builder
# ---------------------------------------------------------------------------


def _fake_pair(tokenizer, young_genes=(5, 6), old_genes=(7, 8, 9), cell_id="pair_0"):
    return CellPair(
        young_tokens=[tokenizer.bos_id, *young_genes, tokenizer.eos_id],
        old_tokens=[tokenizer.bos_id, *old_genes, tokenizer.eos_id],
        cell_id=cell_id,
        metadata={"tissue": "liver"},
    )


def test_build_screen_dataset_round_trip(tokenizer, tmp_path):
    pairs = [_fake_pair(tokenizer, cell_id="cell_A"), _fake_pair(tokenizer, cell_id="cell_B")]
    specs = [
        make_noop_spec(),
        make_knockout_spec(tokenizer, "GENE7"),
        make_overexpression_spec(tokenizer, "GENE4"),
    ]

    ds_path, manifest_path = build_screen_dataset(
        tokenizer, pairs, specs, output_dir=tmp_path,
    )

    ds = datasets.load_from_disk(str(ds_path))
    manifest = pd.read_csv(manifest_path)

    assert len(ds) == len(pairs) * len(specs) == 6
    assert len(manifest) == 6
    assert set(manifest["spec_name"]) == {"baseline", "KO:GENE7", "OE:GENE4"}
    assert list(manifest["row"]) == list(range(6))

    # Rows in manifest align positionally with dataset.
    for row in manifest.itertuples():
        sample = ds[int(row.row)]
        assert sample["input_ids"][-1] == tokenizer.numeric_tokens["0"]
        assert tokenizer.eoq_id in sample["input_ids"]
        if row.spec_name == "KO:GENE7":
            g7 = tokenizer.token_dictionary["GENE7"]
            # The old cell contains GENE7; after KO it must be absent.
            assert g7 not in sample["input_ids"]


def test_build_screen_dataset_skip_too_long_false_raises(tokenizer, tmp_path):
    pairs = [_fake_pair(tokenizer)]
    specs = [make_noop_spec()]
    with pytest.raises(ValueError, match="model_input_size"):
        build_screen_dataset(
            tokenizer, pairs, specs, output_dir=tmp_path,
            model_input_size=4, skip_too_long=False,
        )


def test_build_screen_dataset_all_skipped_raises(tokenizer, tmp_path):
    pairs = [_fake_pair(tokenizer)]
    specs = [make_noop_spec()]
    with pytest.raises(ValueError, match="No prompts"):
        build_screen_dataset(
            tokenizer, pairs, specs, output_dir=tmp_path,
            model_input_size=4, skip_too_long=True,
        )


# ---------------------------------------------------------------------------
# Predictions loader
# ---------------------------------------------------------------------------


def test_load_screen_predictions_joins_and_computes_delta(tokenizer, tmp_path):
    import torch

    pairs = [_fake_pair(tokenizer, cell_id="A"), _fake_pair(tokenizer, cell_id="B")]
    specs = [make_noop_spec(), make_knockout_spec(tokenizer, "GENE7")]
    _, manifest_path = build_screen_dataset(tokenizer, pairs, specs, output_dir=tmp_path)

    # Synthesize a predictions file: 4 samples, rank=0 only.
    preds = torch.tensor([[10.0, 6.0, 12.0, 5.0]])  # [1, B]
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    torch.save({"regression_preds": preds}, pred_dir / "predictions__rank_0.pt")

    df = load_screen_predictions(pred_dir, manifest_path)
    assert len(df) == 4
    assert "predicted_time_gap" in df.columns
    assert "delta_vs_baseline" in df.columns
    # For pair A (pair_idx=0) baseline=10; KO row delta should be 6 - 10 = -4.
    ko_a = df[(df["pair_idx"] == 0) & (df["spec_name"] == "KO:GENE7")]
    assert float(ko_a["delta_vs_baseline"].iloc[0]) == -4.0
    # Pair B (pair_idx=1): baseline=12, KO=5 → delta = -7.
    ko_b = df[(df["pair_idx"] == 1) & (df["spec_name"] == "KO:GENE7")]
    assert float(ko_b["delta_vs_baseline"].iloc[0]) == -7.0


def test_load_screen_predictions_row_count_mismatch_raises(tokenizer, tmp_path):
    import torch

    pairs = [_fake_pair(tokenizer)]
    specs = [make_noop_spec(), make_knockout_spec(tokenizer, "GENE7")]
    _, manifest_path = build_screen_dataset(tokenizer, pairs, specs, output_dir=tmp_path)

    # Manifest has 2 rows; supply 3 predictions to trigger the mismatch check.
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    torch.save({"regression_preds": torch.tensor([[1.0, 2.0, 3.0]])}, pred_dir / "predictions__rank_0.pt")

    with pytest.raises(ValueError, match="disagree"):
        load_screen_predictions(pred_dir, manifest_path)


# ---------------------------------------------------------------------------
# Scoring prompt format (Option C: log-likelihood readout)
# ---------------------------------------------------------------------------


def test_scoring_prompt_layout(tokenizer):
    sp = build_scoring_prompt(tokenizer, context_cell=[5, 6], target_cell=[7, 8, 9])
    bos, eos = tokenizer.bos_id, tokenizer.eos_id
    assert sp.input_ids == [bos, 5, 6, eos, bos, 7, 8, 9, eos]
    # target_start points at first gene token of target (skipping its <bos>)
    assert sp.input_ids[sp.target_start] == 7
    # target_end is exclusive, points at target's <eos>
    assert sp.input_ids[sp.target_end] == eos
    assert sp.input_ids[sp.target_start:sp.target_end] == [7, 8, 9]


def test_scoring_prompt_empty_context_still_valid(tokenizer):
    sp = build_scoring_prompt(tokenizer, context_cell=[], target_cell=[5])
    # <bos> <eos> <bos> 5 <eos>
    assert sp.input_ids == [tokenizer.bos_id, tokenizer.eos_id, tokenizer.bos_id, 5, tokenizer.eos_id]
    assert sp.input_ids[sp.target_start:sp.target_end] == [5]


# ---------------------------------------------------------------------------
# score_logprob
# ---------------------------------------------------------------------------


def test_score_logprob_sum_matches_manual():
    import torch

    # 4 tokens, vocab size 3. We score positions [2, 4) → score tokens 2 and 3.
    input_ids = [0, 1, 2, 1]
    logits = torch.tensor([
        [0.0, 0.0, 0.0],    # predicts input_ids[1]=1 from input_ids[0]=0
        [1.0, 2.0, 3.0],    # predicts input_ids[2]=2 from ...input_ids[1]
        [0.0, 5.0, 0.0],    # predicts input_ids[3]=1
        [0.0, 0.0, 0.0],    # unused (no token 4 to predict)
    ])
    out = score_logprob(logits, input_ids, target_start=2, target_end=4)
    # expected: log_softmax(logits[1])[2] + log_softmax(logits[2])[1]
    lp1 = torch.log_softmax(logits[1], dim=-1)[2].item()
    lp2 = torch.log_softmax(logits[2], dim=-1)[1].item()
    assert out["n_tokens"] == 2
    assert abs(out["sum_logprob"] - (lp1 + lp2)) < 1e-5
    assert abs(out["mean_logprob"] - (lp1 + lp2) / 2) < 1e-5


def test_score_logprob_accepts_batched_logits():
    import torch

    input_ids = [0, 1, 2]
    logits_2d = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])  # vocab=2, but ids have 2 → clip
    # Use a safer example: vocab=3, ids within range
    input_ids = [0, 1, 2]
    logits_2d = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    out_2d = score_logprob(logits_2d, input_ids, target_start=1, target_end=3)
    out_3d = score_logprob(logits_2d.unsqueeze(0), input_ids, target_start=1, target_end=3)
    assert abs(out_2d["sum_logprob"] - out_3d["sum_logprob"]) < 1e-6


def test_score_logprob_rejects_target_start_zero():
    import torch

    with pytest.raises(ValueError, match="target_start"):
        score_logprob(torch.zeros(3, 3), [0, 1, 2], target_start=0, target_end=2)


def test_score_logprob_rejects_empty_target():
    import torch

    with pytest.raises(ValueError, match="target_end"):
        score_logprob(torch.zeros(3, 3), [0, 1, 2], target_start=2, target_end=2)


# ---------------------------------------------------------------------------
# build_screen_dataset_scoring
# ---------------------------------------------------------------------------


def test_build_screen_dataset_scoring_round_trip(tokenizer, tmp_path):
    pairs = [_fake_pair(tokenizer, cell_id="cell_A"), _fake_pair(tokenizer, cell_id="cell_B")]
    specs = [
        make_noop_spec(),
        make_knockout_spec(tokenizer, "GENE7"),
        make_overexpression_spec(tokenizer, "GENE4"),
    ]

    ds_path, manifest_path = build_screen_dataset_scoring(
        tokenizer, pairs, specs, output_dir=tmp_path,
    )

    ds = datasets.load_from_disk(str(ds_path))
    manifest = pd.read_csv(manifest_path)
    assert len(ds) == len(pairs) * len(specs) == 6
    assert len(manifest) == 6
    assert set(manifest.columns) >= {"target_start", "target_end", "spec_name", "pair_idx"}
    assert set(manifest["spec_name"]) == {"baseline", "KO:GENE7", "OE:GENE4"}

    # Every prompt must end with <eos> at target_end.
    for i, row in manifest.iterrows():
        sample = ds[int(row["row"])]
        assert sample["input_ids"][int(row["target_end"])] == tokenizer.eos_id
        assert sample["input_ids"][int(row["target_start"])] != tokenizer.bos_id
        # Baseline prompt = context=old, target=young = bos+old+eos+bos+young+eos.
        # For KO:GENE7 rows the context must no longer contain GENE7.
        if row["spec_name"] == "KO:GENE7":
            g7 = tokenizer.token_dictionary["GENE7"]
            # GENE7 is in the old cell (token 7 by fixture construction).
            context_span = sample["input_ids"][: int(row["target_start"]) - 1]  # up to target's <bos>
            assert g7 not in context_span


def test_build_screen_dataset_scoring_no_trailing_numeric(tokenizer, tmp_path):
    """Scoring prompts must NOT append a numeric placeholder — that's a regression-only artifact."""
    pairs = [_fake_pair(tokenizer)]
    specs = [make_noop_spec()]
    ds_path, _ = build_screen_dataset_scoring(tokenizer, pairs, specs, output_dir=tmp_path)
    ds = datasets.load_from_disk(str(ds_path))
    assert ds[0]["input_ids"][-1] == tokenizer.eos_id
    for num_id in tokenizer.numeric_tokens.values():
        assert num_id not in ds[0]["input_ids"]


def test_build_screen_dataset_scoring_skip_too_long(tokenizer, tmp_path):
    pairs = [_fake_pair(tokenizer)]
    specs = [make_noop_spec()]
    with pytest.raises(ValueError, match="No prompts"):
        build_screen_dataset_scoring(
            tokenizer, pairs, specs, output_dir=tmp_path,
            model_input_size=4, skip_too_long=True,
        )


# ---------------------------------------------------------------------------
# score_screen (integration, with a fake model)
# ---------------------------------------------------------------------------


class _FakeCausalLM:
    """Minimal HF-style model stub returning deterministic logits biased to favor specific tokens."""

    def __init__(self, vocab_size: int, favored_token: int, bias: float = 5.0):
        import torch

        self.vocab_size = vocab_size
        self.favored_token = favored_token
        self.bias = bias
        self.training = False
        self._torch = torch

    def eval(self):
        self.training = False
        return self

    def __call__(self, input_ids):
        torch = self._torch
        b, L = input_ids.shape
        logits = torch.zeros(b, L, self.vocab_size)
        logits[..., self.favored_token] = self.bias

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


def test_score_screen_populates_logprob_columns(tokenizer, tmp_path):
    pairs = [_fake_pair(tokenizer, cell_id="cell_A")]
    specs = [make_noop_spec(), make_knockout_spec(tokenizer, "GENE7")]
    ds_path, manifest_path = build_screen_dataset_scoring(tokenizer, pairs, specs, output_dir=tmp_path)

    # Favor GENE2 (id=6) — irrelevant token, just checks that scoring runs.
    fake = _FakeCausalLM(vocab_size=tokenizer.vocab_size, favored_token=6, bias=5.0)
    df = score_screen(fake, ds_path, manifest_path, device="cpu")

    assert len(df) == 2
    assert {"sum_logprob", "mean_logprob", "n_scored_tokens", "delta_vs_baseline"} <= set(df.columns)
    assert (df["n_scored_tokens"] > 0).all()
    # Baseline's delta is always 0 (it compares to itself).
    baseline_delta = df[df["spec_name"] == "baseline"]["delta_vs_baseline"].iloc[0]
    assert abs(baseline_delta) < 1e-9


def test_score_screen_higher_logprob_when_model_favors_target(tokenizer, tmp_path):
    """If the fake model heavily favors a token that appears in the target, the score should be higher
    than when the model favors an unrelated token."""
    pairs = [_fake_pair(tokenizer, young_genes=(7, 8, 9), cell_id="cell_A")]
    specs = [make_noop_spec()]
    ds_path, manifest_path = build_screen_dataset_scoring(tokenizer, pairs, specs, output_dir=tmp_path)

    # GENE5 id=9 IS in the young cell; GENE1 id=5 is NOT.
    fav = _FakeCausalLM(vocab_size=tokenizer.vocab_size, favored_token=9, bias=10.0)
    unfav = _FakeCausalLM(vocab_size=tokenizer.vocab_size, favored_token=5, bias=10.0)
    df_fav = score_screen(fav, ds_path, manifest_path, device="cpu")
    df_unfav = score_screen(unfav, ds_path, manifest_path, device="cpu")
    assert df_fav["sum_logprob"].iloc[0] > df_unfav["sum_logprob"].iloc[0]
