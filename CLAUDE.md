# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

MaxToki is a temporal AI model for predicting drivers of cell state progression. It is a LLaMA decoder model trained in two stages: (1) autoregressive pretraining over rank-value-encoded single-cell transcriptomes (NextCell), and (2) temporal fine-tuning that adds a TimeBetweenCells MSE regression head and mixes it with cross-entropy via `MaxTokiLossWithReduction`. Training uses NeMo + Megatron-LM for distributed GPU execution; the model **requires CUDA** (TransformerEngine) and cannot run on CPU.

This is a uv-managed monorepo. `pyproject.toml` is a meta-package; actual code lives in `sub-packages/bionemo-*/` (workspace members). `3rdparty/Megatron-LM` is a git submodule — always clone with `--recursive` or run `git submodule update --init --recursive`. NeMo is installed from PyPI, not as a submodule.

## Development Environment

All dev/test/run happens inside the Docker container. Running outside the container is **not supported** — dependency versions (torch 2.3.*, TransformerEngine, Apex, Megatron-LM, flash-attn) are pinned there.

```bash
DOCKER_BUILDKIT=1 docker build --target dev -t maxtoki-dev -f Dockerfile .
# See README.md "Launch the container" for the full docker run invocation.
```

The container installs all `bionemo-*` sub-packages in editable mode via `.devcontainer/postCreateCommand.sh`, so source edits on the host are picked up live.

`justfile` targets (`just build-dev`, `just run-dev`, `just test`) wrap the NVIDIA-internal build/run flow and require an `.env` created by `internal/scripts/setup_env_file.sh` (sets `IMAGE_REPO`, `CACHE_TAG`, `JUPYTER_PORT`, local data/results/models paths, etc.). `just test` also requires a clean git tree. For external contributors the raw `docker build` / `docker run` from the README is the supported path.

## Commands

Run from inside the container unless noted.

**Tests** — the project uses pytest with coverage and nbval:
```bash
# Fast unit tests (skip @pytest.mark.slow, skip notebooks):
ci/scripts/run_pytest_unittests.sh
# Slow tests only:
ci/scripts/run_pytest_slow.sh
# Notebook tests:
ci/scripts/run_pytest_notebooks.sh
# Single test file or test:
pytest sub-packages/bionemo-maxtoki/tests/bionemo/maxtoki/test_train.py -v
pytest sub-packages/bionemo-maxtoki/tests/bionemo/maxtoki/test_train.py::test_name -v
```
`ci/scripts/pytest_runner.sh` iterates over each `sub-packages/bionemo-*/` directory, so running pytest at the repo root from scratch is not equivalent — it loops per sub-package and resets coverage/caches between them.

**Static checks** — `ci/scripts/static_checks.sh` runs all three of:
- `ruff check scripts/ sub-packages/` (line-length 119, google docstrings, see `[tool.ruff]` in `pyproject.toml`)
- `tach check` (enforces inter-package import boundaries)
- `pre-commit run --all-files` (ruff-format, end-of-file/trailing-whitespace, detect-secrets against two baselines `.secrets.baseline` and `.secrets-nb.baseline`, and a license-header check that will auto-insert the header from `license_header`)

## Architecture

### Training / inference entrypoints (all under `bionemo.maxtoki`)
- `train.py` — single entrypoint for both pretraining and temporal fine-tuning. `--pretrain` switches the datamodule to pretraining mode (no query structural tokens); `--use-finetuning-config` swaps `MaxTokiConfig` → `MaxTokiMultitaskFineTuneConfig` (attaches regression head + mixed loss) and is required for second-stage training. `--initial-ckpt-path` loads weights from a previous stage.
- `predict.py` — two modes: regression (`TimeBetweenCells`, default) and autoregressive generation (`--generate-next-cell`, uses KV cache sized by `--buffer-size-gb`). Output is sharded as `predictions__rank_*.pt`.
- `export_hf.py` / `import_hf.py` — NeMo ↔ HuggingFace checkpoint conversion. `import_hf` works by running one training step with lr=0 to materialize a NeMo checkpoint; `export_hf --sanity-check` compares logits and exits non-zero if MAE > 0.05.

### Configs and model classes (in `model.py` and `api.py`)
- `MaxTokiBaseConfig` (api.py) extends NeMo's `Llama32Config1B`.
- `MaxTokiConfig` (model.py) — pretraining config; selects the loss and model class.
- `MaxTokiMultitaskFineTuneConfig` (model.py) — temporal config; attaches regression head, uses `MaxTokiLossWithReduction` (CE + MSE with per-task masking, discussed below).
- `FinetuneLlamaModel` (model.py) — `MCoreGPTModel` subclass with the regression head.
- `MaxTokiLossWithReduction` (model.py) — mixed-loss objective. The TimeBetweenCells loss is only computed on tokens produced *after* `<eoq>`, and `--additive-penalty` penalizes the model for putting probability on non-numeric tokens at time-token positions.

### Data pipeline (`bionemo.maxtoki.data_prep`)
Three sequential stages, invoked as `python -m bionemo.maxtoki.data_prep {tokenize|assemble-paragraphs|assemble-queries}`:
1. **tokenize** — `.h5ad` → rank-value-encoded HF `datasets` directory. Resource defaults live in `bionemo/maxtoki/data_prep/resources/` (token dictionary, gene median, Ensembl mapping); override per-flag.
2. **assemble-paragraphs** — groups cells from the same trajectory (using `--time-group-columns`) into multi-cell sequences with interleaved time-lapse tokens. `--task-ratio` controls the NextCell/TimeBetweenCells mix.
3. **assemble-queries** — builds evaluation query datasets from paragraphs.

Datasets are HuggingFace `datasets` dirs (`load_from_disk`); each sample has a single `input_ids` field. The pretraining format is a single rank-value-encoded cell; the temporal format concatenates multiple cells with `<boq>`/`<eoq>` query structural tokens and numeric time tokens. `MaxTokiTokenizer` is mostly a pass-through — token IDs are precomputed by the data-prep pipeline.

### Attention backend gotcha
The 217M MaxToki model has `head_dim = 1232/8 = 154`, and `154 % 8 != 0`, which disqualifies TransformerEngine's Flash Attention kernel. TE silently falls back to an **unfused O(N²) attention** that needs ~33 GB/layer at `seq_length=16384` and will OOM on most GPUs. When `--use-finetuning-config` loads a checkpoint with an incompatible head dim it automatically switches to PyTorch SDPA (`sdpa_attention.py`); for generation, `--naive-benchmarking-only` disables KV caching entirely when the decode path can't be used.

### Checkpoints
Saved under `<result-dir>/<experiment-name>/dev/checkpoints/` as `epoch={e}-val_loss={v:.2f}-step={s}-consumed_samples={n}/`. Defaults: top-2 by `val_loss` + the last. `--val-check-interval` (default 10000) is clamped to `--num-steps` — for short dev runs, lower it or you only get one checkpoint at the very end. `--output-weights {tied,separate}` is baked into the checkpoint and inherited by all downstream stages.

## Sub-package boundaries (enforced by `tach`)

- `bionemo-core` — core utilities (data loaders, resamplers, resource loading).
- `bionemo-llm` — shared LLM primitives (Lightning module, LR schedulers, callbacks) used by MaxToki.
- `bionemo-maxtoki` — the MaxToki model, data-prep, training, inference, and HF conversion. This is the main sub-package for model work.
- `bionemo-testing` — shared test helpers.
- `internal/infra-bionemo` — build/license tooling (used by pre-commit).

All live under a single `bionemo` namespace package (no `__init__.py` at the namespace root, only at sub-package roots).

## Useful references

- Full training/inference CLI reference with every flag: top-level `README.md` — keep it in sync when adding CLI args.
- Pretrained model: `https://huggingface.co/theodoris-lab/MaxToki`.
- Manuscript: linked from `README.md`.
