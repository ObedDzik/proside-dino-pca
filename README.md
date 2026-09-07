# proside-dino-pca

Prostate cancer detection from transrectal micro-ultrasound, built on a
frozen [DINOv3](https://github.com/facebookresearch/dinov3) backbone. This
compares two decoder paths for turning DINOv3 features into a per-pixel
cancer heatmap:

- **Proside decoder** (`dino_pca/train.py`) — the original SAM-style
  mask/class decoder.
- **UNETR-CNN decoder** (`dino_pca/train_unetr_cnn.py`) — an alternative
  UNETR-style CNN decoder path.

Both support an optional clinical-metadata "prompts" conditioning signal
(patient age, PSA, approximate PSA density) fed into the decoder as a
sparse embedding.

This repo is a standalone extraction of the `dino_pca` project from a
larger private research monorepo (`medAI`). It vendors only the subset of
that monorepo's shared library actually needed to run `dino_pca` — see
"What's vendored" below.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

DINOv3 itself is not vendored here — clone it separately and point
`DINOV3_LIBRARY_PATH` at it:

```bash
git clone https://github.com/ObedDzik/dinov3 /path/to/your/dinov3/clone
```

(Install DINOv3's own dependencies per its README.)

## Environment variables

Every run needs these set (see `debug.sh` / `run_sbatch.sh` / `run_*.sh`
for a full example):

| Variable | Purpose |
|---|---|
| `NCT_RAW_DATA_DIR` | Root directory of the NCT2013 ultrasound cohort |
| `NCT_METADATA_PATH` | Path to that cohort's `metadata.csv` |
| `EXACTVU_PCA_DATA_ROOT` | Root of the broader ExactVu PCa dataset store (also used for `data.root_dir` in the configs) |
| `MEDSAM_CHECKPOINT_DIR` | Directory holding SAM/MedSAM checkpoints |
| `MEDSAM_CHECKPOINT` | Path to the specific MedSAM checkpoint used to init the mask decoder |
| `DINOV3_LIBRARY_PATH` | Path to your local DINOv3 clone (see above) |
| `DINOV3_CHECKPOINTS_PATH` | Directory holding DINOv3 backbone checkpoints |
| `CHECKPOINT` | Base directory this run's checkpoints are written under |
| `JOB_ID` | Any run identifier (SLURM sets this automatically via `$SLURM_JOB_ID`; set it yourself for interactive runs — it's interpolated into the checkpoint path) |
| `WANDB_API_KEY` | Your own Weights & Biases API key (`wandb login` also works) |

`dino_pca/cfg.yaml`'s `data.root_dir` is currently a hardcoded path
(`/datasets/exactvu_pca/OPTIMUM/...`) from the original cluster — update it
(or override via CLI, see below) to point at your own copy of the dataset.

## Running

All four entries below are launched from the repo root. `run_sbatch.sh`,
`debug.sh`, `run_prompts.sh`, `run_unetrcnn_noprompt.sh`, and
`run_unetrcnn_prompts.sh` are SLURM/Compute-Canada reference scripts (they
still have `#SBATCH` headers and `module load` lines specific to that
cluster) — adapt the environment setup at the top to your own
infrastructure, the `python -m ...` invocation at the bottom is what
matters.

Configs are [OmegaConf](https://omegaconf.readthedocs.io/) YAML with CLI
dotlist overrides, e.g. `fold=0 model_kw.floating_point_prompts=[age,psa,approx_psa_density]`.

**Proside decoder, no prompts:**
```bash
python -m dino_pca.train -c dino_pca/cfg.yaml fold=0 name=dino_512_auxdec1024_0
```

**Proside decoder, with prompts:**
```bash
python -m dino_pca.train -c dino_pca/cfg.yaml \
  fold=0 name=dino_512_auxdec1024_prompts_0 \
  model_kw.floating_point_prompts=[age,psa,approx_psa_density]
```

**UNETR-CNN decoder, no prompts:**
```bash
python -m dino_pca.train_unetr_cnn -c dino_pca/cfg.yaml \
  fold=0 name=dino_512_auxdec1024_unetrcnn_noprompt_0
```

**UNETR-CNN decoder, with prompts:**
```bash
python -m dino_pca.train_unetr_cnn -c dino_pca/cfg.yaml \
  fold=0 name=dino_512_auxdec1024_unetrcnn_prompts_0 \
  model_kw.floating_point_prompts=[age,psa,approx_psa_density]
```

`dino_pca/cfg_256.yaml` is the same setup at 256px input resolution instead
of 512px.

## Layout

```
dino_pca/          entrypoints (train.py, train_unetr_cnn.py), dataloader, configs
medAI/             vendored subset of the shared medAI library (datasets, models, losses, etc.)
external_libs/     vendored SAM/MedSAM model code that medAI/modeling depends on
projects/medibot/  one vendored file medAI/modeling/prostnfound.py imports (unused dead import, kept so the module still loads)
```

### What's vendored, and what was trimmed

The upstream `medAI` package's `factories/__init__.py` and
`datasets/__init__.py` unconditionally import several subsystems dino_pca
never touches (self-supervised/iBOT pretraining evaluation, other cohorts'
dataset loaders). Those two `__init__.py` files were intentionally trimmed
here (see the comments at the top of each) to keep this repo's `medAI/`
subset focused on what `dino_pca` actually runs, rather than vendoring the
entire original monorepo. Everything else under `medAI/`, `external_libs/`,
and `projects/` is copied verbatim.
