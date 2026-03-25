# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research code for the paper "Characterizing Domain Gaps in Milling Tool Wear Prediction". Implements 1D CNN and LSTM pipelines for binary tool wear classification (healthy vs. worn) across three milling datasets (NASA, PHM2010, NATURE2024) with zero-shot inference and fine-tuning transfer learning.

## Running experiments

```bash
# Activate virtualenv first
source .venv/bin/activate

# Single training run
python run_all_tasks.py <config.yaml>

# Direct pipeline run (single task, no subprocess dispatch)
python run_new.py <config.yaml>
```

The `run_all_tasks.py` script dispatches to `run_new.py` in separate subprocesses — one per task — to avoid TensorFlow memory leaks between tasks. Use it for `inference` and `transfer_learn` modes that list multiple tasks.

There is no test suite.

## Architecture

### Entry point hierarchy

```
run_all_tasks.py   ← top-level dispatcher; reads mode, spawns subprocesses
    └── run_new.py ← single-task runner; instantiates the correct pipeline
            └── modelPipelinesTL.py ← current pipeline implementation
```

`run.py` and `modelPipelines.py` are older versions without transfer learning support and are superseded by the above.

### Pipeline class hierarchy (`modelPipelinesTL.py`)

```
ModelPipeline (ABC)
├── LSTMPipeline
├── Conv1DPipeline
└── AutoencoderPipeline
```

`ModelPipeline` provides: `loadData()`, `setWearTH()`, `reformatData()`, `prepare_data()`, `save_config()`, `evalModel()`, `evalModelRegression()`, and the three mode dispatchers in `run()`.

Key methods on concrete pipelines: `modelSetup()`, `trainModel()`, `create_transfer_head()`.

### Execution modes (set in YAML `mode:`)

- **`train`**: fit a model from scratch on `train_caseIDs`, evaluate on `test_caseIDs`
- **`inference`**: load a saved model, apply it to new-domain data (zero-shot)
- **`transfer_learn`**: load a saved model, freeze the feature extractor, train a new head on target-domain data

### Data

All datasets are preprocessed and stored as Parquet files with array-valued signal columns. Normalized versions live in `data/`:
- `nasa_milling_normalized.parquet`
- `phm2007_milling_normalized.parquet`
- `nature_milling_2025_normalized.parquet`

Each row is one measurement window. Key columns: signal array columns (e.g. `vib_spindle`), `wear_norm` (0–1 normalized wear value), `caseID`.

The pipeline creates binary labels from `wear_norm` using the `wearTH` threshold from the config.

### Config schema

All behavior is controlled by YAML. Key fields:

| Field | Purpose |
|---|---|
| `mode` | `train` / `inference` / `transfer_learn` |
| `model_type` | `lstm` / `1d_conv` / `autoencoder` |
| `task_type` | `classification` (default) / `regression` |
| `inputPath` | Path to training/scaler-fitting Parquet file |
| `train_caseIDs` / `test_caseIDs` | Lists of integer case IDs |
| `signalColumns` | List of signal column names to use |
| `wearColumnName` / `wearTH` | Wear label binarization |
| `signal_length` | Expected sequence length (must match data) |
| `output_parent_dir` | Root for saving models and results |
| `inference_tasks` | List of `{model_path, inputPath, test_caseIDs}` dicts |
| `transfer_learning_tasks` | List of `{model_path, inputPath, train_caseIDs, test_caseIDs}` dicts |

See `tl_inference_config.yaml` for a full annotated example covering all three modes.

### Outputs

Each run creates a timestamped subdirectory under `output_parent_dir/` containing the saved `.keras` model, a copy of the config YAML, metrics CSV, and plots.

## Dependencies

```bash
pip install -r requirements.txt
# On macOS with Apple Silicon, also:
pip install tensorflow-metal==1.2.0
```

Core stack: TensorFlow 2.18, scikit-learn, pandas, numpy, pyarrow, matplotlib, tqdm.

## Known issues / in-progress

- `modelPipelinesTL.py` contains `# FIX:` comments marking known rough edges
- `requirements.txt` includes `tensorflow-metal` which is macOS-only; remove when running on Linux/HPC
- Some YAML configs under `experiments/` contain hardcoded `/scratch/ruediger/` HPC paths
- See `CLEANUP_PLAN.md` for the full pre-publication cleanup checklist
