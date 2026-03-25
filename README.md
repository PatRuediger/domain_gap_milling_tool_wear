# Characterizing Domain Gaps in Milling Tool Wear Prediction

**Authors:** Patrick Flore
**Affiliation:** RPTU Kaiserslautern-Landau, Leibniz Institut für Verbundwerkstoffe (IVW)

---

## Project Description

This repository contains research code for cross-dataset tool wear prediction in milling operations. The pipeline trains 1D CNN and LSTM models on vibration signal data and evaluates their transferability across three publicly available milling datasets: NASA Milling, PHM2010, and NATURE2024. Tool wear is framed as binary classification (healthy vs. worn) based on a configurable normalized wear threshold, enabling a direct comparison of domain gaps between datasets. The code supports three execution modes: standard base training within a single dataset, zero-shot inference where a model trained on one dataset is evaluated directly on another, and fine-tuning transfer learning where a pre-trained model is adapted to a new domain using a small target-domain training set.

---

## Installation

```bash
git clone https://github.com/PatRuediger/domain_gap_milling_tool_wear.git
cd tool-wear-domain-adaption
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# macOS GPU acceleration (optional):
pip install tensorflow-metal==1.2.0
```

---

## Dataset Acquisition

All three datasets must be downloaded separately and preprocessed into normalized Parquet files before running any experiments. The preprocessing normalizes the raw sensor signals and computes a normalized wear column used for label binarization.

| Dataset | Source | DOI / URL |
|---|---|---|
| **NASA Milling** | NASA Prognostics Data Repository | https://data.nasa.gov/dataset/milling-wear |
| **PHM2010** | IEEE DataPort | https://ieee-dataport.org/documents/2010-phm-society-conference-data-challenge (DOI: 10.21227/JDXD-YY51) |
| **NATURE2024** | Scientific Data (Nature) | https://doi.org/10.1038/s41597-025-04923-y |

After downloading and preprocessing, place the resulting Parquet files in a `data/` directory at the repository root.

---

## Expected Data Directory Structure

```
data/
├── nasa_milling_normalized.parquet
├── phm2007_milling_normalized.parquet
└── nature_milling_2025_normalized.parquet
```

---

## Usage

All experiments are launched via `run_all_tasks.py` with a YAML configuration file. The `mode` key in the config controls which execution path is taken.

**Base training** (train and evaluate within a single dataset):
```bash
python run_all_tasks.py configs/base_training.yaml
```

**Zero-shot inference** (evaluate a pre-trained model on a different dataset without any fine-tuning):
```bash
python run_all_tasks.py configs/zero_shot_inference.yaml
```

**Fine-tuning transfer learning** (adapt a pre-trained model to a new domain using a small training set):
```bash
python run_all_tasks.py configs/fine_tuning.yaml
```

---

## Key YAML Configuration Parameters

| Parameter | Description |
|---|---|
| `mode` | Execution mode: `train`, `inference`, or `transfer_learn` |
| `model_type` | Model architecture: `1d_conv`, `lstm`, or `autoencoder` |
| `task_type` | `classification` (default) or `regression` |
| `inputPath` | Path to the Parquet dataset file |
| `train_caseIDs` / `test_caseIDs` | Lists of integer case IDs for the train/test split |
| `signalColumns` | Signal channel(s) to use as model input, e.g. `[vib_spindle]` |
| `wearColumnName` / `wearTH` | Column name and threshold for binarizing wear labels |
| `signal_length` | Expected sequence length per sample (must match the data) |
| `output_parent_dir` | Root directory for saving trained models and results |
| `inference_tasks` | List of `{model_path, inputPath, test_caseIDs}` dicts for zero-shot evaluation |
| `transfer_learning_tasks` | List of `{model_path, inputPath, train_caseIDs, test_caseIDs}` dicts for fine-tuning |

See `configs/base_training.yaml` for a base training example, `configs/zero_shot_inference.yaml` for zero-shot inference, and `configs/fine_tuning.yaml` for transfer learning.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{flore2025domain,
  title={Characterizing Domain Gaps in Milling Tool Wear Prediction},
  author={Flore, ... and Simon, ... and Hussong, ... and Krenkel, ...},
  journal={...},
  year={2025}
}
```

> Please update with the final publication details.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
