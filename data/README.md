# Data Directory

## Dataset Overview

The three normalized Parquet files required to run experiments are **not included in this repository** and must be downloaded from the original public sources and preprocessed using the provided notebooks. Place the resulting files in this `data/` directory before running any experiments.

---

## NASA Milling Dataset

- **Source:** https://data.nasa.gov/dataset/milling-wear
- **Expected file:** `nasa_milling_normalized.parquet`
- **Description:** CNC milling experiments from the UC Berkeley BEST Lab, hosted by the NASA Prognostics Center of Excellence. Contains 16 cases of milling operations on cast iron and steel workpieces with varying feed rate and depth of cut. Sensor channels include vibration (spindle and table), acoustic emission, and spindle motor current recorded at 250 Hz. Tool wear is measured as flank wear (VB) after each cut.

---

## PHM 2010 Dataset

- **Source:** https://ieee-dataport.org/documents/2010-phm-society-conference-data-challenge
- **DOI:** 10.21227/JDXD-YY51
- **Expected file:** `phm2007_milling_normalized.parquet`
- **Description:** Dataset from the 2010 PHM Society Data Challenge. Milling experiments on Inconel 718 using a 6 mm three-flute ball-nose tungsten carbide cutter on a Röders Tech RFM760 high-speed CNC machine. Seven sensor channels (3-axis cutting force, 3-axis vibration, AE-RMS) recorded simultaneously at 50 kHz. Tool wear labels (flank wear VB in 10^-3 mm) are provided for training cutters C1, C4, and C6.

---

## NATURE 2024 Dataset

- **Source:** https://doi.org/10.1038/s41597-025-04923-y
- **Expected file:** `nature_milling_2025_normalized.parquet`
- **Description:** Open milling dataset published in *Scientific Data* (2025). Provides raw time-series sensor recordings from milling operations with accompanying tool wear measurements, intended as a benchmark resource for data-driven tool condition monitoring research.

---

## Expected Directory Structure

```
data/
├── nasa_milling_normalized.parquet
├── phm2007_milling_normalized.parquet
└── nature_milling_2025_normalized.parquet
```

---

## Parquet File Format

Each Parquet file contains one row per measurement window (one cut or segment). Columns fall into two categories:

- **Metadata columns:** `caseID` (integer identifier for the tool/experiment run), `wear_norm` (normalized wear value in [0, 1]), and additional columns such as raw wear measurements and cut index.
- **Signal columns:** Array-valued columns (e.g. `vib_spindle`) containing the raw time-series window as a 1D array of floating-point values. The array length corresponds to the `signal_length` parameter in the YAML config.

The binary label `wear_class` (healthy vs. worn) is **not stored in the file** — it is derived at runtime by thresholding `wear_norm` at the `wearTH` value specified in the experiment config.

---

## Preprocessing

Scripts for converting the raw downloaded datasets into the normalized Parquet format above are provided in the dataset-specific notebook subdirectories:

| Dataset | Preprocessing notebook |
|---|---|
| NASA Milling | `nasa_notebooks/DataPreprocessing.ipynb` |
| PHM 2010 | `phm_notebooks/DataPreprocessing.ipynb` |
| NATURE 2024 | `nature_milling/DataAnalysis.ipynb` |
