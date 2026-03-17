# VOC Analysis Pipeline for PTR-MS Data

## Overview

A standalone analytical pipeline for processing Volatile Organic Compound (VOC) data from Proton Transfer Reaction Mass Spectrometry (PTR-MS) instruments. This system performs comprehensive time-series analysis including temporal segmentation, anomaly detection, blank subtraction, replicate aggregation, and multivariate statistical analysis (PCA) for experimental VOC datasets.

## Key Features

- **Automated Time-Series Processing**: Temporal splitting with configurable acclimation and transition exclusion windows
- **Statistical Preprocessing**: Block-wise median calculation and robust anomaly removal
- **Experimental Design Support**: Multi-condition (thesis) comparison with replicate aggregation
- **Blank Correction**: Systematic blank subtraction for chemical background removal
- **Multivariate Analysis**: Principal Component Analysis (PCA) for VOC pattern recognition
- **Flexible Configuration**: External schedule files for reproducible experimental workflows

## Project Structure

```
analisi-vocs/
├── voc_analysis_pipeline.py    # Main pipeline orchestrator
├── ptrms_*.py                   # Modular PTR-MS processing functions
├── H0/                          # Condition/thesis H0 data
│   ├── Run2/
│   ├── Run4/
│   ├── Run5/
│   └── Run6/
├── H1/                          # Condition/thesis H1 data
│   └── [Run folders]
├── Blank/                       # Blank measurements
├── config/
│   ├── run_schedule.template.csv    # Experiment timing configuration
│   └── composti da monitorare.xlsx  # Optional: Target m/z list
└── output/
    ├── 01_comparison_raw_median/
    ├── 02_anomaly_removal/
    ├── 03_aggregation/
    ├── 04_blank_removal/
    ├── 05_final_comparison/
    ├── 06_pca_analysis/
    └── comparison_metrics.xlsx
```

## Technical Stack

- **Data Processing**: Python, Pandas, NumPy
- **Statistical Analysis**: Scikit-learn (PCA), SciPy
- **Visualization**: Matplotlib, Seaborn
- **Input Formats**: CSV/TXT (PTR-MS exports), Excel (configuration)

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

### 1. PTR-MS Data Organization

Place preprocessed PTR-MS exports in thesis-specific folders:

- `H0/` - Control or baseline condition
- `H1/` - Treatment or experimental condition
- `Blank/` - Blank measurements for background subtraction

**Supported Formats**: `.txt` or `.csv` (e.g., `chamber_01.h5_export.txt`)

**Recommended Naming**:
```
chamber_01.h5_export.txt
chamber_02.h5_export.txt
```

Data can be organized as:
- Flat structure (files directly in thesis folders)
- Nested structure (files in `Run2`, `Run4`, etc. subfolders)

### 2. Experimental Schedule Configuration (Optional)

Provide timing information via:
- `config/run_schedule.xlsx` (default)
- `config/run_schedule.csv`

**Required Columns**:

| Column | Description |
|--------|-------------|
| `run_id` | Unique run identifier |
| `start_datetime` | Run start timestamp |
| `split_hours` | Offset for temporal phase splitting (H0/H1 → H2/H3) |
| `acclim_hours` | Acclimation period duration |
| `transition_exclude_hours` | Exclusion window around phase transitions |

If absent, hardcoded defaults in `voc_analysis_pipeline.py` will be used.

**Template**: `config/run_schedule.template.csv`

### 3. Target Compound List (Optional)

Specify m/z values of interest in:
- `composti da monitorare.xlsx` (first column = m/z)

If not provided, all detected masses will be analyzed.

## Usage

```bash
python voc_analysis_pipeline.py
```

The pipeline executes sequentially:
1. **Temporal Splitting** → Extract acclimation and experimental phases
2. **Block Median Calculation** → Reduce noise via temporal aggregation
3. **Anomaly Removal** → Statistical outlier detection and filtering
4. **Replicate Aggregation** → Combine technical/biological replicates
5. **Blank Subtraction** → Remove chemical background
6. **Condition Comparison** → Statistical testing and visualization
7. **PCA Analysis** → Dimensionality reduction and pattern discovery

## Output Structure

| Directory | Contents |
|-----------|----------|
| `01_comparison_raw_median/` | Raw vs. median-filtered comparisons |
| `02_anomaly_removal/` | Anomaly detection results and plots |
| `03_aggregation/` | Replicate-aggregated time series |
| `04_blank_removal/` | Blank-subtracted VOC profiles |
| `05_final_comparison/` | Thesis comparison plots and statistics |
| `06_pca_analysis/` | PCA loadings, scores, and biplots |
| `comparison_metrics.xlsx` | Summary statistics and effect sizes |

## Use Cases

- **Chemical Ecology**: VOC profiling of plant-insect interactions
- **Environmental Monitoring**: Real-time air quality analysis
- **Food Science**: Aroma compound characterization
- **Clinical Research**: Breath analysis for disease biomarkers

## Research Profile

- **Keywords**: PTR-MS, VOC time series, anomaly removal, blank subtraction, PCA, reproducible analytics, chemical sensing
- **Domain**: Applied analytical chemistry and experimental data science
- **License**: Open-source for reproducible research and education

## Notes

Raw data files and large datasets have been removed from this repository for privacy and size constraints. This repository contains production-ready code, configuration templates, and directory structure for immediate deployment.

## Citation

If using this pipeline in academic work, please cite:
```
Bottino, A. (2026). VOC Analysis Pipeline for PTR-MS Data.
GitHub: https://github.com/Bottins/analisi-vocs
```

---

**Author**: Alessandro Bottino
**Last Updated**: March 2026
**Repository**: [github.com/Bottins/analisi-vocs](https://github.com/Bottins/analisi-vocs)
