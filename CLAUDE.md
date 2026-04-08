# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python data analysis project for Soil Quality Index (SQI) analysis. It uses machine learning (Random Forest regression) to identify key factors affecting soil quality across different regions (Guangxi and Jiangxi).

## Running the Analysis

```bash
python soil_analysis.py
```

**Prerequisites**: Place `cleaned_data.xlsx` at the hardcoded path `C:/Users/Greal1sh/Desktop/cleaned_data.xlsx` before running. The script expects specific columns: `treatment`, `pH`, `CEC`, `Ex.Ca`, `Ex.Mg`, `AP`, `AK`, `pHBC`, `SQI`, and `region`.

## Dependencies

Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

Python environment uses Conda (see `.vscode/settings.json`).

## Architecture

The analysis follows a linear workflow through `main()`:

1. **Data Loading** (`load_and_preprocess_data`): Reads Excel, checks missing values, encodes `treatment` categorical variable using LabelEncoder
2. **Region Split** (`split_by_region`): Separates data by `region` column (广西/江西)
3. **Model Training** (`train_random_forest`): Trains independent Random Forest regressors (100 estimators) for each region
4. **Feature Importance**: Exports `importance_guangxi.csv` and `importance_jiangxi.csv` to Desktop
5. **Treatment Ranking** (`analyze_treatment_ranking`): Groups by `treatment`, ranks by mean SQI
6. **Report Generation** (`generate_summary_report`): Writes `analysis_report.txt` to Desktop with regional comparison

## Key Implementation Notes

- **Output location**: All results (CSV reports, text summary) are written to `C:/Users/Greal1sh/Desktop/`. Update hardcoded paths in `main()` if working elsewhere.
- **Chinese font support**: Configured for Windows (SimHei, Microsoft YaHei). May need adjustment for Mac/Linux.
- **Feature columns**: `treatment_encoded`, `pH`, `CEC`, `Ex.Ca`, `Ex.Mg`, `AP`, `AK`, `pHBC`
- **Target variable**: `SQI`
