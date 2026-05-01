# HVAC ROM-Degradation Suite

A deployable Streamlit research software package for reduced-order HVAC energy modelling with degradation, maintenance strategies, climate scenarios, validation sheets, early sensitivity ranking, robustness analysis, reporting, and CatBoost surrogate modelling.

## Main design principle

`hvac_v3_engine.py` remains the numerical authority. The Streamlit interface collects inputs, calls the engine, and post-processes outputs. The UI does not duplicate the HVAC energy, degradation, COP, fan, maintenance, or KPI equations.

## What is included

- `hvac_v3_engine.py` — core reduced-order HVAC degradation engine
- `streamlit_app.py` — Streamlit user interface
- `report_addons.py` — upload handling, validation, detailed sheets, zone tables, ZIP export
- `run_example.py` — fast local smoke test
- `requirements.txt` — dependencies
- `examples/sample_daily_weather.csv` — sample weather file
- `docs/flowchart.png` and `docs/flowchart.svg` — journal-ready flowchart

## New features in this version

### 1. Selectable calculation time step

The setup tab includes a selector for:

- Daily, 24 h
- 12-hour
- 6-hour
- 3-hour
- Hourly

The original model is recovered when Daily is selected. Sub-daily modes preserve the same reduced-order equations and scale time-dependent terms by the selected period:

- energy = power × selected period hours
- CO2 = energy × emission factor
- cost = energy × price + maintenance cost
- dust accumulation and fouling growth scale with period length
- linear/exponential degradation increments scale with period length

Weather is still normalized to a 365-day representative year. Sub-daily modes use the daily weather state within each day unless the engine is later extended with native hourly weather profiles.

### 2. Early benchmark sensitivity analysis

The new **Sensitivity & Robustness** tab runs a fast one-at-a-time screening analysis before or beside the full scenario matrix.

For each selected input parameter, the model runs:

- baseline case
- low case: parameter × (1 - perturbation)
- high case: parameter × (1 + perturbation)

The ranking metric is central elasticity:

```text
elasticity = (% KPI change) / (% input change)
```

KPIs evaluated:

- Total Energy MWh
- Total CO2 tonne
- Mean Degradation Index
- Mean Comfort Deviation C
- Total Cost USD

Outputs:

- `early_sensitivity_ranking.csv`
- `early_sensitivity_details.csv`
- `sensitivity_base_summary.csv`
- `figures/early_sensitivity_ranking.png`
- `early_sensitivity_metadata.json`

### 3. Robustness analysis

The robustness tool performs bounded Monte-Carlo perturbation of selected inputs and repeats the selected scenario.

Outputs:

- `robustness_samples.csv`
- `robustness_summary.csv`
- `figures/robustness_kpi_boxplot.png`
- `robustness_metadata.json`

The summary includes:

- mean
- standard deviation
- coefficient of variation
- 5th percentile
- median
- 95th percentile
- minimum
- maximum

### 4. Main UI structure

Tabs are ordered for research workflow:

1. Building Identity & Setup
2. Parameter Switches
3. Scenario Modeling
4. Sensitivity & Robustness
5. Extra UI Tools
6. KPI Charts
7. Surrogate Train / Predict
8. Exports
9. Guide

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

If Streamlit is not recognized:

```bash
python -m streamlit run streamlit_app.py
```

## Test the engine without Streamlit

```bash
python run_example.py
```

This creates `example_run/` and `example_run.zip`.

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload the extracted project files, not the ZIP file.
3. Make sure the repository root contains:

```text
streamlit_app.py
hvac_v3_engine.py
report_addons.py
requirements.txt
README.md
```

4. Open Streamlit Community Cloud.
5. Choose the GitHub repository.
6. Set the main file path to:

```text
streamlit_app.py
```

7. Select Python 3.10 or 3.11.
8. Deploy.

## Outputs from scenario modelling

The model keeps the original outputs:

- daily/time-step dataset CSV
- annual CSV
- summary CSV
- Excel report
- PDF report
- figures
- baseline no-degradation layer

It also adds detailed sheets:

- `fuel_breakdown.csv`
- `comfort.csv`
- `site_data.csv`
- `internal_gains.csv`
- `validation_template.csv`
- `validation_comparison.csv` when uploaded
- `benchmark_summary.csv`
- `zone_analysis.csv`
- `kpi_summary.csv`
- `detailed_outputs.xlsx`

## Notes for publication

Recommended methodology wording:

> A reduced-order HVAC degradation model was implemented with a separated numerical engine and graphical interface. The engine evaluates thermal loads, effective COP, fan power, degradation accumulation, maintenance actions, comfort deviation, energy, cost, and carbon emissions over a selectable time step. Early parameter sensitivity was evaluated by one-at-a-time central elasticity, while model robustness was assessed using bounded Monte-Carlo perturbation of uncertain inputs.

## Important limitation

Hourly and sub-daily modes currently scale the reduced-order daily weather state across sub-daily periods. This is suitable for time-step consistency, sensitivity, and control comparison, but it is not a substitute for full hourly building-energy simulation unless hourly weather/load profiles are explicitly added in a later model extension.
