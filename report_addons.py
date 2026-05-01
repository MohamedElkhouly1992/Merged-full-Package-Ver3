from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Weather upload utilities
# -----------------------------
def _read_csv_fallback(file_or_path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "ISO-8859-1"]
    last_error = None
    for enc in encodings:
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception as exc:  # pragma: no cover - keeps upload robust
            last_error = exc
    raise ValueError(f"Could not read CSV file. Last error: {last_error}")


def _infer_col(cols, candidates):
    normalized = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]
    for c in cols:
        low = str(c).strip().lower()
        for cand in candidates:
            if cand.strip().lower() in low:
                return c
    return None


def normalize_weather_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return engine-format 365-row daily weather dataframe."""
    if df is None or df.empty:
        raise ValueError("Weather dataframe is empty.")
    work = df[[c for c in df.columns if not str(c).startswith("Unnamed")]].copy()
    if {"day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"}.issubset(work.columns):
        out = work[["day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]].copy()
    else:
        date_col = _infer_col(work.columns, ["Date/Time", "date", "datetime", "timestamp", "time"])
        doy_col = _infer_col(work.columns, ["day_of_year", "doy", "day"])
        temp_col = _infer_col(work.columns, ["T_mean_C", "T_amb_C", "Outdoor Dry-Bulb Temperature", "DryBulb", "dry-bulb", "temperature", "temp"])
        tmax_col = _infer_col(work.columns, ["T_max_C", "max temperature", "temperature max", "Tmax"])
        rh_col = _infer_col(work.columns, ["RH_mean_pct", "RH_pct", "Relative Humidity", "humidity", "rh"])
        ghi_col = _infer_col(work.columns, ["GHI_mean_Wm2", "GHI_Wm2", "Global Solar Radiation", "Global Horizontal Solar", "solar", "ghi"])
        direct_col = _infer_col(work.columns, ["Direct Normal Solar", "Direct Solar", "DNI"])
        diffuse_col = _infer_col(work.columns, ["Diffuse Horizontal Solar", "Diffuse Solar", "DHI"])
        if temp_col is None:
            raise ValueError("Weather CSV must contain outdoor temperature column.")
        out = pd.DataFrame()
        if date_col is not None:
            dates = pd.to_datetime(work[date_col], errors="coerce")
            out["day_of_year"] = dates.dt.dayofyear
        elif doy_col is not None:
            out["day_of_year"] = pd.to_numeric(work[doy_col], errors="coerce")
        else:
            out["day_of_year"] = np.arange(1, len(work) + 1)
        out["T_mean_C"] = pd.to_numeric(work[temp_col], errors="coerce")
        out["T_max_C"] = pd.to_numeric(work[tmax_col], errors="coerce") if tmax_col is not None else out["T_mean_C"] + 5.0
        out["RH_mean_pct"] = pd.to_numeric(work[rh_col], errors="coerce") if rh_col is not None else 60.0
        if ghi_col is not None:
            out["GHI_mean_Wm2"] = pd.to_numeric(work[ghi_col], errors="coerce")
        else:
            solar = pd.Series(np.zeros(len(work)), index=work.index, dtype=float)
            if direct_col is not None:
                solar += pd.to_numeric(work[direct_col], errors="coerce").fillna(0.0)
            if diffuse_col is not None:
                solar += pd.to_numeric(work[diffuse_col], errors="coerce").fillna(0.0)
            out["GHI_mean_Wm2"] = solar
    for c in ["day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["day_of_year", "T_mean_C"]).copy()
    out["day_of_year"] = out["day_of_year"].astype(int)
    out = out[(out["day_of_year"] >= 1) & (out["day_of_year"] <= 366)].copy()
    out.loc[out["day_of_year"] > 365, "day_of_year"] = 365
    out = out.groupby("day_of_year", as_index=False).agg(
        T_mean_C=("T_mean_C", "mean"),
        T_max_C=("T_max_C", "mean"),
        RH_mean_pct=("RH_mean_pct", "mean"),
        GHI_mean_Wm2=("GHI_mean_Wm2", "mean"),
    ).sort_values("day_of_year")
    out = out.set_index("day_of_year").reindex(range(1, 366))
    out[["T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]] = out[["T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]].interpolate(limit_direction="both").ffill().bfill()
    out = out.reset_index().rename(columns={"index": "day_of_year"})
    if out.isna().any().any():
        raise ValueError("Weather normalization failed; missing values remain after interpolation.")
    return out[["day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]]


def read_epw_upload(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
    rows = []
    for line in content[8:]:
        parts = line.split(",")
        if len(parts) < 14:
            continue
        try:
            month = int(float(parts[1])); day = int(float(parts[2])); hour = int(float(parts[3]))
            dry = float(parts[6]); rh = float(parts[8]); ghi = float(parts[13])
            ts = pd.Timestamp(year=2001, month=month, day=day, hour=max(0, min(hour - 1, 23)))
            rows.append({"Date/Time": ts, "T_amb_C": dry, "RH_pct": rh, "GHI_Wm2": ghi})
        except Exception:
            continue
    if not rows:
        raise ValueError("No valid hourly rows parsed from EPW upload.")
    hourly = pd.DataFrame(rows)
    hourly = hourly[~((hourly["Date/Time"].dt.month == 2) & (hourly["Date/Time"].dt.day == 29))].copy()
    daily = hourly.groupby(hourly["Date/Time"].dt.dayofyear, as_index=True).agg(
        T_mean_C=("T_amb_C", "mean"),
        T_max_C=("T_amb_C", "max"),
        RH_mean_pct=("RH_pct", "mean"),
        GHI_mean_Wm2=("GHI_Wm2", "mean"),
    ).reset_index().rename(columns={"Date/Time": "day_of_year"})
    if "day_of_year" not in daily.columns:
        daily = daily.rename(columns={daily.columns[0]: "day_of_year"})
    return normalize_weather_df(daily)


def read_weather_upload(uploaded_file) -> pd.DataFrame:
    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith(".epw") or name.endswith(".txt"):
        try:
            return read_epw_upload(uploaded_file)
        except Exception:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            return normalize_weather_df(_read_csv_fallback(uploaded_file))
    return normalize_weather_df(_read_csv_fallback(uploaded_file))


# -----------------------------
# Detailed output sheets
# -----------------------------
def find_result_paths(folder: str | Path) -> Dict[str, Path]:
    folder = Path(folder)
    summary_candidates = [
        folder / "baseline_scenario_summary.csv",
        folder / "three_axis_summary.csv",
        folder / "matrix_summary.csv",
        folder / "one_axis_strategy_summary.csv",
        folder / "one_axis_severity_summary.csv",
        folder / "baseline_no_degradation_summary.csv",
    ]
    annual_candidates = [
        folder / "annual_baseline_scenario.csv",
        folder / "annual_three_axis.csv",
        folder / "annual_matrix.csv",
        folder / "annual_one_axis_strategy.csv",
        folder / "annual_one_axis_severity.csv",
        folder / "baseline_no_degradation_annual.csv",
    ]
    daily_candidates = [
        folder / "baseline_scenario_ml_dataset.csv",
        folder / "three_axis_ml_dataset.csv",
        folder / "matrix_ml_dataset.csv",
        folder / "one_axis_strategy_ml_dataset.csv",
        folder / "one_axis_severity_ml_dataset.csv",
        folder / "baseline_no_degradation_daily.csv",
    ]
    def first_existing(cands):
        for p in cands:
            if p.exists():
                return p
        return cands[0]
    return {"summary": first_existing(summary_candidates), "annual": first_existing(annual_candidates), "daily": first_existing(daily_candidates)}


def _read_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_kpi_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "Total Energy MWh", "Total Cost USD", "Total CO2 tonne", "Mean COP", "Mean Degradation Index", "Mean Comfort Deviation C", "Occupied Discomfort Days"] if c in summary_df.columns]
    return summary_df[cols].copy()


def build_fuel_breakdown(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    df = daily_df.copy()
    mode = df.get("mode", pd.Series("unknown", index=df.index)).astype(str)
    df["cooling_electric_kwh_day"] = np.where(mode.eq("cooling"), df.get("energy_kwh_day", 0), 0.0)
    df["heating_equivalent_kwh_day"] = np.where(mode.eq("heating"), df.get("energy_kwh_day", 0), 0.0)
    df["fan_aux_electric_kwh_day"] = df.get("energy_kwh_day", 0) * 0.0
    keys = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "year"] if c in df.columns]
    if not keys:
        keys = ["year"] if "year" in df.columns else []
    out = df.groupby(keys, as_index=False).agg(
        total_energy_kwh=("energy_kwh_day", "sum"),
        cooling_electric_kwh=("cooling_electric_kwh_day", "sum"),
        heating_equivalent_kwh=("heating_equivalent_kwh_day", "sum"),
        co2_kg=("co2_kg_day", "sum"),
        cost_usd=("cost_usd_day", "sum"),
    )
    return out


def build_comfort_table(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "day", "year", "day_of_year", "occ", "T_sp_C", "T_amb_C", "RH_mean_pct", "comfort_dev_C", "occupied_discomfort_flag"] if c in daily_df.columns]
    return daily_df[cols].copy()


def build_site_data(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["scenario_combo_3axis", "day", "year", "day_of_year", "T_amb_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2", "occ"] if c in daily_df.columns]
    return daily_df[cols].drop_duplicates().copy()


def build_internal_gains(daily_df: pd.DataFrame, bldg=None) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    df = daily_df.copy()
    area = getattr(bldg, "conditioned_area_m2", float(df.get("area_m2", pd.Series([0])).iloc[0] if "area_m2" in df else 0))
    occ_density = getattr(bldg, "occupancy_density_p_m2", 0.0)
    lighting = getattr(bldg, "lighting_w_m2", 0.0)
    equipment = getattr(bldg, "equipment_w_m2", 0.0)
    sensible = getattr(bldg, "sensible_w_per_person", 0.0)
    occ = df.get("occ", 0.0)
    df["people_gain_kw"] = area * occ_density * sensible * occ / 1000.0
    df["lighting_gain_kw"] = area * lighting * occ / 1000.0
    df["equipment_gain_kw"] = area * equipment * occ / 1000.0
    df["total_internal_gain_kw"] = df["people_gain_kw"] + df["lighting_gain_kw"] + df["equipment_gain_kw"]
    cols = [c for c in ["scenario_combo_3axis", "day", "year", "day_of_year", "occ", "people_gain_kw", "lighting_gain_kw", "equipment_gain_kw", "total_internal_gain_kw"] if c in df.columns]
    return df[cols].copy()


def build_benchmark_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "Total Energy MWh" not in summary_df.columns:
        return pd.DataFrame()
    df = summary_df.copy()
    base = None
    if "scenario_combo_3axis" in df.columns:
        baseline_rows = df[df["scenario_combo_3axis"].astype(str).str.contains("BASELINE", case=False, na=False)]
        if not baseline_rows.empty:
            base = float(baseline_rows.iloc[0]["Total Energy MWh"])
    if base is None:
        base = float(df["Total Energy MWh"].iloc[0])
    df["energy_delta_MWh"] = df["Total Energy MWh"] - base
    df["energy_delta_pct"] = 100.0 * df["energy_delta_MWh"] / max(abs(base), 1e-9)
    if "Mean Degradation Index" in df.columns:
        df["degradation_delta"] = df["Mean Degradation Index"] - float(df["Mean Degradation Index"].iloc[0])
    keep = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "Total Energy MWh", "energy_delta_MWh", "energy_delta_pct", "Mean Degradation Index", "degradation_delta", "Mean Comfort Deviation C", "Total CO2 tonne"] if c in df.columns]
    return df[keep]


def build_zone_analysis(daily_df: pd.DataFrame, zone_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if daily_df.empty or zone_df is None or len(zone_df) == 0:
        return pd.DataFrame()
    z = zone_df.copy()
    required = {"zone_name", "area_m2", "occ_density"}
    if not required.issubset(z.columns):
        return pd.DataFrame()
    z["area_m2"] = pd.to_numeric(z["area_m2"], errors="coerce").fillna(0.0)
    z["occ_density"] = pd.to_numeric(z["occ_density"], errors="coerce").fillna(0.0)
    weights = z["area_m2"] * np.maximum(z["occ_density"], 0.01)
    if float(weights.sum()) <= 0:
        weights = z["area_m2"]
    if float(weights.sum()) <= 0:
        weights = pd.Series(np.ones(len(z)), index=z.index)
    weights = weights / weights.sum()
    summaries = []
    keys = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "year"] if c in daily_df.columns]
    grouped = daily_df.groupby(keys, dropna=False) if keys else [((), daily_df)]
    for group_key, grp in grouped:
        base = dict(zip(keys, group_key if isinstance(group_key, tuple) else (group_key,))) if keys else {}
        energy = float(grp.get("energy_kwh_day", pd.Series([0])).sum())
        co2 = float(grp.get("co2_kg_day", pd.Series([0])).sum())
        comfort = float(grp.get("comfort_dev_C", pd.Series([0])).mean())
        deg = float(grp.get("delta", pd.Series([0])).mean())
        for idx, row in z.iterrows():
            rec = dict(base)
            mult = float(weights.loc[idx])
            rec.update({
                "zone_name": row.get("zone_name", f"Zone_{idx+1}"),
                "zone_type": row.get("zone_type", "Custom"),
                "zone_area_m2": float(row.get("area_m2", 0)),
                "zone_occ_density": float(row.get("occ_density", 0)),
                "zone_energy_kwh": energy * mult,
                "zone_co2_kg": co2 * mult,
                "zone_comfort_dev_C": comfort,
                "zone_degradation_index": deg,
            })
            summaries.append(rec)
    return pd.DataFrame(summaries)


def build_validation_template(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        scenarios = ["Example_Scenario"]
    else:
        scenarios = summary_df.get("scenario_combo_3axis", pd.Series(range(len(summary_df)))).astype(str).tolist()
    return pd.DataFrame({
        "scenario_combo_3axis": scenarios,
        "reference_energy_MWh": np.nan,
        "reference_co2_tonne": np.nan,
        "reference_comfort_dev_C": np.nan,
        "reference_degradation_index": np.nan,
        "source_note": "DesignBuilder / EnergyPlus / measured / published reference",
    })


def load_validation_file(file_or_path) -> pd.DataFrame:
    return _read_csv_fallback(file_or_path)


def build_validation_comparison(summary_df: pd.DataFrame, validation_df: pd.DataFrame, source_name: str = "validation") -> pd.DataFrame:
    if summary_df.empty or validation_df.empty:
        return pd.DataFrame()
    val = validation_df.copy()
    scen_val = _infer_col(val.columns, ["scenario_combo_3axis", "Scenario Key", "scenario", "case", "strategy"])
    model = summary_df.copy()
    scen_model = "scenario_combo_3axis" if "scenario_combo_3axis" in model.columns else None
    def pick(df, names):
        return _infer_col(df.columns, names)
    energy_ref = pick(val, ["reference_energy_MWh", "Total Energy MWh", "Energy MWh", "energy", "Energy Consumption (kWh)"])
    co2_ref = pick(val, ["reference_co2_tonne", "Total CO2 tonne", "CO2", "Carbon Footprint"])
    comfort_ref = pick(val, ["reference_comfort_dev_C", "Mean Comfort Deviation C", "comfort"])
    if scen_val and scen_model:
        merged = model.merge(val, left_on=scen_model, right_on=scen_val, how="left", suffixes=("_model", "_ref"))
    else:
        merged = pd.concat([model.reset_index(drop=True), val.reset_index(drop=True)], axis=1)
    rows = []
    for _, r in merged.iterrows():
        rec = {"source": source_name, "scenario_combo_3axis": r.get("scenario_combo_3axis", r.get(scen_val, "case"))}
        if energy_ref and "Total Energy MWh" in merged.columns:
            ref = pd.to_numeric(pd.Series([r.get(energy_ref)]), errors="coerce").iloc[0]
            mod = pd.to_numeric(pd.Series([r.get("Total Energy MWh")]), errors="coerce").iloc[0]
            rec.update({"model_energy_MWh": mod, "reference_energy_MWh": ref, "energy_error_pct": 100.0 * (mod - ref) / max(abs(ref), 1e-9) if pd.notna(ref) else np.nan})
        if co2_ref and "Total CO2 tonne" in merged.columns:
            ref = pd.to_numeric(pd.Series([r.get(co2_ref)]), errors="coerce").iloc[0]
            mod = pd.to_numeric(pd.Series([r.get("Total CO2 tonne")]), errors="coerce").iloc[0]
            rec.update({"model_co2_tonne": mod, "reference_co2_tonne": ref, "co2_error_pct": 100.0 * (mod - ref) / max(abs(ref), 1e-9) if pd.notna(ref) else np.nan})
        if comfort_ref and "Mean Comfort Deviation C" in merged.columns:
            ref = pd.to_numeric(pd.Series([r.get(comfort_ref)]), errors="coerce").iloc[0]
            mod = pd.to_numeric(pd.Series([r.get("Mean Comfort Deviation C")]), errors="coerce").iloc[0]
            rec.update({"model_comfort_dev_C": mod, "reference_comfort_dev_C": ref, "comfort_error_C": mod - ref if pd.notna(ref) else np.nan})
        rows.append(rec)
    return pd.DataFrame(rows)


def build_detailed_tables(folder: str | Path, bldg=None, cfg=None, zone_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    paths = find_result_paths(folder)
    summary_df = _read_if_exists(paths["summary"])
    annual_df = _read_if_exists(paths["annual"])
    daily_df = _read_if_exists(paths["daily"])
    return {
        "kpi_summary": build_kpi_summary(summary_df),
        "fuel_breakdown": build_fuel_breakdown(daily_df),
        "comfort": build_comfort_table(daily_df),
        "site_data": build_site_data(daily_df),
        "internal_gains": build_internal_gains(daily_df, bldg=bldg),
        "validation_template": build_validation_template(summary_df),
        "benchmark_summary": build_benchmark_summary(summary_df),
        "zone_analysis": build_zone_analysis(daily_df, zone_df),
        "summary_copy": summary_df,
        "annual_copy": annual_df,
    }


def save_detailed_outputs(folder: str | Path, tables: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            path = folder / f"{name}.csv"
            df.to_csv(path, index=False)
            saved[name] = str(path)
    xlsx = folder / "detailed_outputs.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.head(50000).to_excel(writer, sheet_name=name[:31], index=False)
    saved["detailed_outputs_excel"] = str(xlsx)
    return saved


def create_zip_from_folder(folder: str | Path) -> Path:
    folder = Path(folder)
    zip_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in folder.rglob("*"):
            if p.is_file() and p != zip_path:
                zf.write(p, arcname=str(p.relative_to(folder)))
    return zip_path


# -----------------------------
# Setup JSON helpers
# -----------------------------
def setup_to_json_bytes(data: Dict[str, Any]) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def setup_from_upload(uploaded_file) -> Dict[str, Any]:
    raw = uploaded_file.getvalue().decode("utf-8")
    return json.loads(raw)
