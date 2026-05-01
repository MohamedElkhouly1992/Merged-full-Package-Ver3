from pathlib import Path

from hvac_v3_engine import (
    BuildingSpec,
    HVACConfig,
    run_scenario_model,
    run_early_sensitivity_analysis,
    run_robustness_analysis,
)
from report_addons import build_detailed_tables, save_detailed_outputs, create_zip_from_folder


def main():
    out = Path("example_run")
    bldg = BuildingSpec(
        building_type="Example educational building",
        location="Synthetic weather example",
        conditioned_area_m2=1200.0,
        floors=3,
        n_spaces=18,
        occupancy_density_p_m2=0.08,
        lighting_w_m2=10.0,
        equipment_w_m2=8.0,
        airflow_m3h_m2=4.0,
        cooling_intensity_w_m2=100.0,
        heating_intensity_w_m2=55.0,
    )
    cfg = HVACConfig(
        years=1,
        hvac_system_type="Chiller_AHU",
        USE_HVAC_PRESET=True,
        USE_DEGRADATION=True,
        TIME_STEP_HOURS=24.0,
        APO_ITERS=4,
        APO_POP=8,
    )
    switches = {
        "sw_use_envelope": True,
        "sw_use_solar": True,
        "sw_use_infiltration": True,
        "sw_use_internal_gains": True,
        "sw_use_people_gains": True,
        "sw_use_lighting_gains": True,
        "sw_use_equipment_gains": True,
        "sw_use_hvac_fans": True,
        "sw_use_cooling": True,
        "sw_use_heating": True,
        "sw_use_degradation": True,
    }
    result = run_scenario_model(
        output_dir=out,
        axis_mode="baseline_scenario",
        bldg=bldg,
        cfg=cfg,
        weather_mode="synthetic",
        fixed_strategy="S2",
        fixed_severity="Moderate",
        fixed_climate="C0_Baseline",
        include_baseline_layer=True,
        include_baseline_as_scenario=True,
        parameter_switches=switches,
        time_step_hours=24.0,
    )
    tables = build_detailed_tables(out, bldg=bldg, cfg=cfg, zone_df=None)
    save_detailed_outputs(out, tables)

    sens_out = out / "sensitivity_robustness"
    run_early_sensitivity_analysis(
        sens_out,
        bldg=bldg,
        cfg=cfg,
        fixed_strategy="S2",
        fixed_severity="Moderate",
        fixed_climate="C0_Baseline",
        analysis_years=1,
        perturbation_pct=0.10,
        parameter_names=["COP_COOL_NOM", "airflow_m3h_m2", "lighting_w_m2"],
    )
    run_robustness_analysis(
        sens_out,
        bldg=bldg,
        cfg=cfg,
        fixed_strategy="S2",
        fixed_severity="Moderate",
        fixed_climate="C0_Baseline",
        analysis_years=1,
        n_samples=5,
        uncertainty_pct=0.10,
        parameter_names=["COP_COOL_NOM", "airflow_m3h_m2", "lighting_w_m2"],
    )

    zip_path = create_zip_from_folder(out)
    print("Example run completed.")
    print(result)
    print(f"ZIP: {zip_path}")


if __name__ == "__main__":
    main()
