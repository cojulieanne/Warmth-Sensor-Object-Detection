import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "4AAA - Multi-Temp Optuna TPE MLflow Optimization.ipynb"
OUTPUT = ROOT / "notebooks" / "AUX - Baseline-Centered Multi-Temp Optuna TPE MLflow Optimization.ipynb"


def replace(source, old, new, label):
    if old not in source:
        raise RuntimeError(f"Could not find replacement target: {label}")
    return source.replace(old, new)


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


nb = json.loads(SOURCE.read_text())
nb = deepcopy(nb)

title = """# AUX — Baseline-Centered Multi-Temperature Optuna TPE + MLflow Optimization

This auxiliary experiment repeats Notebook **4AAA** with one intentional preprocessing change: after smoothing, Primary and Secondary are centered on the first retained post-contact sample **within each trial** before the five ESN input channels are constructed.

For trial \(i\):

\[
P_i^\Delta(t)=P_i(t)-P_i(t_0),\qquad
S_i^\Delta(t)=S_i(t)-S_i(t_0).
\]

The five channels become \(P_i^\Delta\), \(S_i^\Delta\), \(P_i^\Delta-S_i^\Delta\), \(dP_i^\Delta/dt\), and \(dS_i^\Delta/dt\). A `StandardScaler` is still fitted **inside every LOMO fold using only the 13 training materials** and then applied to the held-out material.

Multiplication by \(10^6\) is deliberately not used for model input: a following `StandardScaler` would cancel any constant multiplier. The meaningful experimental change is removal of each trial's starting offset.

Everything else remains matched to 4AAA: one shared ESN–XGBoost configuration, three seeds, 14 material-held-out folds per Optuna candidate, pooled OOF scoring, multivariate TPE, and MLflow tracking.
"""
nb["cells"][0] = markdown(title)

# Give every output and MLflow record a separate AUX namespace.
config = "".join(nb["cells"][3]["source"])
config = replace(
    config,
    'RESULT_STEM = f"{temperature_tag}_{OPTIMIZATION_TARGET}_v7_optuna_tpe_mlflow"',
    'RESULT_STEM = f"{temperature_tag}_{OPTIMIZATION_TARGET}_aux_baseline_centered_optuna_tpe_mlflow"',
    "result stem",
)
config = replace(config, 'MLFLOW_DB_FILE = RESULTS_DIR / "4AAA_mlflow_tracking.db"',
                 'MLFLOW_DB_FILE = RESULTS_DIR / "AUX_baseline_centered_mlflow_tracking.db"',
                 "MLflow database")
config = replace(config, 'MLFLOW_ARTIFACT_DIR = RESULTS_DIR / "mlflow_artifacts"',
                 'MLFLOW_ARTIFACT_DIR = RESULTS_DIR / "aux_baseline_centered_mlflow_artifacts"',
                 "MLflow artifacts")
config = replace(config, 'MLFLOW_EXPERIMENT_NAME = "4AAA_multi_temp_unified_optuna_tpe"',
                 'MLFLOW_EXPERIMENT_NAME = "AUX_baseline_centered_multi_temp_optuna_tpe"',
                 "MLflow experiment")
config = replace(config, 'strftime("4AAA_%Y%m%dT%H%M%SZ")',
                 'strftime("AUX_baseline_centered_%Y%m%dT%H%M%SZ")',
                 "run name")
nb["cells"][3]["source"] = config.splitlines(keepends=True)

# Replace only the input representation. Fold structure and StandardScaler stay intact.
reservoir_cell = "".join(nb["cells"][12]["source"])
old_input = '''def raw_esn_input(trial):
    time = trial["time_from_contact"].to_numpy(float)
    primary = _smooth(trial["Primary"].to_numpy(float))
    secondary = _smooth(trial["Secondary"].to_numpy(float))
    difference = primary - secondary
    primary_rate = np.gradient(primary, time)
    secondary_rate = np.gradient(secondary, time)
    return np.column_stack([
        primary, secondary, difference, primary_rate, secondary_rate
    ])
'''
new_input = '''def raw_esn_input(trial):
    """Five post-contact channels after per-trial baseline centering.

    Centering uses only the trial's first retained sensor sample and never uses
    k, eff, another material, or a future target value. The fold-local
    StandardScaler in build_esn_feature_table remains the model scale transform.
    """
    time = trial["time_from_contact"].to_numpy(float)
    primary = _smooth(trial["Primary"].to_numpy(float))
    secondary = _smooth(trial["Secondary"].to_numpy(float))

    primary = primary - primary[0]
    secondary = secondary - secondary[0]
    difference = primary - secondary
    primary_rate = np.gradient(primary, time)
    secondary_rate = np.gradient(secondary, time)
    return np.column_stack([
        primary, secondary, difference, primary_rate, secondary_rate
    ])
'''
reservoir_cell = replace(reservoir_cell, old_input, new_input, "raw_esn_input")
nb["cells"][12]["source"] = reservoir_cell.splitlines(keepends=True)

# Make provenance explicit in MLflow and the JSON handoff.
optuna_cell = "".join(nb["cells"][18]["source"])
optuna_cell = replace(optuna_cell, '"notebook": "4AAA",',
                      '"notebook": "AUX_baseline_centered_4AAA",\n            "input_preprocessing": "per_trial_post_contact_baseline_centering_then_fold_standardization",',
                      "MLflow tags")
optuna_cell = replace(optuna_cell, '"material_folds": int(metadata["Sample"].nunique()),',
                      '"material_folds": int(metadata["Sample"].nunique()),\n            "input_preprocessing": "per_trial_baseline_centering_then_fold_StandardScaler",',
                      "MLflow params")
nb["cells"][18]["source"] = optuna_cell.splitlines(keepends=True)

output_cell = "".join(nb["cells"][19]["source"])
output_cell = replace(
    output_cell,
    '"source": "one shared configuration selected by pooled grouped-LOMO Optuna TPE search",',
    '"source": "AUX baseline-centered inputs; one shared configuration selected by pooled grouped-LOMO Optuna TPE search",\n    "input_preprocessing": {\n        "physical_channels": ["Primary", "Secondary"],\n        "baseline_definition": "first retained post-contact sample within each trial",\n        "baseline_centering": "channel(t) - channel(t0)",\n        "micro_unit_multiplier_used_for_model": False,\n        "engineered_channels": [\n            "baseline-centered smoothed Primary",\n            "baseline-centered smoothed Secondary",\n            "centered Primary minus centered Secondary",\n            "Primary derivative",\n            "Secondary derivative",\n        ],\n        "model_scaling": "StandardScaler fitted on training materials inside each LOMO fold",\n    },',
    "payload preprocessing",
)
nb["cells"][19]["source"] = output_cell.splitlines(keepends=True)

nb["cells"][11] = markdown("""# 5. Baseline-centered ESN reservoir inputs and compact trajectory summaries

Each temperature-specific trial becomes one row. Before reservoir processing, the smoothed Primary and Secondary trajectories are centered at their respective first retained post-contact values. The five resulting channels are still standardized using only the training portion of each LOMO fold.

Each reservoir unit contributes nine summaries: mean, standard deviation, range, net change, slope, absolute area, time of maximum absolute activation, skewness, and excess kurtosis. Numeric `Temperature_C` is included as the known operating condition.
""")

diagnostic_md = markdown("""## Verify the changed input representation

The first two channels and their difference must start at zero for every trial. Derivative channels need not start at zero. This check runs before the expensive Optuna search.
""")
diagnostic_code = code("""INPUT_CHANNEL_NAMES = (
    "Primary_delta", "Secondary_delta", "Difference_delta",
    "Primary_rate", "Secondary_rate",
)

diagnostic_rows = []
for trial_id, trial in aligned_trials.items():
    channels = raw_esn_input(trial)
    diagnostic_rows.append({
        "trial_id": trial_id,
        "initial_primary_delta": channels[0, 0],
        "initial_secondary_delta": channels[0, 1],
        "initial_difference_delta": channels[0, 2],
        "max_abs_primary_delta": np.max(np.abs(channels[:, 0])),
        "max_abs_secondary_delta": np.max(np.abs(channels[:, 1])),
    })

INPUT_DIAGNOSTICS = pd.DataFrame(diagnostic_rows)
if not np.allclose(
    INPUT_DIAGNOSTICS[[
        "initial_primary_delta", "initial_secondary_delta",
        "initial_difference_delta",
    ]].to_numpy(float),
    0.0,
):
    raise RuntimeError("Baseline-centered channels do not begin at zero.")

print("Baseline-centering check passed for", len(INPUT_DIAGNOSTICS), "trials.")
display(INPUT_DIAGNOSTICS.describe())

example_id = ALIGNMENT["trial_id"].iloc[0]
example_trial = aligned_trials[example_id]
example_time = example_trial["time_from_contact"].to_numpy(float)
example_channels = raw_esn_input(example_trial)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(example_time, example_channels[:, 0], label="Primary Δ")
axes[0].plot(example_time, example_channels[:, 1], label="Secondary Δ")
axes[0].plot(example_time, example_channels[:, 2], label="Primary Δ − Secondary Δ")
axes[0].set(title=f"Baseline-centered channels before fold scaling: {example_id}", ylabel="Sensor change")
axes[0].legend()
axes[1].plot(example_time, example_channels[:, 3], label="dPrimary/dt")
axes[1].plot(example_time, example_channels[:, 4], label="dSecondary/dt")
axes[1].set(xlabel="Time from contact (s)", ylabel="Rate of change")
axes[1].legend()
plt.tight_layout()
plt.show()
""")

# Insert diagnostics immediately after the reservoir/input definition.
nb["cells"][13:13] = [diagnostic_md, diagnostic_code]

# Update downstream headings after insertion and add a clear handoff warning.
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        text = "".join(cell["source"])
        text = text.replace("# 6. Leakage-safe metrics", "# 6. Leakage-safe metrics")
        text = text.replace("# 7. Fold-local", "# 7. Fold-local")
        text = text.replace("# 8. Unified", "# 8. Unified")
        text = text.replace("## Reading 4AAA outputs", "## Reading AUX outputs")
        if text.startswith("# 9. Handoff"):
            text = """# 9. AUX handoff

This notebook writes a separate `aux_baseline_centered_optuna_tpe_mlflow` parameter JSON. It does not overwrite the original 4AAA result.

Do **not** load these hyperparameters into the unmodified 4BB: the consumer must use the same baseline-centered `raw_esn_input` definition. Compare AUX pooled OOF performance with 4AAA before deciding whether to create a matching fixed-analysis notebook.
"""
        cell["source"] = text.splitlines(keepends=True)

# Remove inherited output and execution state.
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

nb.setdefault("metadata", {})["analysis"] = {
    "role": "auxiliary_baseline_centering_ablation",
    "source_notebook": "4AAA - Multi-Temp Optuna TPE MLflow Optimization.ipynb",
    "target": "eff",
    "validation": "shared_hyperparameters_14_fold_LOMO",
}

OUTPUT.write_text(json.dumps(nb, indent=1))
print(OUTPUT)
