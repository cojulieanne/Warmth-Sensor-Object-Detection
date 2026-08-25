import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "4BB - Multi-Temp Fixed Unified ESN Analysis.ipynb"
OUTPUT = ROOT / "notebooks" / "AUX-BB - Baseline-Centered Fixed Unified ESN Analysis.ipynb"


def replace(text, old, new, label):
    if old not in text:
        raise RuntimeError(f"Missing replacement target: {label}")
    return text.replace(old, new)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


nb = deepcopy(json.loads(SOURCE.read_text()))

nb["cells"][0] = md("""# AUX-BB — Baseline-Centered Fixed Unified ESN Analysis

This notebook is the fixed-analysis consumer for **AUX — Baseline-Centered Multi-Temp Optuna TPE + MLflow Optimization**.

It performs no hyperparameter search. It loads the single shared ESN–XGBoost configuration selected by AUX and reproduces the exact input pipeline used during optimization:

1. smooth Primary and Secondary;
2. subtract each trial's first retained post-contact value;
3. construct five input channels;
4. fit `StandardScaler` using only the training portion of each outer fold;
5. generate three seed-specific ESN feature tables;
6. train three XGBoost regressors and average their predictions.

The same optimized hyperparameters are used in every fold, but each fold fits new models because its training data differ.
""")

config = "".join(nb["cells"][3]["source"])
config = replace(
    config,
    'PARAMETER_FILE = RESULTS_DIR / f"{temperature_tag}_{PARAMETER_TARGET}_v6_unified_parameters.json"',
    'PARAMETER_FILE = RESULTS_DIR / (\n    f"{temperature_tag}_{PARAMETER_TARGET}"\n    "_aux_baseline_centered_optuna_tpe_mlflow_parameters.json"\n)\nAUX_BB_RESULT_STEM = f"{temperature_tag}_{PARAMETER_TARGET}_aux_bb_fixed_analysis"',
    "AUX parameter file",
)
config = replace(config, 'raise FileNotFoundError(f"Missing {PARAMETER_FILE}\\nRun Notebook 4AA first.")',
                 'raise FileNotFoundError(f"Missing {PARAMETER_FILE}\\nRun the AUX optimization notebook first.")',
                 "missing file message")
validation_anchor = '''if SAVED_CONFIGURATION["target"] != PARAMETER_TARGET:
    raise ValueError("Saved target does not match PARAMETER_TARGET.")
'''
validation_new = validation_anchor + '''
preprocessing = SAVED_CONFIGURATION.get("input_preprocessing", {})
expected_centering = "channel(t) - channel(t0)"
if preprocessing.get("baseline_centering") != expected_centering:
    raise ValueError(
        "The parameter file is not from the baseline-centered AUX workflow. "
        f"Expected baseline_centering={expected_centering!r}."
    )
if preprocessing.get("micro_unit_multiplier_used_for_model") is not False:
    raise ValueError("Unexpected AUX model-input multiplier provenance.")
'''
config = replace(config, validation_anchor, validation_new, "provenance validation")
config += '\nprint("Input preprocessing:", preprocessing)\n'
nb["cells"][3]["source"] = config.splitlines(keepends=True)

# Match AUX exactly: smooth, then subtract each trial's first retained value.
reservoir = "".join(nb["cells"][14]["source"])
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
    """Reproduce the AUX baseline-centered five-channel input exactly."""
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
reservoir = replace(reservoir, old_input, new_input, "baseline-centered input")
nb["cells"][14]["source"] = reservoir.splitlines(keepends=True)

nb["cells"][13] = md("""# 5. AUX-matched ESN reservoir and trajectory summaries

The reservoir definition and trajectory summaries match AUX. The only intentional difference from original 4BB is the input representation: Primary and Secondary are centered at their first retained post-contact values before the five channels are constructed.
""")

nb["cells"][15] = md("""# 6. Visualize the AUX input-to-reservoir sequence

For descriptive visualization, the selected material is treated as held out and the scaler is fitted using all other materials. This does not feed the evaluation engine; every evaluation fold independently fits its own training-only scaler.
""")

vis = "".join(nb["cells"][16]["source"])
old_vis = '''    # Descriptive scaler only—never passed into the evaluation engine.
    visualization_scaler = StandardScaler().fit(np.vstack([
        raw_esn_input(trial) for trial in aligned_trials.values()
    ]))
'''
new_vis = '''    # LOMO-style descriptive scaler—never passed into the evaluation engine.
    selected_material = aligned_trials[trial_id]["Sample"].iloc[0]
    visualization_scaler = StandardScaler().fit(np.vstack([
        raw_esn_input(trial)
        for trial in aligned_trials.values()
        if trial["Sample"].iloc[0] != selected_material
    ]))
'''
vis = replace(vis, old_vis, new_vis, "LOMO-style visualization scaler")
nb["cells"][16]["source"] = vis.splitlines(keepends=True)

# Replace the four-panel graph with a five-panel causal diagnostic.
nb["cells"][17] = code('''INPUT_NAMES = (
    "Primary Δ", "Secondary Δ", "Primary Δ − Secondary Δ",
    "dPrimary/dt", "dSecondary/dt",
)


def plot_esn_dynamics(trial_id=SELECTED_TRIAL_ID, maximum_units=5):
    trial, state_time, scaled_inputs, states = compute_visualization_states(trial_id)
    full_time = trial["time_from_contact"].to_numpy(float)
    unscaled_inputs = raw_esn_input(trial)

    units = np.unique(np.linspace(
        0, states.shape[1] - 1, min(maximum_units, states.shape[1]), dtype=int
    ))
    fig = plt.figure(figsize=(14, 15))
    grid = fig.add_gridspec(5, 1, height_ratios=[1.0, 1.3, 1.2, 1.8, 0.8], hspace=0.38)

    ax_delta = fig.add_subplot(grid[0])
    ax_delta.plot(full_time, unscaled_inputs[:, 0] * 1e6, label="Primary Δ", linewidth=1.8)
    ax_delta.plot(full_time, unscaled_inputs[:, 1] * 1e6, label="Secondary Δ", linewidth=1.8)
    ax_delta.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    ax_delta.set(title=f"AUX baseline-centered ESN evolution: {trial_id}", ylabel="Δ sensor (×10⁻⁶)")
    ax_delta.legend()

    ax_scaled = fig.add_subplot(grid[1], sharex=ax_delta)
    scaled_time = full_time[ESN_WASHOUT:]
    for column, name in enumerate(INPUT_NAMES):
        ax_scaled.plot(scaled_time, scaled_inputs[:, column], label=name, linewidth=1.2)
    ax_scaled.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    ax_scaled.set(title="Five standardized inputs supplied to the reservoir", ylabel="Training z-score")
    ax_scaled.legend(ncol=2, fontsize=8)

    ax_units = fig.add_subplot(grid[2], sharex=ax_delta)
    for unit in units:
        ax_units.plot(state_time, states[:, unit], label=f"Unit {unit}", linewidth=1.2)
    ax_units.set(title="Representative reservoir units", ylabel="Activation")
    ax_units.legend(ncol=5, fontsize=8)

    ax_heatmap = fig.add_subplot(grid[3], sharex=ax_delta)
    limit = np.max(np.abs(states))
    image = ax_heatmap.imshow(
        states.T, aspect="auto", origin="lower",
        extent=[state_time.min(), state_time.max(), 0, states.shape[1] - 1],
        cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest",
    )
    ax_heatmap.set(title="Full reservoir-state heatmap", ylabel="Reservoir unit")

    # Put the colorbar in an inset axis outside the heatmap. Passing
    # ax=ax_heatmap directly to fig.colorbar would shrink only this subplot,
    # making its time-axis width inconsistent with all other panels.
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    colorbar_axis = inset_axes(
        ax_heatmap,
        width="1.4%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0, 1.0, 1.0),
        bbox_transform=ax_heatmap.transAxes,
        borderpad=0,
    )
    fig.colorbar(image, cax=colorbar_axis, label="Activation")

    ax_norm = fig.add_subplot(grid[4], sharex=ax_delta)
    ax_norm.plot(state_time, np.linalg.norm(states, axis=1), label="L2 state norm")
    ax_norm.plot(state_time, np.mean(np.abs(states), axis=1), label="Mean absolute activation")
    ax_norm.set(xlabel="Time from contact (s)", ylabel="Collective activity")
    ax_norm.legend()
    plt.show()


plot_esn_dynamics()
''')

# Save fixed-analysis outputs under an AUX-BB namespace.
save_md = md("""## Save AUX-BB fixed-analysis results

These files are separate from the AUX optimization outputs. They contain predictions generated after loading and fixing the selected shared hyperparameters.
""")
save_code = code('''ALL_RESULTS_FILE = RESULTS_DIR / f"{AUX_BB_RESULT_STEM}_overall_metrics.csv"
ALL_RESULTS.to_csv(ALL_RESULTS_FILE, index=False)

for (protocol, family, target), table in fold_metrics.items():
    table.to_csv(
        RESULTS_DIR / f"{AUX_BB_RESULT_STEM}_{protocol}_{family}_{target}_fold_metrics.csv",
        index=False,
    )
for (protocol, family, target), table in predictions.items():
    table.to_csv(
        RESULTS_DIR / f"{AUX_BB_RESULT_STEM}_{protocol}_{family}_{target}_oof_predictions.csv",
        index=False,
    )

print("Saved AUX-BB fixed-analysis outputs with stem:", AUX_BB_RESULT_STEM)
print("Overall metrics:", ALL_RESULTS_FILE)
''')
nb["cells"][24:24] = [save_md, save_code]

# Update provenance language throughout without touching computational logic.
for cell in nb["cells"]:
    if cell["cell_type"] != "markdown":
        continue
    text = "".join(cell["source"])
    text = text.replace("Run 4AAA when", "Run AUX when")
    if text.startswith("# 12. Interpretation and provenance"):
        text = """# 12. Interpretation and provenance

- No hyperparameter optimization occurs in AUX-BB; settings are loaded from AUX.
- Per-trial baseline centering exactly matches the AUX optimization input definition.
- `StandardScaler` is fitted only on each outer fold's training trials.
- `Temperature_C` remains a known operating-condition predictor.
- The pooled LOMO OOF result is the main unseen-material diagnostic.
- Per-material LOMO R² is undefined because each held-out material has one constant effusivity; use pooled R² and valid fold-error metrics.
- These CV models evaluate the shared modeling procedure. They are not one permanently serialized deployment model.
"""
    cell["source"] = text.splitlines(keepends=True)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

nb.setdefault("metadata", {})["analysis"] = {
    "role": "fixed_consumer_for_aux_baseline_centered_optimization",
    "parameter_source": "AUX baseline-centered Optuna TPE",
    "target": "eff",
    "input_preprocessing": "per-trial baseline centering plus fold-local StandardScaler",
}

OUTPUT.write_text(json.dumps(nb, indent=1))
print(OUTPUT)
