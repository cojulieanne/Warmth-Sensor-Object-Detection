import json
from pathlib import Path
from textwrap import dedent


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


cells = [
    markdown(r"""
    # 4Av6 — Simplified Multi-Temperature ESN Optimization

    This notebook is the collaboration-friendly version of 4Av5. It keeps the same
    algorithm and experimental pipeline:

    1. load the four temperature datasets and apply standard material properties;
    2. detect contact and retain the absolute, smoothed 0–3 s response;
    3. calculate 27 thermal features;
    4. transform five sensor channels with an ESN and summarize every reservoir unit;
    5. optimize one shared ESN–XGBoost configuration with 14-fold leave-one-material-out validation;
    6. save the best parameters, out-of-fold predictions, metrics, and MLflow records.

    The code is intentionally linear. Checks for unusual files and rare edge cases are
    omitted so collaborators can see the main research method directly.
    """),
    markdown("""
    ## 1. Imports and settings

    Change the trial counts here when doing a quick demonstration.
    """),
    code(r"""
    from datetime import datetime, timezone
    from pathlib import Path
    import json
    import time

    import mlflow
    import numpy as np
    import optuna
    import pandas as pd

    from scipy import linalg
    from scipy.signal import savgol_filter
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    TEMPERATURE_FOLDERS = ("30C", "40C", "50C", "60C")
    TARGET = "eff"
    ANALYSIS_WINDOW = (0.0, 3.0)
    THERMAL_WINDOWS = {"early": (0.0, 1.0), "mid": (1.0, 2.0), "late": (2.0, 3.0)}
    RESERVOIR_SEEDS = (42, 43, 44)

    N_TRIALS = 100
    N_STARTUP_TRIALS = 20
    OPTUNA_SEED = 4042
    STABILITY_WEIGHT = 0.25

    cwd = Path.cwd().resolve()
    PROJECT_ROOT = cwd if (cwd / "data").is_dir() else cwd.parent
    DATA_ROOT = PROJECT_ROOT / "data" / "02_preprocessed"
    RESULTS_DIR = PROJECT_ROOT / "results" / "multi_temp_esn"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    temperature_tag = "-".join(TEMPERATURE_FOLDERS)
    RESULT_STEM = f"{temperature_tag}_{TARGET}_4av6_simplified_absolute_smoothed_0to3_summarized_esn_thermal_optuna_tpe_mlflow"
    CHECKPOINT_FILE = RESULTS_DIR / f"{RESULT_STEM}_trial_checkpoint.csv"
    PARAMETER_FILE = RESULTS_DIR / f"{RESULT_STEM}_parameters.json"

    MLFLOW_DB_FILE = RESULTS_DIR / "4Av6_simplified_mlflow_tracking.db"
    MLFLOW_ARTIFACT_DIR = RESULTS_DIR / "4av6_simplified_mlflow_artifacts"
    MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_FILE.resolve()}"
    MLFLOW_EXPERIMENT = "4Av6_simplified_absolute_summarized_esn_thermal_optuna_tpe"
    MLFLOW_RUN_NAME = datetime.now(timezone.utc).strftime("4Av6_%Y%m%dT%H%M%SZ")

    print("Data:", DATA_ROOT)
    print("Optuna trials:", N_TRIALS)
    print("Results:", RESULTS_DIR)
    """),
    markdown("""
    ## 2. Load trials and assign material properties

    Values of `k`, `rho`, and `cp` in the processed CSV files are replaced by the
    common reference table. Effusivity is calculated as $e=\sqrt{k\rho c_p}$.
    """),
    code(r"""
    STANDARD_PROPERTIES = {
        "ps_foam":  {"k": 0.034, "Mass": 1.0, "Volume": 1.0, "rho": 25.0,   "cp": 1400.0},
        "pu_foam":  {"k": 0.043, "Mass": 1.0, "Volume": 1.0, "rho": 30.0,   "cp": 1400.0},
        "cork":     {"k": 0.043, "Mass": 1.0, "Volume": 1.0, "rho": 240.0,  "cp": 1800.0},
        "wood":     {"k": 0.150, "Mass": 1.0, "Volume": 1.0, "rho": 700.0,  "cp": 1700.0},
        "pdms":     {"k": 0.150, "Mass": 1.0, "Volume": 1.0, "rho": 970.0,  "cp": 1460.0},
        "gypsum":   {"k": 0.170, "Mass": 1.0, "Volume": 1.0, "rho": 800.0,  "cp": 1090.0},
        "cement":   {"k": 0.290, "Mass": 1.0, "Volume": 1.0, "rho": 1440.0, "cp": 750.0},
        "graphite": {"k": 100.0, "Mass": 1.0, "Volume": 1.0, "rho": 1820.0, "cp": 710.0},
        "bismuth":  {"k": 8.1,   "Mass": 1.0, "Volume": 1.0, "rho": 9780.0, "cp": 130.0},
        "titanium": {"k": 21.9,  "Mass": 1.0, "Volume": 1.0, "rho": 4506.0, "cp": 523.0},
        "nickel":   {"k": 90.9,  "Mass": 1.0, "Volume": 1.0, "rho": 8908.0, "cp": 461.0},
        "iron":     {"k": 80.4,  "Mass": 1.0, "Volume": 1.0, "rho": 7874.0, "cp": 449.0},
        "aluminum": {"k": 237.0, "Mass": 1.0, "Volume": 1.0, "rho": 2700.0, "cp": 897.0},
        "copper":   {"k": 401.0, "Mass": 1.0, "Volume": 1.0, "rho": 8960.0, "cp": 385.0},
    }

    SAMPLE_ALIASES = {
        "ps": "ps_foam", "ps foam": "ps_foam", "pu": "pu_foam", "pu foam": "pu_foam",
        "cork": "cork", "cork fine": "cork", "wood": "wood", "pdms": "pdms",
        "gypsum": "gypsum", "cement": "cement", "graphite": "graphite", "carbon": "graphite",
        "bi": "bismuth", "bismuth": "bismuth", "ti": "titanium", "titanium": "titanium",
        "ni": "nickel", "nickel": "nickel", "fe": "iron", "iron": "iron",
        "al": "aluminum", "aluminum": "aluminum", "cu": "copper", "copper": "copper",
    }

    frames = []
    for temperature in TEMPERATURE_FOLDERS:
        for path in sorted((DATA_ROOT / temperature).glob("*.csv")):
            frame = pd.read_csv(path)
            frame["Temperature"] = temperature
            frame["source_file"] = path.name
            frames.append(frame)

    DATA = pd.concat(frames, ignore_index=True)
    DATA["Sample"] = (
        DATA["Sample"].astype(str).str.strip().str.lower().str.replace("_", " ").map(SAMPLE_ALIASES)
    )
    DATA["Trial"] = DATA["Trial"].astype(int)
    DATA["Temperature_C"] = DATA["Temperature"].str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)

    for name in ("k", "Mass", "Volume", "rho", "cp"):
        DATA[name] = DATA["Sample"].map(lambda material: STANDARD_PROPERTIES[material][name])

    DATA["trial_id"] = (
        DATA["Temperature"] + "__" + DATA["Sample"] + "_trial_" + DATA["Trial"].astype(str)
    )
    DATA["eff"] = np.sqrt(DATA["k"] * DATA["rho"] * DATA["cp"])
    DATA = DATA.sort_values(["Temperature", "trial_id", "Time"]).reset_index(drop=True)

    print("Rows:", len(DATA))
    print("Trials:", DATA["trial_id"].nunique())
    print("Materials:", DATA["Sample"].nunique())
    display(pd.DataFrame.from_dict(STANDARD_PROPERTIES, orient="index"))
    """),
    markdown("""
    ## 3. Detect contact and align the 0–3 s analysis window

    Contact is the elbow before the strongest negative derivative of the smoothed
    primary signal. Target properties are not used for alignment.
    """),
    code(r"""
    def find_contact_time(trial, window=15, polyorder=2, threshold=0.30, skip=5):
        trial = trial.sort_values("Time").drop_duplicates("Time")
        time = trial["Time"].to_numpy(float)[skip:]
        signal = trial["Primary"].to_numpy(float)[skip:]

        selected_window = min(window, len(signal))
        if selected_window % 2 == 0:
            selected_window -= 1

        smooth_signal = savgol_filter(signal, selected_window, polyorder, mode="interp")
        derivative = np.gradient(smooth_signal, time)
        strongest = np.argmin(derivative)
        active = derivative < threshold * derivative[strongest]

        elbow = 0
        for position in range(strongest, -1, -1):
            if not active[position]:
                elbow = position + 1
                break
        return time[elbow]


    ALIGNED_TRIALS = {}
    alignment_rows = []

    for trial_id, trial in DATA.groupby("trial_id", sort=False):
        trial = trial.sort_values("Time").drop_duplicates("Time").copy()
        contact_time = find_contact_time(trial)
        trial["time_from_contact"] = trial["Time"] - contact_time
        trial = trial[trial["time_from_contact"].between(*ANALYSIS_WINDOW)].copy()
        ALIGNED_TRIALS[trial_id] = trial.reset_index(drop=True)
        alignment_rows.append({
            "trial_id": trial_id,
            "Temperature": trial["Temperature"].iloc[0],
            "Sample": trial["Sample"].iloc[0],
            "contact_time": contact_time,
            "samples": len(trial),
        })

    ALIGNMENT = pd.DataFrame(alignment_rows)
    print("Aligned trials:", len(ALIGNED_TRIALS))
    display(ALIGNMENT.head())
    """),
    markdown("""
    ## 4. Calculate 27 thermal-response features

    For the primary, secondary, and difference signals, the notebook measures response
    magnitude, area, slope, and rate. Three cross-sensor features complete the table.
    """),
    code(r"""
    def smooth(values, window=11, polyorder=2):
        selected_window = min(window, len(values))
        if selected_window % 2 == 0:
            selected_window -= 1
        return savgol_filter(values, selected_window, polyorder, mode="interp")


    def thermal_features(trial):
        time = trial["time_from_contact"].to_numpy(float)
        primary = smooth(trial["Primary"].to_numpy(float))
        secondary = smooth(trial["Secondary"].to_numpy(float))
        signals = {
            "primary": primary,
            "secondary": secondary,
            "difference": primary - secondary,
        }
        features = {}

        for name, signal in signals.items():
            change = signal - signal[0]
            rate = np.gradient(signal, time)
            features[f"{name}_final_change"] = change[-1]
            features[f"{name}_max_abs_change"] = np.max(np.abs(change))
            features[f"{name}_response_auc"] = np.trapezoid(np.abs(change), time)
            for period, (start, end) in THERMAL_WINDOWS.items():
                mask = (time >= start) & (time <= end)
                features[f"{name}_{period}_slope"] = np.polyfit(time[mask], signal[mask], 1)[0]
            features[f"{name}_max_abs_rate"] = np.max(np.abs(rate))
            features[f"{name}_rate_auc"] = np.trapezoid(np.abs(rate), time)

        features["secondary_primary_auc_ratio"] = (
            features["secondary_response_auc"] / features["primary_response_auc"]
        )
        features["initial_sensor_difference"] = signals["difference"][0]
        features["final_sensor_difference"] = signals["difference"][-1]
        return features


    rows = []
    for trial_id, trial in ALIGNED_TRIALS.items():
        row = thermal_features(trial)
        row.update({
            "trial_id": trial_id,
            "Temperature": trial["Temperature"].iloc[0],
            "Temperature_C": trial["Temperature_C"].iloc[0],
            "Sample": trial["Sample"].iloc[0],
            "Trial": trial["Trial"].iloc[0],
            "k": trial["k"].iloc[0],
            "eff": trial["eff"].iloc[0],
        })
        rows.append(row)

    THERMAL_FEATURES = pd.DataFrame(rows)
    ID_COLUMNS = ["trial_id", "Temperature", "Temperature_C", "Sample", "Trial", "k", "eff"]
    THERMAL_COLUMNS = [column for column in THERMAL_FEATURES if column not in ID_COLUMNS]

    print("Thermal features per trial:", len(THERMAL_COLUMNS))
    display(THERMAL_FEATURES.head())
    """),
    markdown("""
    ## 5. Run the Echo State Network

    The ESN receives five absolute smoothed channels: primary, secondary, their
    difference, and the derivatives of primary and secondary. Each reservoir-unit
    trajectory is represented by the same nine summaries used in 4Av5.
    """),
    code(r"""
    class Reservoir:
        def __init__(self, res_size, leak_rate, input_magnitude, spectral_radius, seed):
            self.res_size = res_size
            self.leak_rate = leak_rate
            rng = np.random.default_rng(seed)
            self.Win = (rng.random((res_size, 6)) - 0.5) * input_magnitude
            W = rng.random((res_size, res_size)) - 0.5
            self.W = W / np.max(np.abs(linalg.eigvals(W))) * spectral_radius

        def run(self, sequence):
            state = np.zeros((self.res_size, 1))
            states = []
            for inputs in sequence:
                inputs = inputs.reshape(-1, 1)
                state = (
                    (1 - self.leak_rate) * state
                    + self.leak_rate * np.tanh(self.Win @ np.vstack((1.0, inputs)) + self.W @ state)
                )
                states.append(state[:, 0].copy())
            return np.asarray(states)


    def esn_input(trial):
        time = trial["time_from_contact"].to_numpy(float)
        primary = smooth(trial["Primary"].to_numpy(float))
        secondary = smooth(trial["Secondary"].to_numpy(float))
        return np.column_stack([
            primary,
            secondary,
            primary - secondary,
            np.gradient(primary, time),
            np.gradient(secondary, time),
        ])


    def summarize_states(time, states):
        features = {}
        for unit, values in enumerate(states.T):
            summaries = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "initial": values[0],
                "net_change": values[-1] - values[0],
                "slope": np.polyfit(time, values, 1)[0],
                "mean_abs": np.mean(np.abs(values)),
                "abs_auc": np.trapezoid(np.abs(values), time),
            }
            for name, value in summaries.items():
                features[f"esn_{name}_u{unit:03d}"] = value
        return features
    """),
    markdown("""
    ## 6. Evaluate one shared ESN–XGBoost candidate

    Each candidate is evaluated with leave-one-material-out validation. The same
    hyperparameters are used in all 14 folds, and predictions are averaged over three
    reservoir seeds. The selection objective is

    $$J=\mathrm{pooled\ NRMSE}_{range}+0.25\,\mathrm{SD}(\mathrm{fold\ NRMSE}).$$
    """),
    code(r"""
    def regression_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {
            "n": len(y_true),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": rmse,
            "nrmse_range": rmse / np.ptp(y_true),
            "r2": r2_score(y_true, y_pred),
            "median_ape_pct": np.median(np.abs((y_true - y_pred) / y_true)) * 100,
        }


    def evaluate_candidate(metadata, params):
        esn_params = {name: params[name] for name in (
            "res_size", "leak_rate", "input_magnitude", "spectral_radius"
        )}
        xgb_params = {name: params[name] for name in (
            "n_estimators", "max_depth", "learning_rate", "min_child_weight",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"
        )}

        fold_rows = []
        prediction_rows = []
        splitter = LeaveOneGroupOut()

        for fold, (train_index, valid_index) in enumerate(
            splitter.split(metadata, groups=metadata["Sample"]), start=1
        ):
            train_meta = metadata.iloc[train_index]
            valid_meta = metadata.iloc[valid_index]
            train_ids = train_meta["trial_id"].tolist()
            valid_ids = valid_meta["trial_id"].tolist()
            predictions_by_seed = []

            for seed in RESERVOIR_SEEDS:
                reservoir = Reservoir(**esn_params, seed=seed)
                feature_rows = []

                for trial_id in train_ids + valid_ids:
                    trial = ALIGNED_TRIALS[trial_id]
                    states = reservoir.run(esn_input(trial))
                    row = summarize_states(trial["time_from_contact"].to_numpy(float), states)
                    row["trial_id"] = trial_id
                    feature_rows.append(row)

                features = pd.DataFrame(feature_rows).set_index("trial_id")
                features = features.join(THERMAL_FEATURES.set_index("trial_id"))
                esn_columns = [column for column in features if column.startswith("esn_")]
                feature_columns = ["Temperature_C"] + esn_columns + THERMAL_COLUMNS

                X_train = features.loc[train_ids, feature_columns]
                X_valid = features.loc[valid_ids, feature_columns]
                y_train = features.loc[train_ids, TARGET]

                xgb = XGBRegressor(
                    objective="reg:squarederror",
                    random_state=seed,
                    n_jobs=-1,
                    tree_method="hist",
                    verbosity=0,
                    **xgb_params,
                )
                model = TransformedTargetRegressor(
                    regressor=Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("xgb", xgb),
                    ]),
                    func=np.log1p,
                    inverse_func=np.expm1,
                    check_inverse=False,
                )
                model.fit(X_train, y_train)
                prediction = np.clip(model.predict(X_valid), y_train.min(), y_train.max())
                predictions_by_seed.append(prediction)

            y_true = valid_meta[TARGET].to_numpy(float)
            y_pred = np.mean(predictions_by_seed, axis=0)
            residual = y_pred - y_true
            rmse = np.sqrt(np.mean(residual ** 2))
            training_range = np.ptp(train_meta[TARGET].to_numpy(float))

            fold_rows.append({
                "fold": fold,
                "held_out_material": valid_meta["Sample"].iloc[0],
                "n": len(y_true),
                "mae": np.mean(np.abs(residual)),
                "rmse": rmse,
                "nrmse_training_range": rmse / training_range,
                "mean_error_bias": np.mean(residual),
                "median_ape_pct": np.median(np.abs(residual / y_true)) * 100,
            })
            prediction_rows.append(pd.DataFrame({
                "trial_id": valid_meta["trial_id"].to_numpy(),
                "Temperature": valid_meta["Temperature"].to_numpy(),
                "Temperature_C": valid_meta["Temperature_C"].to_numpy(),
                "Sample": valid_meta["Sample"].to_numpy(),
                "Trial": valid_meta["Trial"].to_numpy(),
                "fold": fold,
                "y_true": y_true,
                "y_pred": y_pred,
            }))

        folds = pd.DataFrame(fold_rows)
        predictions = pd.concat(prediction_rows, ignore_index=True)
        summary = regression_metrics(predictions["y_true"], predictions["y_pred"])
        fold_nrmse = folds["nrmse_training_range"].to_numpy()
        summary["mean_fold_nrmse_training_range"] = np.mean(fold_nrmse)
        summary["sd_fold_nrmse_training_range"] = np.std(fold_nrmse)
        summary["worst_fold_nrmse_training_range"] = np.max(fold_nrmse)
        summary["prediction_range_coverage"] = np.ptp(predictions["y_pred"]) / np.ptp(metadata[TARGET])
        summary["performance_index"] = (
            summary["nrmse_range"] + STABILITY_WEIGHT * summary["sd_fold_nrmse_training_range"]
        )
        return folds, predictions, summary
    """),
    markdown("""
    ## 7. Optimize with Optuna TPE and record trials in MLflow

    Two known configurations are evaluated first. Optuna then searches exactly the
    same refined parameter ranges as 4Av5. Every trial completes all folds.
    """),
    code(r"""
    METADATA = THERMAL_FEATURES[ID_COLUMNS].reset_index(drop=True)

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=N_STARTUP_TRIALS,
        multivariate=True,
        seed=OPTUNA_SEED,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    anchors = [
        {
            "res_size": 20, "leak_rate": 0.10, "input_magnitude": 1.0, "spectral_radius": 0.90,
            "n_estimators": 500, "max_depth": 3, "learning_rate": 0.05,
            "min_child_weight": 1.0, "subsample": 0.85, "colsample_bytree": 0.85,
            "reg_alpha": 1e-4, "reg_lambda": 1.0,
        },
        {
            "res_size": 11, "leak_rate": 0.70, "input_magnitude": 0.36, "spectral_radius": 0.83,
            "n_estimators": 800, "max_depth": 4, "learning_rate": 0.03,
            "min_child_weight": 0.5, "subsample": 0.90, "colsample_bytree": 0.90,
            "reg_alpha": 1e-5, "reg_lambda": 0.3,
        },
    ]
    for anchor in anchors:
        study.enqueue_trial(anchor)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        experiment_id = client.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=MLFLOW_ARTIFACT_DIR.resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)

    best_result = {"loss": np.inf}
    history_rows = []

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as parent_run:
        PARENT_RUN_ID = parent_run.info.run_id
        mlflow.set_tags({
            "notebook": "4Av6",
            "model": "summarized_ESN_plus_thermal_features_plus_XGBoost",
            "validation": "14_fold_leave_one_material_out",
            "target": TARGET,
        })
        mlflow.log_params({
            "n_trials": N_TRIALS,
            "n_startup_trials": N_STARTUP_TRIALS,
            "optimizer_seed": OPTUNA_SEED,
            "stability_weight": STABILITY_WEIGHT,
            "reservoir_seeds": ",".join(map(str, RESERVOIR_SEEDS)),
            "analysis_window": str(ANALYSIS_WINDOW),
        })

        def objective(trial):
            params = {
                "res_size": trial.suggest_int("res_size", 10, 30),
                "leak_rate": trial.suggest_float("leak_rate", 0.10, 0.70),
                "input_magnitude": trial.suggest_float("input_magnitude", 0.25, 2.00),
                "spectral_radius": trial.suggest_float("spectral_radius", 0.70, 1.20),
                "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.20, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.10, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 1.00),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 1.00),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 3.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 30.0, log=True),
            }
            started = time.perf_counter()
            folds, predictions, summary = evaluate_candidate(METADATA, params)
            elapsed = time.perf_counter() - started

            metrics = {
                "objective_J": summary["performance_index"],
                "pooled_nrmse_range": summary["nrmse_range"],
                "pooled_r2": summary["r2"],
                "pooled_rmse": summary["rmse"],
                "pooled_mae": summary["mae"],
                "mean_fold_nrmse_training_range": summary["mean_fold_nrmse_training_range"],
                "sd_fold_nrmse_training_range": summary["sd_fold_nrmse_training_range"],
                "worst_fold_nrmse_training_range": summary["worst_fold_nrmse_training_range"],
                "prediction_range_coverage": summary["prediction_range_coverage"],
                "duration_seconds": elapsed,
            }

            with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                for _, row in folds.iterrows():
                    material = row["held_out_material"]
                    mlflow.log_metrics({
                        f"fold_{material}_rmse": row["rmse"],
                        f"fold_{material}_nrmse_training_range": row["nrmse_training_range"],
                        f"fold_{material}_median_ape_pct": row["median_ape_pct"],
                    })

            for name, value in metrics.items():
                trial.set_user_attr(name, value)
            history_rows.append({"trial": trial.number + 1, **params, **metrics})

            if summary["performance_index"] < best_result["loss"]:
                best_result.update({
                    "loss": summary["performance_index"],
                    "params": params.copy(),
                    "folds": folds.copy(),
                    "predictions": predictions.copy(),
                    "summary": summary.copy(),
                })

            print(
                f"Trial {trial.number + 1:03d}/{N_TRIALS}: "
                f"J={summary['performance_index']:.4f}, "
                f"NRMSE={summary['nrmse_range']:.4f}, R²={summary['r2']:.4f}, "
                f"time={elapsed / 60:.1f} min"
            )
            return summary["performance_index"]

        study.optimize(
            objective,
            n_trials=N_TRIALS,
            callbacks=[lambda study, trial: study.trials_dataframe().to_csv(CHECKPOINT_FILE, index=False)],
            gc_after_trial=True,
        )

        mlflow.log_metrics({
            "best_objective_J": best_result["loss"],
            "best_pooled_nrmse_range": best_result["summary"]["nrmse_range"],
            "best_pooled_r2": best_result["summary"]["r2"],
        })
        mlflow.log_params({f"best_{name}": value for name, value in best_result["params"].items()})
        mlflow.log_artifact(str(CHECKPOINT_FILE), artifact_path="study")
    """),
    markdown("""
    ## 8. Review and save the best result

    The saved files are the handoff from optimization notebook 4A to the fixed-model
    analysis notebook 4B.
    """),
    code(r"""
    SEARCH_HISTORY = pd.DataFrame(history_rows).sort_values("objective_J").reset_index(drop=True)
    FOLD_METRICS = best_result["folds"]
    OOF_PREDICTIONS = best_result["predictions"]
    SUMMARY = best_result["summary"]
    BEST_PARAMETERS = best_result["params"]
    ESN_PARAMETER_NAMES = ("res_size", "leak_rate", "input_magnitude", "spectral_radius")
    XGB_PARAMETER_NAMES = (
        "n_estimators", "max_depth", "learning_rate", "min_child_weight",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"
    )
    BEST_ESN_PARAMETERS = {name: BEST_PARAMETERS[name] for name in ESN_PARAMETER_NAMES}
    BEST_ESN_PARAMETERS["washout"] = 0
    BEST_XGB_PARAMETERS = {name: BEST_PARAMETERS[name] for name in XGB_PARAMETER_NAMES}

    POOLED_METRICS = pd.DataFrame([{
        "target": TARGET,
        "protocol": "4av6_simplified_absolute_smoothed_0to3_summarized_esn_thermal_grouped_lomo_optuna_tpe",
        **{name: SUMMARY[name] for name in (
            "n", "mae", "rmse", "nrmse_range", "r2", "median_ape_pct",
            "prediction_range_coverage", "performance_index"
        )},
    }])

    TEMPERATURE_METRICS = pd.DataFrame([
        {"Temperature": temperature, **regression_metrics(part["y_true"], part["y_pred"])}
        for temperature, part in OOF_PREDICTIONS.groupby("Temperature")
    ])
    FOLD_STABILITY = pd.DataFrame([{
        "folds": len(FOLD_METRICS),
        "mean_fold_mae": FOLD_METRICS["mae"].mean(),
        "sd_fold_mae": FOLD_METRICS["mae"].std(ddof=1),
        "worst_fold_mae": FOLD_METRICS["mae"].max(),
        "mean_fold_rmse": FOLD_METRICS["rmse"].mean(),
        "sd_fold_rmse": FOLD_METRICS["rmse"].std(ddof=1),
        "worst_fold_rmse": FOLD_METRICS["rmse"].max(),
        "mean_fold_nrmse_training_range": SUMMARY["mean_fold_nrmse_training_range"],
        "sd_fold_nrmse_training_range": SUMMARY["sd_fold_nrmse_training_range"],
        "worst_fold_nrmse_training_range": SUMMARY["worst_fold_nrmse_training_range"],
    }])

    files = {
        "search_history": RESULTS_DIR / f"{RESULT_STEM}_search_history.csv",
        "fold_metrics": RESULTS_DIR / f"{RESULT_STEM}_fold_metrics.csv",
        "oof_predictions": RESULTS_DIR / f"{RESULT_STEM}_oof_predictions.csv",
        "pooled_metrics": RESULTS_DIR / f"{RESULT_STEM}_pooled_metrics.csv",
        "fold_stability": RESULTS_DIR / f"{RESULT_STEM}_fold_stability.csv",
        "temperature_metrics": RESULTS_DIR / f"{RESULT_STEM}_temperature_metrics.csv",
    }
    SEARCH_HISTORY.to_csv(files["search_history"], index=False)
    FOLD_METRICS.to_csv(files["fold_metrics"], index=False)
    OOF_PREDICTIONS.to_csv(files["oof_predictions"], index=False)
    POOLED_METRICS.to_csv(files["pooled_metrics"], index=False)
    FOLD_STABILITY.to_csv(files["fold_stability"], index=False)
    TEMPERATURE_METRICS.to_csv(files["temperature_metrics"], index=False)

    payload = {
        "source": "4Av6 simplified presentation version of the 4Av5 pipeline",
        "temperature_folders": list(TEMPERATURE_FOLDERS),
        "target": TARGET,
        "analysis_window": list(ANALYSIS_WINDOW),
        "reservoir_seeds": list(RESERVOIR_SEEDS),
        "thermal_feature_names": THERMAL_COLUMNS,
        "summary_features": [
            "mean", "std", "min", "max", "initial", "net_change", "slope", "mean_abs", "abs_auc"
        ],
        "input_preprocessing": "direct absolute Savitzky-Golay-smoothed values; no fitted scaler",
        "regressor_inputs": "nine summaries per ESN unit + 27 thermal features + Temperature_C",
        "esn_parameters": BEST_ESN_PARAMETERS,
        "xgb_parameters": BEST_XGB_PARAMETERS,
        "selection_metrics": {
            "objective_J": SUMMARY["performance_index"],
            "pooled_nrmse_range": SUMMARY["nrmse_range"],
            "pooled_r2": SUMMARY["r2"],
            "prediction_range_coverage": SUMMARY["prediction_range_coverage"],
            "sd_fold_nrmse_training_range": SUMMARY["sd_fold_nrmse_training_range"],
        },
        "optuna": {
            "sampler": "multivariate TPESampler",
            "n_trials": N_TRIALS,
            "n_startup_trials": N_STARTUP_TRIALS,
            "optimizer_seed": OPTUNA_SEED,
            "stability_weight": STABILITY_WEIGHT,
            "search_space": {
                "res_size": ["integer", 10, 30],
                "leak_rate": ["float", 0.10, 0.70],
                "input_magnitude": ["float", 0.25, 2.00],
                "spectral_radius": ["float", 0.70, 1.20],
                "n_estimators": ["integer", 200, 1200],
                "max_depth": ["integer", 2, 6],
                "learning_rate": ["log_float", 0.015, 0.20],
                "min_child_weight": ["log_float", 0.10, 10.0],
                "subsample": ["float", 0.65, 1.00],
                "colsample_bytree": ["float", 0.40, 1.00],
                "reg_alpha": ["log_float", 1e-6, 3.0],
                "reg_lambda": ["log_float", 0.01, 30.0],
            },
        },
        "mlflow": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment": MLFLOW_EXPERIMENT,
            "parent_run_id": PARENT_RUN_ID,
        },
    }
    with PARAMETER_FILE.open("w") as file:
        json.dump(payload, file, indent=2)

    with mlflow.start_run(run_id=PARENT_RUN_ID):
        for path in [*files.values(), PARAMETER_FILE]:
            mlflow.log_artifact(str(path), artifact_path="best_trial_outputs")

    print("Best parameters")
    display(pd.Series(BEST_PARAMETERS))
    print("Pooled out-of-fold performance")
    display(POOLED_METRICS)
    print("Material-held-out folds")
    display(FOLD_METRICS)
    print("Across-material fold stability")
    display(FOLD_STABILITY)
    print("Performance by temperature")
    display(TEMPERATURE_METRICS)
    print("Saved parameter handoff:", PARAMETER_FILE)
    print(f'mlflow ui --backend-store-uri "{MLFLOW_TRACKING_URI}"')
    """),
    markdown("""
    ## Interpretation

    `pooled_metrics` summarizes all out-of-fold predictions. `fold_metrics` shows how
    the shared model behaves when each material is completely unseen. These results
    are model-selection estimates; they are not an untouched external test result.
    """),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path("notebooks/4Av6 - Simplified Multi-Temp ESN Optuna MLflow Optimization.ipynb")
output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(output)
