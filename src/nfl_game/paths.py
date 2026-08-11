from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

V2_FEATURES_PATH = PROCESSED_DIR / "game_features_ridge_v2.parquet"
V2_MANIFEST_PATH = PROCESSED_DIR / "ridge_v2_manifest.json"
V2_OUTER_PREDICTIONS_PATH = PROCESSED_DIR / "ridge_v2_outer_predictions.parquet"
V2_EVALUATION_PATH = PROCESSED_DIR / "ridge_v2_evaluation.json"
V2_ABLATION_PATH = PROCESSED_DIR / "ridge_v2_ablation.parquet"
V2_CALIBRATION_PATH = PROCESSED_DIR / "ridge_v2_calibration.json"
V2_TRACKER_LEDGER_PATH = PROCESSED_DIR / "tracker_ledger_ridge_v2.parquet"
