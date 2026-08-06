"""CLI: compare XGBoost, CatBoost and blend on current OpenDota data."""

from src.pipeline.train_compare import run_model_compare


if __name__ == "__main__":
    run_model_compare()
