import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            run = wandb.init(job_type="basic_cleaning")
            # 1. Download the artifact (raw data)
            artifact = run.use_artifact("sample.csv:latest")
            artifact_path = artifact.file()

            # 2. Load data
            import pandas as pd
            df = pd.read_csv(artifact_path)

            # 3. Clean data using Config (Rubric Requirement)
            min_price = config['etl']['min_price']
            max_price = config['etl']['max_price']
            
            # Filter rows based on price
            df = df[(df['price'] > min_price) & (df['price'] < max_price)]

            # 4. Save and Upload Clean Data
            df.to_csv("clean_sample.csv", index=False)
            
            clean_artifact = wandb.Artifact(
                name="clean_sample.csv",
                type="cleaned_data",
                description="Data cleaned by basic_cleaning step"
            )
            clean_artifact.add_file("clean_sample.csv")
            run.log_artifact(clean_artifact)
            run.finish()

        if "data_check" in active_steps:
            # Run the data check component we just edited
            _ = mlflow.run(
                "src/data_check",
                "main",
                env_manager="conda",
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                env_manager="conda",
                parameters={
                    "input": "clean_sample.csv",
                    "test_size": config["modeling"]["test_size"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "random_seed": config["modeling"]["random_seed"]
                },
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################

            pass

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            pass


if __name__ == "__main__":
    go()
