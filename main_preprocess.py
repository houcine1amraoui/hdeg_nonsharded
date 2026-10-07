from src.data_preparation.prepare_windows import main_prepare_windows
from src.data_preparation.data_preparation import preprocessing_pipeline
from src.utils.save_utils import save_processed_data
import yaml
import argparse

def main_preprocess():
    """
    Typical preprocessing pipeline:
    1s-raw data -> actors split -> noise filtering -> 5s-downsampling 
    -> normalization -> sliding windows
    """
    # 1. Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str)
    args = parser.parse_args()

    # override project_root_directory
    if args.project_root_dir:
        config["project_root_dir"] = args.project_root_dir

    root = config["project_root_dir"]
    print(root)
    
    splits_norm, timestamps, scaler, devices = preprocessing_pipeline(config) 
    
    save_processed_data(splits_norm, timestamps, scaler, devices, config)

    # windows construction
    main_prepare_windows(config)

    print("Data preprocessing Done.")

if __name__ == "__main__":
    main_preprocess()