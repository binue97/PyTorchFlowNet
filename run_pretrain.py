import argparse
import os
import yaml
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from tools import flow_transforms
from custom_datasets import FlyingChairsDataLoader


def main():
    print('')
    print("\n[ Configure ]")
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("-i", "--config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # ========== Read YAML file ==========
    config = None
    print(f"--- Config file path: {args.config}")
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            # print(config)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    print("--- Done reading config file")


    # ========== Results ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config["result_path"] , timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(f"--- Output path: {output_path}")
    
    train_writer = SummaryWriter(os.path.join(output_path, "train"))
    test_writer = SummaryWriter(os.path.join(output_path, "test"))


    # ========== Data Loader ==========
    print("\n[ Data Loader ]")
    dataset_type = config['dataset']['type']
    data_loader = None
    if dataset_type == 'flying_chairs':
        data_loader = FlyingChairsDataLoader(config)
  



if __name__ == "__main__":
    main()
