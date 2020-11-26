import argparse
from plantsegtools.trainingtools.config_wizard import train_configurator_wizard
import glob


def parse():
    parser = argparse.ArgumentParser(description="Script to start the automated training wizard")
    parser.add_argument("--path", type=str, required=True, help='root path to h5 dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    if len(glob.glob(args.path + '/**/*.h5')) < 2:
        print("Please provide a root path with at least two h5 stacks (one for training one for validation)")
    else:
        train_configurator_wizard(args.path)
