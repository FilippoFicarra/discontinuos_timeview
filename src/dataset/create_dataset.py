import argparse
from dataset_creator import DatasetCreator
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Create and save a custom sine wave dataset.")
    
    parser.add_argument("--max_discontinuities", type=int, default=4, help="Maximum number of discontinuities.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples in the dataset.")
    parser.add_argument("--num_features", type=int, default=15, help="Number of features for each sample.")
    parser.add_argument("--end_time", type=int, default=10, help="End time for the sine wave.")
    parser.add_argument("--output_file", type=str, default="src/data/dataset.json", help="Output file to save the dataset.")
    
    args = parser.parse_args()
    
    return args

def main():
    
    args = parse_args()
    creator = DatasetCreator(
        max_discontiuities=args.max_discontinuities,
        num_samples=args.num_samples,
        num_features=args.num_features,
        end_time=args.end_time
    )
    
    function_dict = creator.create_dataset()
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    
    with open(args.output_file, "w") as f:
        json.dump(function_dict, f, indent=4)



if __name__ == "__main__":
    main()
