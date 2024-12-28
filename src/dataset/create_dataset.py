import argparse
import os

from dataset_creator import SyntheticDataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Create and save a custom function dataset.")
    
    parser.add_argument("--max_discontinuities", type=int, default=4, help="Maximum number of discontinuities.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples in the dataset.")
    parser.add_argument("--num_features", type=int, default=10, help="Number of features for each sample.")
    parser.add_argument("--end_time", type=int, default=10, help="End time for the function.")
    parser.add_argument("--output_folder", type=str, default="src/data", help="Output file to save the dataset.")
    parser.add_argument("--num_t_points", type=int, default=100, help="Number of points")
    args = parser.parse_args()
    
    return args

def main():
    
    args = parse_args()
    creator = SyntheticDataGenerator(
        max_discontinuities=args.max_discontinuities,
        num_samples=args.num_samples,
        num_features=args.num_features,
        end_time=args.end_time,
        num_t_points=args.num_t_points
    )
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    dataset = creator.generate_synthetic_data()
    dataset.to_csv(os.path.join(args.output_folder, "synthetic_data.csv"), index=False)



if __name__ == "__main__":
    main()
