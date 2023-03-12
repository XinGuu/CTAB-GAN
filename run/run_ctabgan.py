# Importing the model
import os

from model.ctabgan import CTABGAN
# Importing the evaluation metrics
from model.eval.evaluation import get_utility_metrics,stat_sim,privacy_metrics
# Importing standard libraries
import numpy as np
import pandas as pd
import glob
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset", type=str, choices=["adult", "bank", "cervical"], default="adult")
    parser.add_argument("--raw_csv_name", type=str, default="train", help="Depend on whether you name it train.csv or data.csv")
    parser.add_argument("--num_exp", type=int, default=1, help="Specifying the replication number")
    parser.add_argument("--epoch", type=int, default=150, help="Set as 150 in paper")
    parser.add_argument("--batch_size", type=int, default=500, help="Set as 500 in original code")
    parser.add_argument("--clip_norm", type=float, default=1., help="Set as 1 in original code")
    parser.add_argument("--epsilon", type=float, default=1.)
    parser.add_argument("--delta", type=float, default=1e-5, help="Set as 1e-5 in original code")

    args = parser.parse_args()
    # Specifying the replication number
    num_exp = args.num_exp
    # Specifying the name of the dataset used
    dataset = args.dataset

    root_dir = '../'
    # Specifying the path of the dataset used
    real_path = f"{root_dir}/datasets/raw/{dataset}/{args.raw_csv_name}.csv"
    # Specifying the root directory for storing generated data
    fake_file_root = f"{root_dir}/Fake_Datasets"

    # load attributes
    with open(f"{root_dir}/datasets/raw/{dataset}/attrs.json", 'r') as f:
        attrs = json.load(f)

    categorical_columns = attrs["categorical"]
    integer_columns = attrs["numerical"]
    if attrs["target"] == '':
        problem_type = {None: None}
    else:
        problem_type = {"classification/regression": attrs["target"]}

    # Initializing the synthesizer object and specifying input parameters
    # Notice: If you have continuous variable, you do not need to explicitly assign it. It will be treated like
    # that by default
    synthesizer = CTABGAN(raw_csv_path=real_path,
                          test_ratio=0.20,      # this doesn't matter now. i removed train/test split in the code
                          categorical_columns=categorical_columns,
                          log_columns=[],
                          mixed_columns={},
                          general_columns=["age"],
                          non_categorical_columns=[],
                          integer_columns=integer_columns,
                          problem_type=problem_type)

    # Fitting the synthesizer to the training dataset, generating synthetic data and storing generated data
    for i in range(num_exp):
        synthesizer.fit(clip_norm=args.clip_norm,
                        epsilon=args.epsilon,
                        delta=args.delta,
                        batch_size=args.batch_size,
                        epoch=args.epoch)
        syn = synthesizer.generate_samples()

        # Save generated synthetic data
        fake_file_dir = f"{fake_file_root}/{dataset}/"
        if not os.path.exists(fake_file_dir):
            os.makedirs(fake_file_dir)
        syn.to_csv(fake_file_dir + f"{dataset}_fake_{i}.csv", index=False)

