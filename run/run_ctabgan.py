# Importing the model
from model.ctabgan import CTABGAN
# Importing the evaluation metrics
from model.eval.evaluation import get_utility_metrics,stat_sim,privacy_metrics
# Importing standard libraries
import numpy as np
import pandas as pd
import glob
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset", type=str, choices=["adult", "bank"], default="adult")
    parser.add_argument("--num_exp", type=int, default=1, help="Specifying the replication number")

    args = parser.parse_args()
    # Specifying the replication number
    num_exp = args.num_exp
    # Specifying the name of the dataset used
    dataset = args.dataset

    root_dir = '../'
    # Specifying the path of the dataset used
    real_path = f"{root_dir}/datasets/raw/{dataset}/train.csv"
    # Specifying the root directory for storing generated data
    fake_file_root = "Fake_Datasets"

    # load attributs
    import json

    with open(f"{root_dir}/datasets/raw/{dataset}/attrs.json", 'r') as f:
        attrs = json.load(f)

    categorical_columns = attrs["categorical"]
    integer_columns = attrs["numerical"]
    problem_type = {"Classification": attrs["target"]}

    # Initializing the synthesizer object and specifying input parameters
    # Notice: If you have continuous variable, you do not need to explicitly assign it. It will be treated like
    # that by default
    synthesizer = CTABGAN(raw_csv_path=real_path,
                          test_ratio=0.20,
                          categorical_columns=categorical_columns,
                          log_columns=[],
                          mixed_columns={},
                          integer_columns=integer_columns,
                          problem_type=problem_type,
                          epochs=150)

    # Fitting the synthesizer to the training dataset and generating synthetic data
    for i in range(num_exp):
        synthesizer.fit()
        syn = synthesizer.generate_samples()
        syn.to_csv(fake_file_root + "/" + dataset + "/" + dataset + "_fake_{exp}.csv".format(exp=i), index=False)

