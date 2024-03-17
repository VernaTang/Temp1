"""This script samples K examples randomly without replacement from the original data."""
#Add some operations for MELD dataset
import argparse
import os
import numpy as np
import pandas as pd
from pandas import DataFrame


def get_label(task, line):
    return line[0]


def load_datasets(data_dir, tasks):
    datasets = {}
    for task in tasks:
        dataset = {}
        dirname = os.path.join(data_dir, task)
        splits = ["train", "dev", "test"]
        for split in splits:
            filename = os.path.join(dirname, f"{split}.csv")
            dataset[split] = pd.read_csv(filename, header=None)
        datasets[task] = dataset
    return datasets


def split_header(task, lines):
    """
    Returns if the task file has a header or not. Only for GLUE tasks.
    """
    if task == "MELD":
        return lines[0:1], lines[1:]
    else:
        raise ValueError("Unknown GLUE task.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16,
                        help="Training examples for each class.")
    parser.add_argument("--task", type=str, nargs="+",
                        default=['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA', 'MRPC', 'QQP', 'STS-B',
                                 'MNLI', 'SNLI', 'QNLI', 'RTE'],
                        help="Task names")
    parser.add_argument("--seed", type=int, nargs="+",
                        default=[100, 13, 21, 42, 87],
                        help="Random seeds")

    parser.add_argument("--data_dir", type=str, default="data/original", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output path")
    parser.add_argument("--mode", type=str, default='k-shot', choices=['k-shot', 'k-shot-10x'],
                        help="k-shot or k-shot-10x (10x dev set)")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.mode)

    k = args.k
    print("K =", k)
    datasets = load_datasets(args.data_dir, args.task)

    for seed in args.seed:
        print("Seed = %d" % (seed))
        for task, dataset in datasets.items():
            # Set random seed
            np.random.seed(seed)

            # Shuffle the training set
            print("| Task = %s" % (task))
            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                # GLUE style
                train_header, train_lines = split_header(task, dataset["train"])
                np.random.shuffle(train_lines)
            elif task == "MELD":
                # For the MELD dataset
                train_header = dataset["train"].loc[0, [4, 1]].values.tolist()
                train_lines = dataset['train'].loc[1:, [4, 1]].values.tolist()
                np.random.shuffle(train_lines)
            else:
                # Other datasets
                train_header = dataset["train"][0:1].values.tolist()
                train_lines = dataset['train'][1:].values.tolist()
                np.random.shuffle(train_lines)

            # Set up dir
            task_dir = os.path.join(args.output_dir, task)
            setting_dir = os.path.join(task_dir, f"{k}-{seed}")
            os.makedirs(setting_dir, exist_ok=True)

            # Write test splits
            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                # GLUE style
                # Use the original development set as the test set (the original test sets are not publicly available)
                for split, lines in dataset.items():
                    if split.startswith("train"):
                        continue
                    split = split.replace('dev', 'test')
                    with open(os.path.join(setting_dir, f"{split}.csv"), "w") as f:
                        for line in lines:
                            f.write(str(line))
            elif task == "MELD":
                new_dataset = dataset['test'].loc[1:, [4, 1]]
                new_dataset.to_csv(os.path.join(setting_dir, 'test.csv'), header=False, index=False)
            else:
                # Other datasets
                # Use the original test sets
                dataset['test'].to_csv(os.path.join(setting_dir, 'test.csv'), header=False, index=False)


                # Get label list for balanced sampling
            label_list = {}
            for line in train_lines:
                label = get_label(task, line)
                print("label is: " + label)
                if label not in label_list:
                    label_list[label] = [line]
                else:
                    label_list[label].append(line)

            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                with open(os.path.join(setting_dir, "train.tsv"), "w") as f:
                    for line in train_header:
                        f.write(line)
                    for label in label_list:
                        for line in label_list[label][:k]:
                            f.write(line)
                name = "dev.tsv"
                if task == 'MNLI':
                    name = "dev_matched.tsv"
                elif task == 'MELD':
                    name = "dev.csv"
                # with open(os.path.join(setting_dir, name), "w") as f:
                #     for line in train_header:
                #         f.write(line)
                #     for label in label_list:
                #         dev_rate = 11 if '10x' in args.mode else 2
                #         for line in label_list[label][k:k * dev_rate]:
                #             f.write(line)
            else:
                new_train = []
                for label in label_list:
                    for line in label_list[label][:k]:
                        new_train.append(line)
                new_train = DataFrame(new_train)
                new_train.to_csv(os.path.join(setting_dir, 'train.csv'), header=False, index=False)

                new_dev = []
                for label in label_list:
                    dev_rate = 11 if '10x' in args.mode else 2
                    for line in label_list[label][k:k * dev_rate]:
                        new_dev.append(line)
                new_dev = DataFrame(new_dev)
                #new_dev.to_csv(os.path.join(setting_dir, 'dev.csv'), header=False, index=False)


if __name__ == "__main__":
    main()
