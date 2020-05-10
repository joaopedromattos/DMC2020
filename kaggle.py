# coding: utf-8
import time
import argparse

import numpy as np
import pandas as pd


def load_data(path, price=False):
    """price: weather or not the file contains the simulatedPrice"""
    expected_cols = ["demandPrediction"]

    try:
        df = pd.read_csv(path, sep="|", index_col="itemID")
    except Exception as e:
        print("ERROR")
        print("Maybe you forgot to encode with '|' ?")
        raise e
    # Check cols are correct
    if not price and (df.columns != expected_cols).any():
        raise ValueError("Expected the following columns ONLY\n"
                         + str(expected_cols))
    pred_type = df["demandPrediction"].dtype
    if pred_type != "int64":
        raise ValueError(f"Expected demandPrediction to be a int, got {pred_type}")
    return df


def main(path, divide):
    answers = load_data("autotest.csv", price=True)
    answers.sort_index(inplace=True)

    predictions = load_data(path)
    predictions.sort_index(inplace=True)
    assert (predictions.index == answers.index).all()

    revenue = np.minimum(answers["demandPrediction"], predictions["demandPrediction"])
    revenue = revenue.astype(float)
    revenue *= answers["simulationPrice"]

    fee = np.maximum(predictions["demandPrediction"]-answers["demandPrediction"],
                     np.zeros(len(predictions)))
    fee = fee.astype(float)
    fee *= 0.6*answers["simulationPrice"]

    total = (revenue + fee).sum()
    if divide:
        total /= 1e6

    # Print score with drumrolls
    print("Calculating your score...")
    for i in range(1, 6):
        time.sleep(1)
        print("..."*i)
    print(total)
    print("\n\n")
    name = f'{path.split("/")[-1].split(".")[0]}_{total:.2f}.csv'
    print(f"Saving updated prediction to {name}")
    predictions.to_csv(name, sep="|")


if __name__ == '__main__':
    # print(args.accumulate(args.integers))
    example = "\nFor example: 'python3 kaggle.py my_submission.csv'"
    parser = argparse.ArgumentParser(description='Calculate DMC score/1e6. --'
                                     + example)
    parser.add_argument('path', type=str, help='Path to CSV file from this dir')
    parser.add_argument('--divide', default=True,
                        help='Weather or not to divide by 1e6 (default: True)')

    args = parser.parse_args()
    print(args)
    main(args.path, args.divide)
