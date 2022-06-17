# Main script
from sklearn.model_selection import train_test_split
from src.logistic_regression import LogisticRegression
import pandas as pd


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    df = pd.read_csv("data/data.csv")
    print(df)
    print(df.describe())
    X = df.drop("diagnosis", axis=1)
    Y = df["diagnosis"]
    model_lr = LogisticRegression(5, 4, X, Y)


if __name__ == '__main__':
    print_hi('PyCharm')

