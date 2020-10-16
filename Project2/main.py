import sklearn
from sklearn import datasets
import pandas as pd


def main():
    data = datasets.load_breast_cancer()
    print(data.keys())
    print(data.DESCR)

if __name__ == '__main__':
    main()
