import pandas as pd

from base.dataset import DataSet
from base.plotting import plot_long_tail
from models.baseline_model import BaselineRecommender

if __name__ == "__main__":
    import os

    os.chdir('..')
    dataset = DataSet()
    dataset.to_pickle()

    # dataset = DataSet()
    # model = Word2VecRecommender(
    #     dataset=dataset
    # )
    # model.evaluate_performance()
    # model.evaluate_accuracy(10)
    a = 2
