import random

import pandas as pd

from base.dataset import DataSet
from base.results import Results
from models.ad_hoc_model import AdHocRecommender
from models.word2vec_model import Word2VecRecommender

if __name__ == "__main__":
    transactions: pd.DataFrame = pd.read_parquet(
        '/Users/josean/Desktop/Playground/recommender-systems/data/shopping/e-commerce.parquet.gzip'
    )

    transactions.sample(frac=0.5).to_parquet(
        '/Users/josean/Desktop/Playground/recommender-systems/data/shopping/e-commerce_small.parquet.gzip',
        compression='gzip',
    )
    transactions.sample(frac=0.25).to_parquet(
        '/Users/josean/Desktop/Playground/recommender-systems/data/shopping/e-commerce_smaller.parquet.gzip',
        compression='gzip',
    )
    # dataset = DataSet()
    # model = Word2VecRecommender(
    #     dataset=dataset
    # )
    # model.evaluate_performance()
    # model.evaluate_accuracy(10)
    a = 2
