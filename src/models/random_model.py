from typing import Any

import pandas as pd

from base.dataset import DataSet
from models.abstract_model import RecommenderModel


class RandomRecommender(RecommenderModel):
    """
    Random recommendation
    """

    def setup_model(self, **kwargs):
        pass

    def model_name(self):
        return "Random Recommender model"

    def _get_recommendations(
        self, user: str, item: str, n_recommendations: int, **kwargs
    ) -> (pd.Series, pd.DataFrame):
        candidates = self.dataset.products.sample(n=n_recommendations).drop(
            columns=['category_id', 'category_code']
        )
        return candidates.index, candidates
