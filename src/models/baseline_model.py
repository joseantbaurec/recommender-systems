import itertools

import pandas as pd
from scipy import sparse

from models.abstract_model import RecommenderModel


class BaselineRecommender(RecommenderModel):
    """
    Simplest non-random recommender model, used as baseline.

    Constructs a co-occurrence matrix with how many times each pair of items are bought "together".

    Gives recommendations for a given item prompt by returning the k most popular items bought together with the prompt.
    """

    co_occurrence_matrix: sparse.dok_matrix

    def setup_model(self, **kwargs):
        print('Building co-occurrence matrix...')
        self.co_occurrence_matrix = self._get_co_occurrence_matrix()
        print('Done')

    def _get_co_occurrence_matrix(self) -> sparse.dok_matrix:
        n_items = self.dataset.n_products
        matrix = sparse.dok_array((n_items, n_items))
        for _, user in self.dataset.users.iterrows():
            for item1, item2 in itertools.combinations(user['train_relevant_items'], 2):
                index1 = self.dataset.product_to_index[item1]
                index2 = self.dataset.product_to_index[item2]
                if index2 < index1:
                    index1, index2 = index2, index1
                matrix[index1, index2] += 1
        return matrix

    def model_name(self):
        return "Baseline Recommender model"

    def _get_recommendations(
        self, user: str, item: str, n_recommendations: int, **kwargs
    ) -> (pd.Series, pd.DataFrame):
        anchor_item_index = self.dataset.product_to_index[item]
        recs = {}
        for item_index in range(self.dataset.n_products):
            i, j = min(item_index, anchor_item_index), max(
                item_index, anchor_item_index
            )
            co_occurrence = self.co_occurrence_matrix[i, j]
            co_item = self.dataset.index_to_product[item_index]
            recs[co_item] = co_occurrence
        recs = pd.Series(recs, name='co-popularity')
        recs = pd.merge(
            self.dataset.products, recs, left_index=True, right_index=True, how='right'
        )
        recs = recs.sort_values('co-popularity', ascending=False).head(
            n_recommendations
        )
        return recs.index, recs

    @property
    def matrix_density(self) -> float:
        nnz = self.co_occurrence_matrix.getnnz()
        n_rows, n_cols = self.co_occurrence_matrix.shape
        return nnz / (n_rows * n_cols)
