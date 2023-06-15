import pandas as pd

from base.dataset import DataSet
from models.abstract_model import RecommenderModel


class AdHocRecommender(RecommenderModel):
    """
    Ad-hoc recommendation system for the e-commerce multi-store. Given a product as input, recommendations are
    extracted using the following method:
    - Compute category proximity: a percentage of how much the category of two products match. For example,
      two products from the same category have a score of 1, while one from electronics.smartphones and
      another from electronics.headphones have a score of 0.5. Keep only those with a score of at least 0.5
    - Same brand: 1 if they match, 0 otherwise.
    - Popularity within category: from selection, get percentile of each product when ordering by times sold.
    - Cheapest first: when two candidates have the same score, the cheapest gets priority.
    """

    category_score_cutoff: float
    brand_score_weight: float
    popularity_score_weight: float

    def __init__(
        self,
        dataset: DataSet,
        category_score_cutoff: float = 0.5,
        brand_score_weight: float = 0.0005,
        popularity_score_weight: float = 0.9995,
        **kwargs
    ):
        super().__init__(
            dataset,
            category_score_cutoff=category_score_cutoff,
            brand_score_weight=brand_score_weight,
            popularity_score_weight=popularity_score_weight,
            **kwargs
        )

    def setup_model(
        self,
        category_score_cutoff: float,
        brand_score_weight: float,
        popularity_score_weight: float,
        **kwargs
    ):
        self.category_score_cutoff = category_score_cutoff
        self.brand_score_weight = brand_score_weight
        self.popularity_score_weight = popularity_score_weight

    def model_name(self) -> str:
        return "Ad-Hoc Recommender model"

    def _get_recommendations(
        self, user: str, item: str, n_recommendations: int, **kwargs
    ) -> (pd.Series, pd.DataFrame):
        suggestions = self.dataset.products.copy()
        query_product = suggestions.loc[item]
        # Get category proximity
        suggestions['category_score'] = self._get_category_score(
            suggestions, query_product
        )
        suggestions = suggestions[
            suggestions['category_score'] >= self.category_score_cutoff
        ]
        # Get brand proximity
        suggestions['brand_score'] = suggestions['brand'] == query_product['brand']
        # Get product popularity
        suggestions['popularity_score'] = self._get_popularity_score(suggestions)
        # Compute ad-hoc score
        suggestions['score'] = suggestions['category_score'] * (
            self.brand_score_weight * suggestions['brand_score']
            + self.popularity_score_weight * suggestions['popularity_score']
        )
        suggestions = suggestions.sort_values(
            by=['score', 'price'], ascending=[False, True]
        )
        candidates = suggestions.head(n_recommendations).drop(
            columns=['category_id', 'category_code']
        )
        return candidates.index, candidates

    def _get_category_score(
        self, suggestions: pd.DataFrame, anchor_product: pd.Series
    ) -> pd.Series:
        anchor_categories = anchor_product[self.dataset.category_fields]
        product_categories = suggestions[self.dataset.category_fields]
        category_match = product_categories.eq(anchor_categories, axis=1)
        category_match_percentage = (
            category_match.sum(axis=1) / anchor_categories.count()
        )
        return category_match_percentage

    def _get_popularity_score(self, suggestions: pd.DataFrame) -> pd.Series:
        metrics = suggestions.merge(
            self.dataset.metrics, left_index=True, right_index=True, how='left'
        )
        metrics['sales_count'] = metrics['sales_count'].fillna(0)
        popularity = metrics['sales_count'] / metrics['sales_count'].sum()
        return popularity
