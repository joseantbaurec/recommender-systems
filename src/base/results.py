import numpy as np
import pandas as pd


class Results:
    def __init__(self, recommendations: pd.Series, user_relevant_items: list[str]):
        self.recommendations: pd.Series = recommendations
        self.matches: pd.Series = pd.Series(
            recommendations.isin(user_relevant_items), name='matches'
        )
        self.n_recommendations: int = len(self.matches)
        self.n_relevant_items: int = len(user_relevant_items)

    def _matches_first_k_recommendations(self, k: int) -> pd.Series:
        return self.matches.iloc[: min(k, self.n_recommendations)]

    def _match_kth_recommendation(self, k: int) -> int:
        return self.matches.iloc[min(k - 1, self.n_recommendations)]

    def recall_at_k(self, k: int) -> float:
        relevant_recommendations = self._matches_first_k_recommendations(k).sum()
        return relevant_recommendations / self.n_relevant_items

    def precision_at_k(self, k: int) -> float:
        relevant_recommendations = self._matches_first_k_recommendations(k).sum()
        return relevant_recommendations / min(k, self.n_recommendations)

    def average_precision_at_k(self, k: int) -> float:
        instances = [
            self.precision_at_k(i + 1) * self._match_kth_recommendation(i + 1)
            for i in range(k)
        ]
        return sum(instances) / min(self.n_relevant_items, self.n_recommendations)

    def rank_at_k(self, k: int) -> int:
        new_index_matches = self._matches_first_k_recommendations(k)
        rank = new_index_matches.values.argmax()
        if new_index_matches.sum():
            return rank + 1
        else:
            return np.nan

    def reciprocal_rank_at_k(self, k: int) -> float:
        rank = self.rank_at_k(k)
        if np.isnan(rank):
            return 0
        else:
            return 1 / rank
