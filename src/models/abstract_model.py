import random
import time
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from prettytable import PrettyTable
from tqdm import tqdm

from base.dataset import DataSet
from base.results import Results


class RecommenderModel(ABC):
    """
    Generic class for Recommender Model, for the e-commerce dataset.
    """

    dataset: DataSet = None
    setup_time: float = None

    def __init__(self, dataset: DataSet, **kwargs):
        self.dataset = dataset
        t0 = time.perf_counter()
        self.setup_model(**kwargs)
        t1 = time.perf_counter()
        self.setup_time = t1 - t0

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def setup_model(self, **kwargs):
        pass

    def recommend(
        self,
        user: str | None = None,
        item: str | None = None,
        n_recommendations: int = 10,
        validation_run: bool = False,
        silent: bool = False,
        **kwargs,
    ) -> (Results | list[Results], Any):
        if user is None:
            user = self.dataset.get_random_user().name
            if not silent:
                print(f'Chose user {user} as recommender target')
        user_items = self.dataset.users.loc[user, 'train_relevant_items']
        if item is None:
            item = random.choice(user_items)
            if not silent:
                print(f'Chose item {item} as recommender prompt')

        recommendations, model_info = self._get_recommendations(
            user, item, n_recommendations, **kwargs
        )
        results = Results(recommendations, user_items)
        if validation_run:
            results = [
                results,
                Results(
                    recommendations,
                    self.dataset.users.loc[user, 'validation_relevant_items'],
                ),
            ]
        return results, model_info

    @abstractmethod
    def _get_recommendations(
        self, user: str, item: str, n_recommendations: int, **kwargs
    ) -> (pd.Series, Any):
        pass

    def evaluate_performance(self, n_runs: int = 100):
        times = []
        for i in tqdm(range(n_runs)):
            user = str(self.dataset.get_random_user().name)
            item = self.dataset.get_random_product_from_user(user)
            t0 = time.perf_counter()
            _ = self.recommend(user, item, silent=True)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        print('------------------------------------')
        print(f'{self.model_name} - Performance')
        print('------------------------------------')
        print(f'Model setup time: {self.setup_time:.3f}s')
        print(f'Average time: {sum(times)/len(times):.3f}s')
        print(f'Worst time: {max(times):.3f}s')
        print(f'Best time: {min(times):.3f}s')

    def evaluate_accuracy(self, k: int, n_runs: int = 100):
        stats = {
            'Average': lambda x: x.mean(),
            'Median': lambda x: x.median(),
            'Highest': lambda x: x.max(),
            'Lowest': lambda x: x.min(),
        }
        metrics = {
            f'MAP@{k}': {
                'values_full': [],
                'values_validation': [],
                'format': lambda x: f'{x:.4f}',
                'calc': lambda x, r: x.average_precision_at_k(r),
            },
            f'R@{k}': {
                'values_full': [],
                'values_validation': [],
                'format': lambda x: f'{x:.4f}',
                'calc': lambda x, r: x.recall_at_k(r),
            },
            f'P@{k}': {
                'values_full': [],
                'values_validation': [],
                'format': lambda x: f'{x*100:.2f}%',
                'calc': lambda x, r: x.precision_at_k(r),
            },
            f'HR@{k}': {
                'values_full': [],
                'values_validation': [],
                'format': lambda x: f'{x*100:.2f}%',
                'calc': lambda x, r: x.hit_rate_at_k(r),
            },
            f'Rank@{k}': {
                'values_full': [],
                'values_validation': [],
                'format': lambda x: f'{x:.4f}',
                'calc': lambda x, r: x.rank_at_k(r),
            },
            f'RecRank@{k}': {
                'values_full': [],
                'values_validation': [],
                'format': lambda x: f'{x:.4f}',
                'calc': lambda x, r: x.reciprocal_rank_at_k(r),
            },
        }
        for i in tqdm(range(n_runs)):
            user = str(self.dataset.get_random_user().name)
            results, _ = self.recommend(
                user, n_recommendations=k, silent=True, validation_run=True
            )
            for name, metric in metrics.items():
                f = metric['calc']
                metric['values_full'].append(f(results[0], k))
                metric['values_validation'].append(f(results[1], k))

        for phase in ['validation', 'full']:
            table = PrettyTable(['Metric', *stats])
            for metric, results in metrics.items():
                values = pd.Series(results[f'values_{phase}'])
                f = results['format']
                table.add_row([metric, *[f(stat(values)) for stat in stats.values()]])

            print('------------------------------------')
            print(f'{self.model_name} - {phase.capitalize()} statistics')
            print('------------------------------------')
            print(table)
