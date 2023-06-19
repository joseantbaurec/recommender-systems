from typing import Any

import gensim.models
import pandas as pd
import tqdm

from models.abstract_model import RecommenderModel


class Word2VecRecommender(RecommenderModel):
    """
    Skip-gram (1 prompt -> context probability) implementation of Word2Vec embedding algorithm to detect patterns in successive user purchases.

    Hopes to maximize the click-rate of related products, with the assumption that it also increases the purchase rate of those products.
    """

    sessions: list[list[str]]
    model: gensim.models.Word2Vec

    @property
    def model_name(self) -> str:
        return "Word2Vec SG session-based Recommender"

    def setup_model(self, **kwargs):
        print('Extracting user sessions...')
        self.sessions = self._extract_user_sessions()
        print('Training model...')
        self.model = self._train_word2vec_model(**kwargs)
        print('Done!')

    def _extract_user_sessions(self) -> list[list[str]]:
        group_count = self.dataset.all_transactions.groupby('user_session').agg(
            item_count=('product_id', 'count')
        )['item_count']

        events = self.dataset.all_transactions
        events = events.sort_values(['user_session', 'event_time'])
        # Build sessions by grouping by "user_session". Avoid groupby if possible
        sessions = []
        i = 0
        with tqdm.tqdm(total=events.shape[0]) as progress_bar:
            while i < events.shape[0]:
                session_id = events.iloc[i]['user_session']
                session_length = group_count[session_id]
                session_items = list(events.iloc[i : i + session_length]['product_id'])
                sessions.append(list(dict.fromkeys(session_items)))
                i += session_length
                progress_bar.update(session_length)
        sessions = [session for session in sessions if len(session) > 1]
        return sessions

    def _train_word2vec_model(self, **kwargs) -> gensim.models.Word2Vec:
        kwargs['workers'] = 4
        kwargs['sg'] = 1
        model = gensim.models.Word2Vec(**kwargs)
        print('Bulding vocabulary...')
        model.build_vocab(self.sessions)
        print('Training neural network...')
        model.train(
            self.sessions,
            total_examples=model.corpus_count,
            epochs=kwargs.get('epochs', 50),
        )
        return model

    def _get_recommendations(
        self, user: str, item: str, n_recommendations: int, **kwargs
    ) -> (pd.Series, Any):
        recommendations = self.model.wv.most_similar(
            positive=item, topn=n_recommendations
        )
        recommendations = self.dataset.products.loc[
            [rec[0] for rec in recommendations]
        ].drop(columns=['category_id', 'category_code'])
        return recommendations.index, recommendations
