from typing import Any

import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import split

from base.dataset import DataSet
from models.abstract_model import RecommenderModel


class MatrixFactorizationRecommender(RecommenderModel):
    """
    Factorizes interaction/rating matrix (user-product) into two lower-rank matrices: R = U x P, using k as the intermediate dimension.

    This k turns out to represent the dimension of the latent-feature space: both matrices U and P work as an embedding of both users and items into a k-dimensional space,
    called the latent feature space. It can be understood as an embedding where each basis vector represents a "concept", and each user and item is divided into a linear
    combination of these concepts.

    For items, this represents how much each item "belongs" to every concept. For users, this represents how much they "like" each concept. Therefore, the final score
    of how much user X would like item P is the dot product of these embeddings.
    """

    user_item_confidence: pd.DataFrame
    spark: SparkSession
    model: ALSModel

    def __init__(self, dataset: DataSet, cross_validate_model: bool = False, **kwargs):
        super().__init__(dataset, cross_validate_model=cross_validate_model, **kwargs)

    @property
    def model_name(self) -> str:
        return "Matrix Factorization implicit Recommender"

    def setup_model(self, cross_validate_model: bool, **kwargs):
        print('Extracting user-item confidence...')
        self.user_item_confidence = self._extract_user_item_confidence()
        print('Starting PySpark session...')
        self.spark = (
            SparkSession.builder.master('local[1]').appName('recsys.com').getOrCreate()
        )
        print('Training ALS model...')
        self.model = self._train_ALS_model(cross_validate_model)
        print('Done!')

    def _extract_user_item_confidence(self) -> pd.DataFrame:
        confidence = self.dataset.relevants.groupby(['user_id', 'product_id']).agg(
            confidence=('category_id', 'count')
        )
        confidence = confidence.reset_index()
        return confidence

    def _train_ALS_model(self, cross_validate_model: bool) -> ALSModel:
        confidence_DF = self.spark.createDataFrame(self.user_item_confidence)
        confidence_DF = confidence_DF.withColumn(
            'user', split(confidence_DF['user_id'], '-').getItem(1).cast('int')
        )
        confidence_DF = confidence_DF.withColumn(
            'product', split(confidence_DF['product_id'], '-').getItem(1).cast('int')
        )
        als = ALS(
            implicitPrefs=True,
            userCol='user',
            itemCol='product',
            ratingCol='confidence',
            coldStartStrategy='drop',
        )
        if cross_validate_model:
            params = (
                ParamGridBuilder()
                .addGrid(als.regParam, [0.01, 0.05, 0.1, 0.15])
                .addGrid(als.rank, [10, 50, 100, 150])
                .build()
            )
            evaluator = RegressionEvaluator(
                metricName='rmse', labelCol='rating', predictionCol='prediction'
            )
            validator = CrossValidator(
                estimator=als,
                estimatorParamMaps=params,
                evaluator=evaluator,
                parallelism=4,
            )
            model = validator.fit(confidence_DF).bestModel
        else:
            model = als.fit(confidence_DF)
        return model

    def _get_recommendations(
        self, user: str, item: str, n_recommendations: int, **kwargs
    ) -> (pd.Series, Any):
        user_index = int(user.split('-')[1])
        user_DF = self.spark.createDataFrame(
            pd.DataFrame([[user_index]], columns=['user'])
        )

        recommendations_DF = self.model.recommendForUserSubset(
            user_DF, n_recommendations
        ).toPandas()
        recommendations: list = recommendations_DF.loc[0, 'recommendations']
        recommendations: dict = {
            'P-' + str(row.product): row.rating for row in recommendations
        }
        recommendations: pd.Series = pd.Series(recommendations, name='score')
        recommendations.index.name = 'product_id'

        recommendations: pd.DataFrame = self.dataset.products.merge(
            recommendations, how='right', left_index=True, right_index=True
        )
        return recommendations.index, recommendations
