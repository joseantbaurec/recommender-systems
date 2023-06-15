import pickle
import random

import pandas as pd


def _get_file_from_dataset_size(dataset_size: str, file_type: str = 'parquet') -> str:
    base_path = 'data/shopping/e-commerce<dataset_size>.<file_type>'
    match dataset_size:
        case 'full':
            size = ''
        case 'small':
            size = '_small'
        case 'smaller':
            size = '_smaller'
        case _:
            raise ValueError(f'Unrecognized dataset size: {dataset_size}')
    match file_type:
        case 'parquet':
            extension = 'parquet.gzip'
        case 'pickle':
            extension = 'pickle'
        case _:
            print(f'Warning: Unhandled file extension .{file_type}')
            extension = file_type
    base_path = base_path.replace('<dataset_size>', size)
    base_path = base_path.replace('<file_type>', extension)
    return base_path


class DataSet:
    subcategory_depth = 1

    @classmethod
    def from_pickle(cls, dataset_size: str = 'small') -> 'DataSet':
        file_path = _get_file_from_dataset_size(dataset_size, file_type='pickle')
        with open(file_path, 'rb') as file:
            print('Reading file...')
            loaded_dataset: DataSet = pickle.load(file)
        print('Done!')
        return loaded_dataset

    def __init__(
        self, dataset_size: str = 'small', train_test_split: float = 0.2, **kwargs
    ):
        print('Reading file...')
        file_path = _get_file_from_dataset_size(dataset_size)
        transactions: pd.DataFrame = pd.read_parquet(file_path)
        print('Parsing dataset...')
        transactions['product_id'] = 'P-' + transactions['product_id'].astype(str)
        transactions['user_id'] = 'U-' + transactions['user_id'].astype(str)
        transactions['user_session'] = 'S-' + transactions['user_session'].astype(str)
        self.all_transactions: pd.DataFrame = transactions
        print('Building metrics...')
        self.metrics: pd.DataFrame = self.purchases.groupby('product_id').agg(
            sales_count=('product_id', 'count'),
            total_sales=('price', 'sum'),
        )
        print('Inferring product list...')
        self.products: pd.DataFrame = self._get_products()
        print('Building user-item relations...')
        self.users: pd.DataFrame = self._get_users(train_test_split)
        pti, itp, ipt = self._index_products()
        self.product_to_index: dict[str, int] = pti
        self.index_to_product: dict[int, str] = itp
        self.index_product_table: pd.DataFrame = ipt
        print('Done!')

    @property
    def views(self) -> pd.DataFrame:
        views = self.all_transactions[self.all_transactions['event_type'] == 'view']
        return views

    @property
    def purchases(self) -> pd.DataFrame:
        views = self.all_transactions[self.all_transactions['event_type'] == 'purchase']
        return views

    @property
    def relevants(self) -> pd.DataFrame:
        views = self.all_transactions[
            self.all_transactions['event_type'].isin(['purchase', 'cart'])
        ]
        return views

    def _get_products(self) -> pd.DataFrame:
        products = (
            self.all_transactions[
                ['product_id', 'category_id', 'category_code', 'brand', 'price']
            ]
            .drop_duplicates()
            .sort_values(by=['product_id', 'price'], ascending=[True, False])
        )
        # Products are often duplicated because of price (transactions are not the best source of this info)
        dupes = products['product_id'].duplicated()
        products = products[~dupes]
        # Expand category into subcategories
        categories = products['category_code'].str.split('.', expand=True)
        self.subcategory_depth: int = categories.shape[1]
        self.category_fields: list[str] = [
            f'category_code_L{i+1}' for i in range(self.subcategory_depth)
        ]
        categories.columns = self.category_fields
        products = pd.concat([products, categories], axis=1).set_index('product_id')
        return products

    def _get_users(self, train_test_split: float) -> pd.DataFrame:
        users = self.relevants.groupby('user_id').agg(
            relevant_items=('product_id', lambda x: list(x.drop_duplicates()))
        )
        users['n_relevant'] = users['relevant_items'].apply(len)
        users['validation_n_relevant'] = (
            train_test_split * users['n_relevant']
        ).astype(int)
        users['train_n_relevant'] = users['n_relevant'] - users['validation_n_relevant']
        users[['validation_relevant_items', 'train_relevant_items']] = users.apply(
            lambda row: (
                row['relevant_items'][: row['validation_n_relevant']],
                row['relevant_items'][row['validation_n_relevant'] :],
            ),
            axis=1,
            result_type='expand',
        )
        return users

    def get_random_product(self) -> pd.Series:
        product = self.products.sample(n=1).iloc[0]
        return product

    def get_random_product_from_user(
        self, user: str, for_validation: bool = False
    ) -> str:
        prefix = 'train_'
        if for_validation:
            prefix = 'validation_'
        user_items = self.users.loc[user, prefix + 'relevant_items']
        item = random.choice(user_items)
        return item

    def get_random_user(self, minimum_interactions: int = 10) -> pd.Series:
        user = (
            self.users.query(f"n_relevant >= {minimum_interactions}")
            .sample(n=1)
            .iloc[0]
        )
        return user

    @property
    def n_products(self) -> int:
        return self.products.shape[0]

    @property
    def n_users(self) -> int:
        return self.users.shape[0]

    def _index_products(self) -> (dict[str, int], dict[int, str], pd.DataFrame):
        index = pd.Series(self.products.index)
        itp = index.to_dict()
        pti = index.reset_index().set_index('product_id').to_dict()['index']
        return pti, itp, index.reset_index()
