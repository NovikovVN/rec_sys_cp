import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

# Модель второго уровня
from lightgbm import LGBMClassifier


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting='tfidf'):

        print('Preparing tops...')
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        print('Preparing matrix...')
        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        print('Preparing dicts...')
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        print('Weighting...')
        if weighting == 'tfidf':
            self.user_item_matrix = tfidf_weight(self.user_item_matrix.T).T
        else:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        print('Fitting als...')
        self.model = self.fit(self.user_item_matrix)
        print('Fitting own recommender...')
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        print('Complete.')

    @staticmethod
    def _prepare_matrix(data):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=30, regularization=0.01, iterations=15):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=1)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():

            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                        user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                        N=N,
                                        filter_already_liked_items=False,
                                        filter_items=[self.itemid_to_id[999999]],
                                        recalculate_user=True)]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        if user not in self.userid_to_id.keys():
            return self.top_purchases['item_id'].tolist()[:N]

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        if user not in self.userid_to_id.keys():
            return self.top_purchases['item_id'].tolist()[:N]

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        if user not in self.userid_to_id.keys():
            return self.top_purchases['item_id'].tolist()[:N]

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        if user not in self.userid_to_id.keys():
            return self.top_purchases['item_id'].tolist()[:N]

        res = []

        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)
        similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:]   # удалим юзера из запроса

        for user in similar_users:
            res.extend(self.get_own_recommendations(user, N=1))

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_top_items(self, _, N=5):
        """Рекомендуем топ-N всех товаров"""

        return self.top_purchases['item_id'].head(N).tolist()


class LightGBRecommender:
    """Рекоммендации, которые можно получить с ALS и LGBMClassifier
    Input
    -----
    data_train_lvl_1: pd.DataFrame
        Данные для обучения модели первого уровня (ALS)
    data_train_lvl_2: pd.DataFrame
        Данные для обучения модели второго уровня (LGBMClassifier)
    item_features: pd.DataFrame
        Данные о товарах
    user_features: pd.DataFrame
        Данные о пользователях
    """

    def __init__(self, data_train_lvl_1, data_train_lvl_2,
                 item_features, user_features):
        self.fit(data_train_lvl_1, data_train_lvl_2,
                 item_features, user_features)

    def fit(self, data_train_lvl_1, data_train_lvl_2, item_features, user_features):
        """Обучение ALS и LGBMClassifier"""

        print('FIRST LEVEL:')
        self.recommender = MainRecommender(data_train_lvl_1, 'tfidf')

        print('\nSECOND LEVEL:')
        print('Preparing users and items...')

        users_lvl_2 = pd.DataFrame(data_train_lvl_2['user_id'].unique())
        users_lvl_2.columns = ['user_id']

        train_users = data_train_lvl_1['user_id'].unique()
        users_lvl_2 = users_lvl_2[users_lvl_2['user_id'].isin(train_users)]

        users_lvl_2['candidates'] = users_lvl_2['user_id'].apply( \
                                    lambda x: self.recommender.get_als_recommendations(x, N=200))

        s = users_lvl_2.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'item_id'

        users_lvl_2 = users_lvl_2.drop('candidates', axis=1).join(s)
        users_lvl_2['is_recommended'] = 1

        users_lvl_2.reset_index(drop=True, inplace=True)

        targets_lvl_2 = data_train_lvl_2[['user_id', 'item_id']].copy()
        targets_lvl_2.drop_duplicates(inplace=True)
        targets_lvl_2['target'] = 1

        targets_lvl_2 = users_lvl_2.merge(targets_lvl_2, on=['user_id', 'item_id'], how='outer')
        targets_lvl_2['target'].fillna(0, inplace=True)
        targets_lvl_2['is_recommended'].fillna(0, inplace=True)
        is_recommended = targets_lvl_2['is_recommended'] == 1

        print('Merging features...')

        targets_lvl_2 = targets_lvl_2.merge(item_features, on='item_id', how='left')
        targets_lvl_2 = targets_lvl_2.merge(user_features, on='user_id', how='left')

        self.handcrafted_features, \
        self.fillna_values = self.make_handcrafted_features(data_train_lvl_2, item_features)
        targets_lvl_2 = targets_lvl_2.merge(self.handcrafted_features, on='user_id', how='left')

        print('Fitting lgb...')

        X_train = targets_lvl_2.drop(['user_id', 'is_recommended', 'item_id', 'target'], axis=1)
        y_train = targets_lvl_2[['target']].values.ravel()

        self.cat_feats = ['manufacturer', 'department', 'brand', 'commodity_desc',
                          'sub_commodity_desc', 'curr_size_of_product', 'age_desc',
                          'marital_status_code', 'income_desc', 'homeowner_desc',
                          'hh_comp_desc', 'household_size_desc', 'kid_category_desc']

        X_train[self.cat_feats] = X_train[self.cat_feats].astype('category')
        self.lgb_feats = X_train.columns

        self.lgb = LGBMClassifier(objective='binary', n_estimators=100,
                                  max_depth=15, random_state=42, class_weight='balanced',
                                  categorical_column=self.cat_feats)
        self.lgb.fit(X_train, y_train)
        print('Complete.')

        self.feature_importances_ = self._set_feature_importances()

        return self.lgb

    def predict(self, users, train_users, item_features, user_features):
        """Получение предсказаний для списка пользователей"""

        users_df = pd.DataFrame(users)
        users_df.columns = ['user_id']

        users_df = users_df[users_df['user_id'].isin(train_users)]
        users_df['candidates'] = users_df['user_id'].apply( \
                                 lambda x: self.recommender.get_own_recommendations(x, N=200))

        s = users_df.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'item_id'

        users_df = users_df.drop('candidates', axis=1).join(s)

        users_df.reset_index(drop=True, inplace=True)

        users_df = users_df.merge(item_features, on='item_id', how='left')
        users_df = users_df.merge(user_features, on='user_id', how='left')
        users_df = users_df.merge(self.handcrafted_features, on='user_id', how='left')
        users_df = users_df.fillna(self.fillna_values)

        X = users_df.drop(['user_id', 'item_id'], axis=1)
        X[self.cat_feats] = X[self.cat_feats].astype('category')

        y_proba = self.lgb.predict_proba(X)[:, 1]

        answers = users_df.loc[:, ['user_id', 'item_id']].copy()
        answers['rank'] = y_proba

        answers.sort_values(by=['user_id', 'rank', 'item_id'], ascending=[True, False, True], inplace=True)

        answers = answers.groupby(by='user_id').agg({'item_id': lambda x: list(x)}).reset_index()
        answers.rename(columns={'item_id': 'lgb'}, inplace=True)

        return answers

    def _set_feature_importances(self):
        """Оформление значений важности признаков модели lgm"""

        feature_importances = pd.Series(self.lgb.feature_importances_, self.lgb_feats)
        feature_importances.sort_values(ascending=False, inplace=True)

        return feature_importances

    @staticmethod
    def make_handcrafted_features(data, item_features):
        """Создание собственных признаков"""

        def n_unique(x):
            return len(set(x))

        # Количество транзакций пользователя,
        # количество посещенных им магазинов,
        # количества посещенных им отделов
        temporary_df = data.merge(item_features, on='item_id', how='left')
        handcrafted_features = temporary_df.groupby('user_id').agg({'basket_id': n_unique,
                                                                    'store_id': n_unique,
                                                                    'department': n_unique})
        handcrafted_features.rename(columns={'basket_id': 'n_transactions',
                                             'store_id': 'user_per_stores',
                                             'department': 'n_departments'}, inplace=True)

        # Средний чек пользователя
        average_bill = data.groupby(['user_id', 'basket_id']).agg({'sales_value': np.mean})
        average_bill = average_bill.reset_index().groupby('user_id').agg({'sales_value': np.mean}).reset_index()
        average_bill.rename(columns={'sales_value': 'average_bill'}, inplace=True)
        handcrafted_features = handcrafted_features.merge(average_bill, on='user_id', how='inner')

        # Данные для заполнения в датафрейме у неизвестных пользщователей
        fillna_values = {'n_transactions': handcrafted_features['n_transactions'].median(),
                         'user_per_stores': handcrafted_features['user_per_stores'].median(),
                         'n_departments': handcrafted_features['n_departments'].median(),
                         'average_bill': handcrafted_features['average_bill'].mean()}

        return handcrafted_features, fillna_values


if __name__ == '__main__':
    pass
