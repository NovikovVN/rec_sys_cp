import numpy as np
import pandas as pd


def prefilter_items(data, item_features, take_n_popular=5000):
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['user_id'] /= data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    this_year = data['week_no'].max() - data['week_no'] <= 52
    not_this_year_items = data.loc[~this_year, 'item_id'].unique()
    this_year_items = data.loc[this_year, 'item_id'].unique()
    not_this_year_items = not_this_year_items[~np.isin(not_this_year_items, this_year_items)].tolist()
    data = data[~data['item_id'].isin(not_this_year_items)]

    # Уберем не интересные для рекоммендаций категории (department)
    department_size = pd.DataFrame(item_features.\
                                   groupby('department')['item_id'].nunique().\
                                   sort_values(ascending=False)).reset_index()
    department_size.columns = ['department', 'n_items']

    rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
    items_in_rare_departments = item_features[item_features['department'].isin(rare_departments)].item_id.unique().tolist()

    data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / np.maximum(data['quantity'], 1)
    data = data[data['price'] > 2]

    # Уберем слишком дорогие товарыs
    data = data[data['price'] < 50]

    # Возьмём топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold',ascending=False).head(take_n_popular).item_id.tolist()

    # Заведём фиктивный item_id (если пользователь покупа товар из топа, то он покупал такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    # ...
    return data


def filter_top_items(data, take_n_popular=5000):
    # Возьмём топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold',ascending=False).head(take_n_popular).item_id.tolist()

    # Заведём фиктивный item_id (если пользователь покупа товар из топа, то он покупал такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data


if __name__ == '__main__':
    pass
