import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data/"
SAVE_DIR = "saved_cf/"
os.makedirs(SAVE_DIR, exist_ok=True)

df_prior = pd.read_csv(os.path.join(DATA_DIR, "order_products__prior.csv"))

all_users = df_prior['user_id'].unique()
selected_users = np.random.choice(all_users, size=50000, replace=False)
df_prior_small = df_prior[df_prior['user_id'].isin(selected_users)]

top_products = df_prior_small['product_id'].value_counts().head(15000).index
df_prior_small = df_prior_small[df_prior_small['product_id'].isin(top_products)]

start_date = pd.to_datetime("2020-01-01")

df_prior_small["cumulative_days"] = df_prior_small.groupby("user_id")["days_since_prior_order"].cumsum()
df_prior_small["order_date"] = start_date + pd.to_timedelta(df_prior_small["cumulative_days"], unit="D")

current_dow = df_prior_small["order_date"].dt.dayofweek
dow_diff = (df_prior_small["order_dow"] - current_dow) % 7
df_prior_small["order_date"] = df_prior_small["order_date"] + pd.to_timedelta(dow_diff, unit="D")
df_prior_small["order_date"] = df_prior_small["order_date"] + pd.to_timedelta(df_prior_small["order_hour_of_day"], unit="h")

df_prior_small.drop("eval_set", axis=1, inplace=True, errors='ignore')

user_item_matrix = df_prior_small.pivot_table(
    index="user_id",
    columns="product_id",
    values="reordered",
    aggfunc="sum",
    fill_value=0
)

item_sim_matrix = pd.DataFrame(
    cosine_similarity(user_item_matrix.T),
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

user_item_matrix.to_pickle(os.path.join(SAVE_DIR, "user_item_matrix.pkl"))
item_sim_matrix.to_pickle(os.path.join(SAVE_DIR, "item_similarity_matrix.pkl"))

