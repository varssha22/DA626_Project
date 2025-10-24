import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ======================================================
# 1️⃣ Custom Transformer for Reorder Rate Mapping
# ======================================================
class ReorderRateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, aisle_col='aisle', dept_col='department', target_col='reordered'):
        self.aisle_col = aisle_col
        self.dept_col = dept_col
        self.target_col = target_col
        self.aisle_map_ = {}
        self.dept_map_ = {}

    def fit(self, X, y=None):
        df = X.copy()
        if self.target_col in df.columns:
            self.aisle_map_ = df.groupby(self.aisle_col)[self.target_col].mean().to_dict()
            self.dept_map_ = df.groupby(self.dept_col)[self.target_col].mean().to_dict()
        return self

    def transform(self, X):
        df = X.copy()
        df['aisle_reorder_rate'] = df[self.aisle_col].map(self.aisle_map_).fillna(0.5)
        df['dept_reorder_rate'] = df[self.dept_col].map(self.dept_map_).fillna(0.5)
        return df

# ======================================================
# 2️⃣ Feature extraction for a single product (for CBF)
# ======================================================
def extract_features_for_cbf(product_id, current_cart, df_products, user_last_order_time=None):
    """
    Returns a dataframe with features ready for CBF model prediction.
    """
    prod_row = df_products[df_products['product_id'] == product_id].iloc[0]

    features = {
        'product_name': prod_row['product_name'],
        'aisle': prod_row['aisle'],
        'department': prod_row['department'],
        'order_hour_of_day': 12 if user_last_order_time is None else user_last_order_time.hour,
        'order_dow': 0 if user_last_order_time is None else user_last_order_time.weekday(),
        'add_to_cart_order': len(current_cart) + 1,
        'days_since_prior_order': 0 if user_last_order_time is None else (pd.Timestamp.now() - user_last_order_time).days
    }

    return pd.DataFrame([features])
