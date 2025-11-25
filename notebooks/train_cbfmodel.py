import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import gensim
from gensim.models import Word2Vec

DATA_DIR = "data/"

products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
aisles   = pd.read_csv(os.path.join(DATA_DIR, "aisles.csv"))
depart   = pd.read_csv(os.path.join(DATA_DIR, "departments.csv"))
df_prior = pd.read_csv(os.path.join(DATA_DIR, "order_products__prior.csv"))

products = products.merge(aisles, on="aisle_id", how="left")
products = products.merge(depart, on="department_id", how="left")

product_tokens = [name.lower().split() for name in products['product_name']]

w2v_model = Word2Vec(
    sentences=product_tokens,
    vector_size=500,
    window=5,
    min_count=1,
    workers=4
)

def get_product_vector(tokens):
    vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if len(vecs) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vecs, axis=0)

product_name_vecs = np.array([get_product_vector(tokens) for tokens in product_tokens])

numeric_features = df_prior.groupby('product_id').agg({
    'reordered': 'mean',
    'add_to_cart_order': 'mean'
}).reset_index()

products = products.merge(numeric_features, on='product_id', how='left').fillna(0)

encoder_aisle = OneHotEncoder()
aisles_encoded = encoder_aisle.fit_transform(products[['aisle_id']])

encoder_dept = OneHotEncoder()
dept_encoded = encoder_dept.fit_transform(products[['department_id']])

scaler = MinMaxScaler()
num_features = scaler.fit_transform(products[['reordered', 'add_to_cart_order']])
num_features_sparse = sp.csr_matrix(num_features)

product_features = sp.hstack([
    sp.csr_matrix(product_name_vecs),
    aisles_encoded,
    dept_encoded,
    num_features_sparse
])

nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(product_features)

def get_similar_products(product_idx, top_k=10):
    distances, indices = nn.kneighbors(product_features[product_idx], n_neighbors=top_k)
    return indices[0], distances[0]

os.makedirs("saved_cbf", exist_ok=True)

w2v_model.save("saved_cbf/product_w2v.model")

with open("saved_cbf/encoder_aisle.pkl", "wb") as f:
    pickle.dump(encoder_aisle, f)

with open("saved_cbf/encoder_dept.pkl", "wb") as f:
    pickle.dump(encoder_dept, f)

with open("saved_cbf/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

sp.save_npz("saved_cbf/product_features.npz", product_features)

with open("models/cbf_model.pkl", "wb") as f:
    pickle.dump(nn, f)
