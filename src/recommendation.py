import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.sasrec_model import SasRec
from utils.preprocess_sasrec import SASRecPipeline
from utils.cbf_scores_func import get_cbf_scores

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU use to prevent TF crash on Streamlit Cloud


# --- Cached resource loaders ---
@st.cache_resource
def load_metadata():
    products = pd.read_csv("data/products.csv")
    aisles = pd.read_csv("data/aisles.csv")
    departments = pd.read_csv("data/departments.csv")
    products = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")
    return products


@st.cache_resource
def load_precomputed_models():
    user_item_matrix = load("models/user_item_matrix_compressed.pkl")
    item_sim_matrix = load("models/item_sim_matrix_compressed.pkl")
    nn, product_features, cbf_products = load("models/cbf_model.joblib")
    return user_item_matrix, item_sim_matrix, nn, product_features, cbf_products


@st.cache_resource
def load_sasrec_pipeline(products):
    NUM_ITEMS = products['product_id'].nunique()
    MAX_SEQ_LEN = 15

    sasrec_model = SasRec(
        vocabulary_size=NUM_ITEMS + 1,
        max_sequence_length=MAX_SEQ_LEN,
        hidden_dim=32,
        num_heads=2,
        num_layers=2,
        dropout=0.3
    )

    # Dummy input to build model
    sasrec_model({
        "item_ids": tf.zeros((1, MAX_SEQ_LEN), dtype=tf.int32),
        "padding_mask": tf.zeros((1, MAX_SEQ_LEN), dtype=tf.bool)
    })

    sasrec_model.load_weights("models/best_sasrec_model (1).h5")

    sasrec_pipeline = SASRecPipeline(
        sasrec_model, vocab_size=NUM_ITEMS, max_context_len=MAX_SEQ_LEN
    )
    return sasrec_pipeline


# --- Load core data/models lazily ---
products = load_metadata()
user_item_matrix, item_sim_matrix, nn, product_features, cbf_products = load_precomputed_models()
sasrec_pipeline = load_sasrec_pipeline(products)


# --- Hybrid recommender for a new user ---
def recommend_for_new_user(selected_products):
    """
    selected_products: list of product names the user has added so far
    returns: DataFrame of top recommendations
    """
    if not selected_products:
        return pd.DataFrame(columns=["product_name", "aisle", "department"])

    user_history_ids = products[products["product_name"].isin(selected_products)]["product_id"].tolist()
    if len(user_history_ids) == 0:
        return pd.DataFrame(columns=["product_name", "aisle", "department"])

    NUM_ITEMS = products.shape[0]

    # =========================================================
    # 🔹 1️⃣ SASRec Recommendations
    # =========================================================
    sasrec_df = pd.DataFrame()
    if len(user_history_ids) >= 10:
        sasrec_recs = sasrec_pipeline.recommend(user_history_ids, top_k=3)
        sasrec_recs = [(pid, score) for pid, score in sasrec_recs if 1 <= pid <= NUM_ITEMS]
        sasrec_pids = [pid for pid, _ in sasrec_recs]
        sasrec_df = products.loc[products["product_id"].isin(sasrec_pids),
                                 ["product_name", "aisle", "department"]].copy()

    # =========================================================
    # 🔹 2️⃣ CF Recommendations
    # =========================================================
    valid_ids = [pid for pid in user_history_ids if pid in item_sim_matrix.index]

    if valid_ids:
        cf_scores = item_sim_matrix.loc[valid_ids].mean(axis=0).values
        cf_top_indices = np.argsort(cf_scores)[::-1][:3]
        cf_pids = item_sim_matrix.columns[cf_top_indices].tolist()
    else:
        cf_pids = []

    cf_df = products.loc[products["product_id"].isin(cf_pids),
                         ["product_name", "aisle", "department"]].copy()

    # =========================================================
    # 🔹 3️⃣ CBF Recommendations
    # =========================================================
    cbf_scores = get_cbf_scores(selected_products, products, nn, product_features)
    cbf_top_indices = np.argsort(cbf_scores)[::-1]
    cbf_top_indices = [idx for idx in cbf_top_indices
                       if products.iloc[idx]["product_name"] != selected_products[-1]]
    cbf_top_indices = cbf_top_indices[:3]
    cbf_pids = products.iloc[cbf_top_indices]["product_id"].tolist()

    cbf_df = products.iloc[cbf_top_indices][["product_name", "aisle", "department"]].copy()

    # =========================================================
    # 🔹 4️⃣ Combine All
    # =========================================================
    combined = pd.concat([cbf_df, cf_df, sasrec_df]).drop_duplicates(
        subset=["product_name"]).reset_index(drop=True)

    return combined
''' 
   # 2️⃣ SASRec scores
    sasrec_recs = sasrec_pipeline.recommend(user_history_ids, top_k=NUM_ITEMS)
    sasrec_scores = np.zeros(NUM_ITEMS)
    for pid, score in sasrec_recs:
        idx = pid - 1  # adjust for 0-based indexing
        if 0 <= idx < NUM_ITEMS:
            sasrec_scores[idx] = score

    # Optional normalization
    if sasrec_scores.max() > 0:
        sasrec_scores /= sasrec_scores.max()
    
    # 3️⃣ CF scores
    valid_ids = [pid for pid in user_history_ids if pid in item_sim_matrix.index]

    if valid_ids:
        cf_scores = item_sim_matrix.loc[valid_ids].mean(axis=0).values
    else:
        # if none of the user's items exist in CF matrix, fill with zeros
        cf_scores = np.zeros(len(item_sim_matrix.columns))

    if cf_scores.max() > 0:
        cf_scores /= cf_scores.max()

    # --- Expand CF scores to full SASRec item set ---
    cf_scores_full = np.zeros(len(sasrec_scores))  # length 49688
    cf_product_ids = item_sim_matrix.columns.tolist()  # ✅ use CF matrix columns (actual product IDs)

    for i, pid in enumerate(cf_product_ids):
        # ✅ fill only where pid is within SASRec range
        if 0 <= pid - 1 < len(cf_scores_full):
            cf_scores_full[pid - 1] = cf_scores[i]

    cf_scores = cf_scores_full
    # 4️⃣ CBF scores
    cbf_scores = get_cbf_scores(selected_products=selected_products,products=cbf_products,nn=nn,product_features=product_features)

    # 5️⃣ Weighted hybrid fusion
    final_scores = cbf_scores #0.3 * sasrec_scores + 0.2 * cf_scores + 
    top_indices = np.argsort(final_scores)[::-1][:6]
    recommendations = products.iloc[top_indices][["product_name", "aisle", "department"]]

    return recommendations'''


"""    

with open("models/cbf_pipeline.pkl", "rb") as f:
    cbf_pipeline = pickle.load(f)
cbf_input_list = []
    for p_name in selected_products:
        pid = products[products['product_name'] == p_name]['product_id'].values[0]
        current_time = pd.Timestamp.now()
        feats = extract_features_for_cbf(
            product_id=pid,
            current_cart=selected_products,
            df_products=products,
            user_last_order_time=current_time
        )
        cbf_input_list.append(feats)
    cbf_input_df = pd.concat(cbf_input_list, ignore_index=True)

if cbf_pipeline is not None:
        cbf_scores = cbf_pipeline.predict_proba(cbf_input_df)[:, 1]
        # Make it full-length vector
        cbf_scores_full = np.zeros(NUM_ITEMS)
        for i, pid in enumerate(cbf_input_df['product_name'].map(lambda x: products[products['product_name']==x]['product_id'].values[0])):
            cbf_scores_full[pid-1] = cbf_scores[i]
        cbf_scores = cbf_scores_full
    else:
        cbf_scores = np.zeros(NUM_ITEMS)"""
