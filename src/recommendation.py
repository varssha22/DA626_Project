import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load
import pickle
import warnings

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.preprocess_bert4rec import TFLiteBert4RecPipeline
from utils.cbf_scores_func import get_cbf_scores

# Disable GPU on Streamlit Cloud
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ===================== LOAD CORE MATRICES =====================

user_item_matrix = load("models/user_item_matrix_compressed.pkl")
item_sim_matrix = load("models/item_sim_matrix_compressed.pkl")
nn, product_features, cbf_products = load("models/cbf_model.joblib")


# ===================== LOAD METADATA =====================

products = pd.read_csv("data/products.csv")
aisles = pd.read_csv("data/aisles.csv")
departments = pd.read_csv("data/departments.csv")

products = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")

NUM_ITEMS = products["product_id"].nunique()
MAX_SEQ_LEN = 50


# ===================== LOAD / BUILD ID MAPPINGS =====================

mapping_p_to_idx_path = "models/product_id_to_index.pkl"
mapping_idx_to_p_path = "models/index_to_product_id.pkl"

if os.path.exists(mapping_p_to_idx_path) and os.path.exists(mapping_idx_to_p_path):
    with open(mapping_p_to_idx_path, "rb") as f:
        product_id_to_idx = pickle.load(f)
    with open(mapping_idx_to_p_path, "rb") as f:
        idx_to_product_id = pickle.load(f)
else:
    warnings.warn(
        "product_id_to_index.pkl or index_to_product_id.pkl missing. "
        "Using fallback mapping. Make sure this matches training."
    )
    unique_ids = sorted(products["product_id"].unique())
    product_id_to_idx = {pid: (i + 1) for i, pid in enumerate(unique_ids)}
    idx_to_product_id = {i + 1: pid for i, pid in enumerate(unique_ids)}

VOCAB_SIZE = max(idx_to_product_id.keys()) + 1


# ===================== LOAD BERT4REC TFLITE MODEL =====================

bert_pipeline = TFLiteBert4RecPipeline(
    model_path="models/bert4rec_compressed.tflite",
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    pad_id=0
)


# =============================================================
#                    HYBRID RECOMMENDER
# =============================================================

def recommend_for_new_user(selected_products):
    """
    selected_products: list of product names the user selected
    returns: hybrid recommendations (CBF + CF + BERT4Rec)
    """

    if not selected_products:
        return pd.DataFrame(columns=["product_name", "aisle", "department"])

    # Convert product names → raw IDs
    user_history_ids = products[
        products["product_name"].isin(selected_products)
    ]["product_id"].tolist()

    if len(user_history_ids) == 0:
        return pd.DataFrame(columns=["product_name", "aisle", "department"])

    # =============================================================
    # 1️⃣ BERT4Rec
    # =============================================================

    bert_df = pd.DataFrame()

    mapped_ids = [product_id_to_idx[pid] for pid in user_history_ids if pid in product_id_to_idx]

    if len(mapped_ids) > 0:
        bert_preds = bert_pipeline.recommend(mapped_ids, top_k=10)

        bert_indices = []
        if len(bert_preds) > 0:
            first = bert_preds[0]
            if isinstance(first, (list, tuple)) and len(first) >= 1:
                bert_indices = [int(idx) for idx, _ in bert_preds]
            else:
                bert_indices = [int(x) for x in bert_preds]

        bert_product_ids = [
            idx_to_product_id[idx] for idx in bert_indices if idx in idx_to_product_id
        ]

        if bert_product_ids:
            bert_df = products.loc[
                products["product_id"].isin(bert_product_ids),
                ["product_name", "aisle", "department"]
            ].copy()

    # =============================================================
    # 2️⃣ Collaborative Filtering (CF)
    # =============================================================

    valid_ids = [pid for pid in user_history_ids if pid in item_sim_matrix.index]

    if valid_ids:
        cf_scores = item_sim_matrix.loc[valid_ids].mean(axis=0).values
        cf_top_indices = np.argsort(cf_scores)[::-1][:3]
        cf_pids = item_sim_matrix.columns[cf_top_indices].tolist()
    else:
        cf_pids = []

    cf_df = products.loc[
        products["product_id"].isin(cf_pids),
        ["product_name", "aisle", "department"]
    ].copy()

    # =============================================================
    # 3️⃣ Content-Based Filtering (CBF)
    # =============================================================

    cbf_scores = get_cbf_scores(selected_products, products, nn, product_features)
    cbf_top_indices = np.argsort(cbf_scores)[::-1]

    cbf_top_indices = [
        idx for idx in cbf_top_indices
        if products.iloc[idx]["product_name"] != selected_products[-1]
    ]

    cbf_top_indices = cbf_top_indices[:3]

    cbf_df = products.iloc[cbf_top_indices][
        ["product_name", "aisle", "department"]
    ].copy()

    # =============================================================
    # 4️⃣ Combine All
    # =============================================================

    combined = pd.concat(
        [cbf_df, cf_df, bert_df],
        axis=0
    ).drop_duplicates(
        subset=["product_name"]
    ).reset_index(drop=True)

    return combined
