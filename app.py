import sys
import os

sys.path.append(os.path.dirname(__file__))  # add project root
import streamlit as st
import pandas as pd
from src.recommendation import recommend_for_new_user  # import the hybrid recommender
import pickle

st.set_page_config(page_title="Hybrid Product Recommender", layout="wide")

# --- Load product metadata ---
@st.cache_resource
def load_products():
    products = pd.read_csv("data/products.csv")
    aisles = pd.read_csv("data/aisles.csv")
    departments = pd.read_csv("data/departments.csv")
    products = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")
    return products

products = load_products()

# --- Streamlit UI ---
st.title("🛒 Hybrid Product Recommender")
st.write("Add products to your cart and get real-time top recommendations!")

if 'user_cart' not in st.session_state:
    st.session_state.user_cart = []

# Product selection
product_to_add = st.selectbox("Select a product to add to your cart", options=products['product_name'])
if st.button("Add to Cart"):
    if product_to_add not in st.session_state.user_cart:
        st.session_state.user_cart.append(product_to_add)

st.write("### 🛍 Current Cart")
st.write(st.session_state.user_cart)

# Show recommendations
if st.session_state.user_cart:
    recs = recommend_for_new_user(st.session_state.user_cart)  # use imported function
    st.write("### 🔝 Top Recommendations for You")
    st.dataframe(recs.reset_index(drop=True))

