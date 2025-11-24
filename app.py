import sys
import os
import streamlit as st
import pandas as pd
import base64
from src.recommendation import recommend_for_new_user

st.set_page_config(page_title="🛒 Smart Grocery Recommender", layout="wide")

def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        [data-testid="stHeader"] {{
            background-color: rgba(0,0,0,0);
        }}
        div.stButton > button:first-child {{
            background-color: #34a853;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 0.5em 1em;
        }}
        /* --- limit dropdown height to show 3 visible items --- */
        div[data-baseweb="select"] ul {{
            max-height: 6em; /* about 3 items */
            overflow-y: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("utils/shopping_bg.jpeg")
st.markdown(
    """
    <style>

    h1 {
        color: #002f06 !important;
    }
    h2, h3 {
        color: #11823b !important;
        font-weight: 700 !important;
    }

    label, p, span, .stMarkdown {
        color: #02231c !important;
        font-weight: 600 !important;
    }

    .stButton > button {
        background-color: #34a853 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        padding: 0.5em 1em !important;
        border: none !important;
        transition: 0.2s !important;
    }

    .stButton > button:hover {
        background-color: #2e8b48 !important;
        transform: scale(1.03);
    }

    [data-baseweb="select"] > div {
        background-color: rgba(52,168,83,0.12) !important;
        border-radius: 8px !important;
    }

    [data-baseweb="select"] ul li {
        color: #222 !important;
    }

    .stTextInput > div > div > input {
        background-color: rgba(255,255,255,0.10) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #34a853 !important;
    }

    .dataframe tbody td,
    .dataframe thead th {
        color: #f5f5f5 !important;
    }

    a, a:link, a:visited {
        color: #F4B400 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource(show_spinner=False)
def load_products():
    products = pd.read_csv("data/products.csv")
    aisles = pd.read_csv("data/aisles.csv")
    departments = pd.read_csv("data/departments.csv")
    products = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")
    return products

products = load_products()

@st.cache_data(show_spinner=False)
def get_recommendations(cart):
    if not cart:
        return pd.DataFrame()
    return recommend_for_new_user(cart)

st.title("🥦 Smart Grocery Recommender")
st.write("Select items from the dropdown and discover your next favorite groceries instantly! 🛍️")

if 'user_cart' not in st.session_state:
    st.session_state.user_cart = []

product_to_add = st.selectbox(
    "🛒 Choose a product to add:",
    options=products["product_name"].tolist(),
    key="product_select_scroll"
)

if st.button("➕ Add to Cart"):
    if product_to_add and product_to_add not in st.session_state.user_cart:
        st.session_state.user_cart.append(product_to_add)

st.markdown("### 🛍️ Your Cart")
if st.session_state.user_cart:
    st.write(", ".join(st.session_state.user_cart))
else:
    st.info("Your cart is empty. Start by adding a product above! 🍎")

if st.session_state.user_cart:
    recs = get_recommendations(st.session_state.user_cart)
    if not recs.empty:
        st.markdown("### 🔝 Top Recommendations for You 🍇")
        st.dataframe(recs.reset_index(drop=True))
    else:
        st.warning("No recommendations found yet! Try adding more products.")
