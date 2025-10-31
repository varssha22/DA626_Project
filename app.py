import sys
import os
import streamlit as st
import pandas as pd
import base64
from src.recommendation import recommend_for_new_user

# --- PAGE CONFIG ---
st.set_page_config(page_title="🛒 Smart Grocery Recommender", layout="wide")

# --- BACKGROUND IMAGE ---
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
# --- FONT & COLOR STYLING ---
st.markdown(
    """
    <style>
    /* ----- Specific heading colors ----- */
    /* Title produced by st.title() */
    .stApp h1 {
        color: #FFD700 !important;   /* gold for main title */
    }
    /* Subheadings like st.markdown("### ...") */
    .stApp h2, .stApp h3 {
        color: #34a853 !important;   /* green for subheadings */
        font-weight: 700;
    }

    /* ----- Labels and small headings ----- */
    /* Label text for widgets (selectbox labels, etc.) */
    label, .stMarkdown, .stText {
        color: #FFA500 !important;   /* orange */
        font-weight: 600;
    }

    /* ----- Buttons ----- */
    div.stButton > button:first-child {
        background-color: #34a853 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        padding: 0.5em 1em !important;
        border: none !important;
        transition: 0.2s !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #2e8b48 !important;
        transform: scale(1.03);
    }

    /* ----- Selectbox inner box styling ----- */
    /* This targets the visible select box container */
    div[data-baseweb="select"] > div {
        background-color: rgba(52,168,83,0.12) !important; /* soft green bg */
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    /* When the options list is shown, style the list items */
    div[data-baseweb="select"] ul li {
        color: #222 !important; /* dark text in the dropdown options */
    }

    /* ----- Text input styling ----- */
    input[type="text"], .stTextInput input {
        background-color: rgba(255,255,255,0.10) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #34a853 !important;
    }

    /* ----- DataFrame/table text color ----- */
    .stDataFrame table td, .stDataFrame table th {
        color: #f5f5f5 !important;
    }

    /* ----- small helper: keep links visible ----- */
    a, a:link, a:visited {
        color: #F4B400 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- CACHE PRODUCT DATA ---
@st.cache_resource(show_spinner=False)
def load_products():
    products = pd.read_csv("data/products.csv")
    aisles = pd.read_csv("data/aisles.csv")
    departments = pd.read_csv("data/departments.csv")
    products = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")
    return products

products = load_products()

# --- CACHE RECOMMENDATIONS ---
@st.cache_data(show_spinner=False)
def get_recommendations(cart):
    if not cart:
        return pd.DataFrame()
    return recommend_for_new_user(cart)

# --- TITLE & HEADER ---
st.title("🥦 Smart Grocery Recommender")
st.write("Select items from the dropdown and discover your next favorite groceries instantly! 🛍️")

# --- CART SESSION ---
if 'user_cart' not in st.session_state:
    st.session_state.user_cart = []

# --- PRODUCT DROPDOWN (SCROLLABLE) ---
product_to_add = st.selectbox(
    "🛒 Choose a product to add:",
    options=products["product_name"].tolist(),
    key="product_select_scroll"
)

# --- ADD TO CART BUTTON ---
if st.button("➕ Add to Cart"):
    if product_to_add and product_to_add not in st.session_state.user_cart:
        st.session_state.user_cart.append(product_to_add)

# --- DISPLAY CART ---
st.markdown("### 🛍️ Your Cart")
if st.session_state.user_cart:
    st.write(", ".join(st.session_state.user_cart))
else:
    st.info("Your cart is empty. Start by adding a product above! 🍎")

# --- DISPLAY RECOMMENDATIONS ---
if st.session_state.user_cart:
    recs = get_recommendations(st.session_state.user_cart)
    if not recs.empty:
        st.markdown("### 🔝 Top Recommendations for You 🍇")
        st.dataframe(recs.reset_index(drop=True))
    else:
        st.warning("No recommendations found yet! Try adding more products.")
