# Recommendation-System-Design-Project

# 🚀 How to Run the Project

This guide explains how to set up, train, and run the full **Personalized Retail Recommender System** including **Collaborative Filtering (CF)**, **Content-Based Filtering (CBF)**, and **BERT4Rec Sequential Recommendation**.

---

## 📥 1. Clone the Repository

```bash
git clone https://github.com/varssha22/DA626_Project.git
cd DA626_Project
```

---

## 📦 2. Install Dependencies

It is recommended to create a new environment.

### Using pip
```bash
pip install -r requirements.txt
```

### Using Conda
```bash
conda create -n recommender python=3.10 -y
conda activate recommender
pip install -r requirements.txt
```

---

## 📁 3. Prepare the Dataset

Place the following files inside the `data/` folder:

```
data/
│── aisles.csv
│── departments.csv
│── products.csv
│── orders.csv
│── order_products__prior.parquet  (or .csv)
│── train_sequences.pkl
│── val_sequences.pkl
│── test_sequences.pkl
```

Ensure all filenames match exactly.

---

## 🏗 4. Train All Models

The project provides **three training scripts**.

---

### 🔹 4.1 Train Collaborative Filtering (CF)

Builds:
- User-Item Matrix  
- Item-Item Similarity Matrix  

Run:

```bash
python train_cf.py
```

Outputs saved in:

```
saved_cf/
│── user_item_matrix.pkl
│── item_similarity_matrix.pkl
```

---

### 🔹 4.2 Train Content-Based Filtering (CBF)

Builds:
- Word2Vec model  
- One-hot encoders  
- MinMax scaler  
- Sparse feature matrix  
- Trained cosine KNN model  

Run:

```bash
python train_cbf.py
```

Outputs saved in:

```
saved_cbf/
│── product_w2v.model
│── encoder_aisle.pkl
│── encoder_dept.pkl
│── scaler.pkl
│── product_features.npz
│── nn_model.pkl
```

---

### 🔹 4.3 Train BERT4Rec Sequential Model

Make sure the three sequence files exist in `data/`:

```
train_sequences.pkl
val_sequences.pkl
test_sequences.pkl
```

Run BERT4Rec training:

```bash
python train_bert4rec.py
```

Best checkpoint saved in:

```
checkpoints/bert4rec_best.keras
```

---

## 🧪 5. Run Evaluation (NDCG, Precision, Recall, HitRate)

If you want to evaluate the models:

```bash
python evaluate.py
```

---

## 🧪 6. Run Streamlit app

If you want to see the top recommendations:

```bash
streamlit run app.py
```

---

## 📂 7. Project Structure

```
DA626_Project/
│── data/
│── saved_cf/
│── saved_cbf/
│── checkpoints/
│── train_cf.py
│── train_cbf.py
│── train_bert4rec.py
│── bert4rec_architecture.py
│── main.py
│── utils/
│── README.md
│── requirements.txt
```

---
