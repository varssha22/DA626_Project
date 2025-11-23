import pandas as pd
import numpy as np

def get_cbf_scores(selected_products, products, nn, product_features):
    cbf_scores = np.zeros(product_features.shape[0])

    if not selected_products:
        return cbf_scores

    latest_product = selected_products[-1]

    if latest_product not in products["product_name"].values:
        return cbf_scores

    latest_idx = products[products["product_name"] == latest_product].index[0]

    distances, indices = nn.kneighbors(product_features[latest_idx], n_neighbors=len(products))
    similarities = 1 - distances.flatten()

    cbf_scores[indices] = similarities

    if cbf_scores.max() > 0:
        cbf_scores /= cbf_scores.max()

    return cbf_scores
