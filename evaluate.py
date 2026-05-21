# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_DIM = 64
FUSED_DIM = 64
NUM_LAYERS = 2

TOPK = [5, 10, 20]

print("Device:", DEVICE)


# =========================================================
# PATHS
# =========================================================
DRIVE_ROOT = "."

TRAIN_PATH = os.path.join(DRIVE_ROOT, "train.csv")
TEST_PATH  = os.path.join(DRIVE_ROOT, "test.csv")

ENC_PATH   = os.path.join(DRIVE_ROOT, "encoders.npz")

IMG_PATH   = os.path.join(DRIVE_ROOT, "image_embeddings.npy")
DESC_PATH  = os.path.join(DRIVE_ROOT, "desc_embeddings.npy")
META_PATH  = os.path.join(DRIVE_ROOT, "metadata_embeddings.npy")

MODEL_PATH = os.path.join(DRIVE_ROOT, "best_model.pt")


# =========================================================
# LOAD DATA
# =========================================================
print("Loading data...")

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

enc_data = np.load(ENC_PATH, allow_pickle=True)

user_enc = LabelEncoder()
item_enc = LabelEncoder()

user_enc.classes_ = enc_data["user_classes"]
item_enc.classes_ = enc_data["item_classes"]

all_df = pd.concat([train_df, test_df])

num_users = int(all_df["user"].nunique())
num_items = int(all_df["item"].nunique())

print(f"num_users={num_users}")
print(f"num_items={num_items}")


# =========================================================
# TRAIN / TEST DICTS
# =========================================================
train_mat = {}

for r in train_df.itertuples(index=False):

    train_mat.setdefault(
        int(r.user),
        set()
    ).add(int(r.item))

test_dict = {}

for r in test_df.itertuples(index=False):

    test_dict.setdefault(
        int(r.user),
        set()
    ).add(int(r.item))


# =========================================================
# LOAD EMBEDDINGS
# =========================================================
print("Loading multimodal embeddings...")

image_emb = torch.tensor(
    np.load(IMG_PATH),
    dtype=torch.float32,
    device=DEVICE
)

desc_emb = torch.tensor(
    np.load(DESC_PATH),
    dtype=torch.float32,
    device=DEVICE
)

meta_emb = torch.tensor(
    np.load(META_PATH),
    dtype=torch.float32,
    device=DEVICE
)


# =========================================================
# BUILD RATING-AWARE ADJ
# =========================================================
def build_norm_adj(train_df, num_users, num_items, device):

    user_idx = []
    item_idx = []
    edge_weight = []

    for r in train_df.itertuples(index=False):

        u = int(r.user)
        i = int(r.item)

        rating = float(r.rating) / 5.0

        user_idx.append(u)
        item_idx.append(i + num_users)

        edge_weight.append(rating)

    row = torch.tensor(
        user_idx + item_idx,
        dtype=torch.long,
        device=device,
    )

    col = torch.tensor(
        item_idx + user_idx,
        dtype=torch.long,
        device=device,
    )

    edge_weight = torch.tensor(
        edge_weight + edge_weight,
        dtype=torch.float32,
        device=device,
    )

    N = num_users + num_items

    deg = torch.zeros(
        N,
        dtype=torch.float32,
        device=device,
    )

    deg.index_add_(0, row, edge_weight)

    deg_inv_sqrt = deg.pow(-0.5)

    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    norm_weight = (
        edge_weight
        * deg_inv_sqrt[row]
        * deg_inv_sqrt[col]
    )

    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        norm_weight,
        (N, N),
        dtype=torch.float32,
        device=device,
    ).coalesce().to_sparse_csr()

    return adj


print("Building adjacency...")

adj = build_norm_adj(
    train_df,
    num_users,
    num_items,
    DEVICE
)


# =========================================================
# MODEL
# =========================================================
class MMCLGCN(nn.Module):

    def __init__(
        self,
        num_users,
        num_items,
        adj,
        image_emb,
        desc_emb,
        meta_emb,
        g_dim=64,
        fused_dim=64,
        num_layers=2,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        self.user_emb = nn.Embedding(num_users, g_dim)
        self.item_emb = nn.Embedding(num_items, g_dim)

        self.register_buffer("image_emb", image_emb)
        self.register_buffer("desc_emb", desc_emb)
        self.register_buffer("meta_emb", meta_emb)

        self.adj = adj

        self.image_proj = nn.Linear(image_emb.shape[1], fused_dim)
        self.text_proj  = nn.Linear(desc_emb.shape[1], fused_dim)
        self.meta_proj  = nn.Linear(meta_emb.shape[1], fused_dim)
        self.graph_proj = nn.Linear(g_dim, fused_dim)

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim * 4, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def _propagate(self):

        x = torch.cat([
            self.user_emb.weight.float(),
            self.item_emb.weight.float()
        ], dim=0)

        all_embs = [x]

        for _ in range(self.num_layers):

            x = torch.sparse.mm(self.adj, x)

            all_embs.append(x)

        final = torch.stack(all_embs, dim=0).mean(dim=0)

        users_g = final[:self.num_users]
        items_g = final[self.num_users:]

        return users_g, items_g

    def forward(self):

        users_g, items_g = self._propagate()

        i_g_proj = self.graph_proj(items_g)

        i_v = self.image_proj(self.image_emb)
        i_t = self.text_proj(self.desc_emb)
        i_m = self.meta_proj(self.meta_emb)

        fused = self.fusion(
            torch.cat([i_g_proj, i_v, i_t, i_m], dim=1)
        )

        return users_g, fused


# =========================================================
# LOAD MODEL
# =========================================================
print("Loading model...")

model = MMCLGCN(
    num_users=num_users,
    num_items=num_items,
    adj=adj,
    image_emb=image_emb,
    desc_emb=desc_emb,
    meta_emb=meta_emb,
    g_dim=G_DIM,
    fused_dim=FUSED_DIM,
    num_layers=NUM_LAYERS,
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded.")


# =========================================================
# METRICS
# =========================================================
def precision_at_k(gt, pred, k):

    pred_k = pred[:k]

    return len(set(pred_k) & set(gt)) / k


def recall_at_k(gt, pred, k):

    pred_k = pred[:k]

    return len(set(pred_k) & set(gt)) / len(gt)


def ndcg_at_k(gt, pred, k):

    pred_k = pred[:k]

    dcg = 0.0

    for idx, item in enumerate(pred_k):

        if item in gt:

            dcg += 1.0 / np.log2(idx + 2)

    idcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(min(len(gt), k))
    )

    return dcg / idcg if idcg > 0 else 0.0


# =========================================================
# EVALUATION
# =========================================================
print("\nEvaluating...")

metrics = {
    k: {
        "precision": [],
        "recall": [],
        "ndcg": []
    }
    for k in TOPK
}

with torch.no_grad():

    users_g, fused = model()

    for user, gt_items in tqdm(test_dict.items()):

        scores = torch.matmul(
            users_g[user],
            fused.t()
        )

        # remove train items
        train_items = list(
            train_mat.get(user, set())
        )

        if train_items:
            scores[train_items] = -1e9

        top_items = torch.topk(
            scores,
            k=max(TOPK)
        ).indices.cpu().numpy()

        for k in TOPK:

            metrics[k]["precision"].append(
                precision_at_k(gt_items, top_items, k)
            )

            metrics[k]["recall"].append(
                recall_at_k(gt_items, top_items, k)
            )

            metrics[k]["ndcg"].append(
                ndcg_at_k(gt_items, top_items, k)
            )


# =========================================================
# RESULTS
# =========================================================
print("\n================ RESULTS ================\n")

for k in TOPK:

    precision = np.mean(metrics[k]["precision"])
    recall    = np.mean(metrics[k]["recall"])
    ndcg      = np.mean(metrics[k]["ndcg"])

    print(f"K = {k}")

    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}   : {recall:.4f}")
    print(f"NDCG@{k}     : {ndcg:.4f}")

    print()