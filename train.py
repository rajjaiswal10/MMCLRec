
import os
import random
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# CONFIG / HYPERPARAMETERS
# =========================================================
SEED = 42
torch.backends.cudnn.benchmark = True
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

G_DIM      = 64
FUSED_DIM  = 64
NUM_LAYERS = 2

CL_WEIGHT  = 0.1
TEMP       = 0.5

LR         = 1e-3
REG_LAMBDA = 1e-4

EPOCHS     = 70
BATCH_SIZE = 8192
NUM_NEG    = 1
PATIENCE   = 10

USE_AMP = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# =========================================================
# LOAD DATA
# =========================================================
if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
    raise FileNotFoundError("train.csv or test.csv not found. Run split.py first.")

print("Loading train/test splits...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

enc_data = np.load(ENC_PATH, allow_pickle=True)
user_enc = LabelEncoder()
item_enc = LabelEncoder()
user_enc.classes_ = enc_data["user_classes"]
item_enc.classes_ = enc_data["item_classes"]

all_df    = pd.concat([train_df, test_df])
num_users = int(all_df["user"].nunique())
num_items = int(all_df["item"].nunique())
print(f"num_users={num_users} | num_items={num_items}")
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

train_mat = {}
for r in train_df.itertuples(index=False):
    train_mat.setdefault(int(r.user), set()).add(int(r.item))

test_dict = {}
for r in test_df.itertuples(index=False):
    test_dict.setdefault(int(r.user), set()).add(int(r.item))

print(f"Train users: {len(train_mat)} | Test users: {len(test_dict)}")


# =========================================================
# LOAD MULTIMODAL EMBEDDINGS
# =========================================================
print("Loading multimodal embeddings...")
image_emb_np = np.load(IMG_PATH)
desc_emb_np  = np.load(DESC_PATH)
meta_emb_np  = np.load(META_PATH)
print("Shapes:", image_emb_np.shape, desc_emb_np.shape, meta_emb_np.shape)

image_emb_t = torch.tensor(image_emb_np, dtype=torch.float32)
desc_emb_t  = torch.tensor(desc_emb_np,  dtype=torch.float32)
meta_emb_t  = torch.tensor(meta_emb_np,  dtype=torch.float32)


# =========================================================
# BUILD NORMALISED ADJACENCY MATRIX
# =========================================================
def build_norm_adj(train_mat, num_users, num_items, device):
    user_idx, item_idx = [], []
    for u, items in train_mat.items():
        for i in items:
            user_idx.append(u)
            item_idx.append(i + num_users)

    row = torch.tensor(user_idx + item_idx, dtype=torch.long)
    col = torch.tensor(item_idx + user_idx, dtype=torch.long)

    N       = num_users + num_items
    deg     = torch.bincount(row, minlength=N).float()
    deg_inv = deg.pow(-0.5)
    deg_inv[torch.isinf(deg_inv)] = 0.0
    norm    = deg_inv[row] * deg_inv[col]

    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        norm,
        (N, N),
        dtype=torch.float32,
        device=device,
    ).coalesce()

    return adj


print("Building adjacency matrix...")
adj_gpu = build_norm_adj(train_mat, num_users, num_items, DEVICE)
print(f"Adj: {adj_gpu.shape}  nnz={adj_gpu._nnz()}  "
      f"device={adj_gpu.device}  dtype={adj_gpu.dtype}")


# =========================================================
# BATCH SAMPLER
# =========================================================
all_users = np.array(list(train_mat.keys()))

train_lists = {
    u: np.array(list(items))
    for u, items in train_mat.items()
}

def sample_batch(
    train_mat,
    num_users,
    num_items,
    batch_size,
    num_neg=1,
):

    # sample users
    users = np.random.choice(
        all_users,
        size=batch_size,
        replace=True
    )

    # positive sampling
    pos_items = np.array([
        np.random.choice(train_lists[u])
        for u in users
    ])

    # negative sampling
    neg_items = np.random.randint(
        0,
        num_items,
        size=batch_size
    )

    # rejection sampling
    for idx, u in enumerate(users):

        while neg_items[idx] in train_mat[u]:

            neg_items[idx] = np.random.randint(num_items)

    return (
        torch.from_numpy(users).long(),
        torch.from_numpy(pos_items).long(),
        torch.from_numpy(neg_items).long(),
    )

# =========================================================
# MODEL
# =========================================================
class MMCLGCN(nn.Module):
    """
    LightGCN propagation -> multimodal fusion -> BPR + InfoNCE.

    All tensors are float32 throughout. AMP is disabled globally
    (USE_AMP=False) because torch.sparse.mm has no float16 CUDA kernel.
    """

    def __init__(
        self,
        num_users,
        num_items,
        adj,            
        image_emb,      # (M, img_dim)
        desc_emb,       # (M, txt_dim)
        meta_emb,       # (M, meta_dim)
        g_dim=64,
        fused_dim=64,
        num_layers=2,
    ):
        super().__init__()

        self.num_users  = num_users
        self.num_items  = num_items
        self.num_layers = num_layers

        # collaborative embeddings
        self.user_emb = nn.Embedding(num_users, g_dim)
        self.item_emb = nn.Embedding(num_items, g_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # fixed multimodal buffers (not parameters)
        self.register_buffer("image_emb", image_emb)
        self.register_buffer("desc_emb",  desc_emb)
        self.register_buffer("meta_emb",  meta_emb)
        self.adj = adj

        # learnable projections
        self.image_proj = nn.Linear(image_emb.shape[1], fused_dim)
        self.text_proj  = nn.Linear(desc_emb.shape[1],  fused_dim)
        self.meta_proj  = nn.Linear(meta_emb.shape[1],  fused_dim)
        self.graph_proj = nn.Linear(g_dim, fused_dim)

        # fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim * 4, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def _propagate(self):


        with torch.autocast(device_type="cuda", enabled=False):

            x = torch.cat(
                [
                    self.user_emb.weight.float(),
                    self.item_emb.weight.float()
                ],
                dim=0
            )

            adj = self.adj.float()

            all_embs = [x]

            for _ in range(self.num_layers):

                x = torch.sparse.mm(adj, x)

                all_embs.append(x)

            final = torch.stack(all_embs, dim=0).mean(dim=0)

        users_g = final[:self.num_users]
        items_g = final[self.num_users:]

        return users_g, items_g

    def _project_modalities(self):
        return (
            self.image_proj(self.image_emb),
            self.text_proj(self.desc_emb),
            self.meta_proj(self.meta_emb),
        )

    def _fuse(self, items_g):
        i_g_proj       = self.graph_proj(items_g)
        i_v, i_t, i_m = self._project_modalities()
        fused = self.fusion(
            torch.cat([i_g_proj, i_v, i_t, i_m], dim=1)
        )
        return fused, i_g_proj, i_v, i_t, i_m

    def forward(self):
        users_g, items_g                = self._propagate()
        fused, i_g_proj, i_v, i_t, i_m = self._fuse(items_g)
        return users_g, fused, i_g_proj, i_v, i_t, i_m


# =========================================================
# LOSS FUNCTIONS
# =========================================================
def bpr_loss(u_emb, pos_emb, neg_emb, reg_lambda=REG_LAMBDA):
    pos_scores = (u_emb * pos_emb).sum(dim=1)
    neg_scores = (u_emb * neg_emb).sum(dim=1)
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    reg  = (
        u_emb.norm(2).pow(2)
        + pos_emb.norm(2).pow(2)
        + neg_emb.norm(2).pow(2)
    ) / max(1, u_emb.size(0))
    return loss + reg_lambda * reg


def info_nce(x1, x2, temp=TEMP):
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    logits = torch.matmul(x1, x2.t()) / temp
    labels = torch.arange(x1.size(0), device=x1.device)
    return (
        F.cross_entropy(logits,     labels)
        + F.cross_entropy(logits.t(), labels)
    ) / 2.0


# =========================================================
# EVALUATION
# =========================================================
@torch.no_grad()
def evaluate_recall(model, train_mat, val_mat, k=20):
    model.eval()
    users_g, fused, *_ = model.forward()

    recall_total = 0
    user_count   = 0

    for user, ground_truth in val_mat.items():
        if not ground_truth:
            continue
        scores = torch.matmul(users_g[user], fused.t())
        train_items = list(train_mat.get(user, set()))
        if train_items:
            scores[train_items] = -1e9
        top_k = torch.topk(scores, k=k).indices.cpu().tolist()
        recall_total += len(set(top_k) & set(ground_truth)) / len(ground_truth)
        user_count   += 1

    return recall_total / user_count if user_count > 0 else 0.0


def train_mmclgcn(
    model,
    train_mat,
    val_mat,
    num_users,
    num_items,
    lr=LR,
    reg_lambda=REG_LAMBDA,
    epochs=EPOCHS,
    batch_size=8192,
    num_neg=NUM_NEG,
    cl_weight=CL_WEIGHT,
    temp=TEMP,
    patience=PATIENCE,
):

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        fused=True if torch.cuda.is_available() else False
    )

    best_recall = 0.0
    trigger = 0

    num_interactions = sum(len(v) for v in train_mat.values())

    steps = max(1, num_interactions // batch_size)

    print(f"Training: epochs={epochs} steps/epoch≈{steps}")

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0.0

        pbar = tqdm(range(steps), desc=f"Epoch {epoch}/{epochs}")

        for _ in pbar:

            # ============================================
            # SAMPLE
            # ============================================
            users, pos, neg = sample_batch(
                train_mat,
                num_users,
                num_items,
                batch_size,
                num_neg
            )

            users = users.to(DEVICE, non_blocking=True)
            pos = pos.to(DEVICE, non_blocking=True)
            neg = neg.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # ============================================
            # FORWARD
            # ============================================
            users_g, fused, i_g_proj, i_v, i_t, i_m = model()

            # ============================================
            # BPR
            # ============================================
            u_emb = users_g[users]

            pos_emb = fused[pos]
            neg_emb = fused[neg]

            loss_bpr = bpr_loss(
                u_emb,
                pos_emb,
                neg_emb,
                reg_lambda,
            )

            # ============================================
            # CONTRASTIVE
            # ============================================
            items_unique = torch.unique(
                torch.cat((pos, neg))
            )

            g_sel = i_g_proj[items_unique]
            v_sel = i_v[items_unique]
            t_sel = i_t[items_unique]
            m_sel = i_m[items_unique]

            loss_cl = (
                info_nce(g_sel, v_sel, temp)
                + info_nce(g_sel, t_sel, temp)
                + info_nce(g_sel, m_sel, temp)
            )

            # ============================================
            # TOTAL LOSS
            # ============================================
            loss = loss_bpr + cl_weight * loss_cl

            # ============================================
            # BACKWARD
            # ============================================
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "gpu": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB"
            })

        avg_loss = total_loss / steps

        print(f"\nEpoch {epoch} | Loss: {avg_loss:.4f}")

        # ============================================
        # VALIDATION
        # ============================================
        if epoch % 10 == 0:

            recall = evaluate_recall(
                model,
                train_mat,
                val_mat,
                k=20
            )

            print(f"Recall@20: {recall:.4f}")

            if recall > best_recall:

                best_recall = recall
                trigger = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                    },
                    "best_model.pt"
                )

                print("Best model saved.")

            else:

                trigger += 1

                if trigger >= patience:
                    print("Early stopping")
                    break

    return model

# =========================================================
# BUILD & TRAIN
# =========================================================
model = MMCLGCN(
    num_users=num_users,
    num_items=num_items,
    adj=adj_gpu,
    image_emb=image_emb_t,
    desc_emb=desc_emb_t,
    meta_emb=meta_emb_t,
    g_dim=G_DIM,
    fused_dim=FUSED_DIM,
    num_layers=NUM_LAYERS,
).to(DEVICE)

print("Model parameters:", sum(p.numel() for p in model.parameters()))

if torch.cuda.is_available():
    print("user_emb :", model.user_emb.weight.device, model.user_emb.weight.dtype)
    print("adj      :", model.adj.device,             model.adj.dtype)
    print("image_emb:", model.image_emb.device,       model.image_emb.dtype)

model = train_mmclgcn(
    model=model,
    train_mat=train_mat,
    val_mat=test_dict,
    num_users=num_users,
    num_items=num_items,
    lr=LR,
    reg_lambda=REG_LAMBDA,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    num_neg=NUM_NEG,
    cl_weight=CL_WEIGHT,
    temp=TEMP,
    patience=PATIENCE,
)

