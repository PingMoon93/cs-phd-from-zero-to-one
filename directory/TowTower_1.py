# 所有评分的item定为正样本，每个正样本随机采2个负样本
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =============== 数据加载：构造稠密ID & 返回整表 item_genre_table =================
def load_data(path_rating="ml-100k/u.data",
              path_user="ml-100k/u.user",
              path_item="ml-100k/u.item",
              test_ratio=0.2, seed=42):
    # ratings
    ratings = pd.read_csv(
        path_rating, sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python"
    )[["user_id", "item_id", "rating"]]
    ratings["rating"] = ratings["rating"].astype(np.float32)

    # users
    users = pd.read_csv(
        path_user, sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        engine="python"
    )[["user_id", "age", "gender", "occupation"]]
    users["gender_code"] = (users["gender"].astype(str).str.upper() == "F").astype(np.int64)
    users["occupation_code"] = users["occupation"].astype("category").cat.codes.astype(np.int64)
    n_occupation = users["occupation_code"].max() + 1

    # items（19个genre one-hot）
    item_raw = pd.read_csv(path_item, sep="|", header=None, engine="python", encoding="latin-1")
    genre_col = list(range(item_raw.shape[1] - 19, item_raw.shape[1]))
    items = item_raw.iloc[:, [0] + genre_col].copy()
    items.columns = ["item_id"] + [f"g{j}" for j in range(19)]
    n_genre = 19

    # 合并特征
    df = ratings.merge(users[["user_id", "gender_code", "occupation_code"]], on="user_id", how="left")
    df = df.merge(items[["item_id"] + [f"g{j}" for j in range(19)]], on="item_id", how="left")

    # 构造稠密ID
    ucat = df["user_id"].astype("category")
    icat = df["item_id"].astype("category")
    df["u"] = ucat.cat.codes.astype(np.int64)
    df["i"] = icat.cat.codes.astype(np.int64)
    n_users = ucat.cat.categories.size
    n_items = icat.cat.categories.size

    # —— 关键改动：隐式反馈 —— 只要出现评分就视作正样本（1.0）
    df["label"] = 1.0

    # 为“所有稠密物品ID”准备 genre 整表，避免负采样时 KeyError
    genre_cols = [f"g{j}" for j in range(19)]
    cats = icat.cat.categories  # 稠密ID 0..n_items-1 对应的原始 item_id 顺序
    item_genre_table = (
        items.set_index("item_id")
             .reindex(cats)[genre_cols]
             .fillna(0.0)
             .to_numpy(np.float32)        # 形状：(n_items, 19)
    )

    # 切分训练/测试（按交互行随机打乱切分）
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * (1 - test_ratio))
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    def pack(split_df):
        return {
            "user": split_df["u"].to_numpy(np.int64),
            "item": split_df["i"].to_numpy(np.int64),
            "rating": split_df["rating"].to_numpy(np.float32),
            "label": split_df["label"].to_numpy(np.float32),  # 全部是1.0（正样本）
            "gender": split_df["gender_code"].to_numpy(np.int64),
            "occupation": split_df["occupation_code"].to_numpy(np.int64),
            "genre": split_df[[f"g{j}" for j in range(19)]].to_numpy(np.float32),
        }, split_df

    train, train_df = pack(train_df)
    test, test_df = pack(test_df)

    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "n_occupation": users["occupation_code"].max() + 1,
        "n_genre": n_genre,
        "item_genre_table": item_genre_table
    }
    return train, test, meta, train_df, test_df


device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============（保留）显式二分类Dataset：测试对比用，不在本版本中使用 ===============
class RatingDSBinary(Dataset):
    def __init__(self, data):
        self.u = torch.as_tensor(data["user"], dtype=torch.long)
        self.i = torch.as_tensor(data["item"], dtype=torch.long)
        self.y = torch.as_tensor(data["label"], dtype=torch.float32)
        self.gender = torch.as_tensor(data["gender"], dtype=torch.long)
        self.occupation = torch.as_tensor(data["occupation"], dtype=torch.long)
        self.genre = torch.as_tensor(data["genre"], dtype=torch.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return (self.u[idx], self.i[idx], self.y[idx],
                self.gender[idx], self.occupation[idx], self.genre[idx])


# ======================== 隐式反馈：仅正样本 + 未交互负采样 =========================
class PosOnlyWithNegSamplingImplicitDS(Dataset):
    """
    - 只存正样本（已评分的 user-item）。
    - collate 时为每个正例，从“该用户未交互”的物品中采k个负例。
    - 需要 item_genre_table（n_items x n_genre），按稠密物品ID索引。
    """
    def __init__(self, train_dict, n_items: int, item_genre_table: np.ndarray,
                 n_neg: int = 2, seed: int = 42):
        self.rng = np.random.default_rng(seed)

        # 正样本张量
        self.u_pos = torch.as_tensor(train_dict["user"], dtype=torch.long)
        self.i_pos = torch.as_tensor(train_dict["item"], dtype=torch.long)
        self.gender_pos = torch.as_tensor(train_dict["gender"], dtype=torch.long)
        self.occ_pos = torch.as_tensor(train_dict["occupation"], dtype=torch.long)
        self.genre_pos = torch.as_tensor(train_dict["genre"], dtype=torch.float32)

        self.n_items = int(n_items)
        self.n_neg = int(n_neg)

        # 整表物品genre，按稠密ID直接索引
        self.item_genre_table = torch.as_tensor(item_genre_table, dtype=torch.float32)

        # 用户 -> 该用户看过的物品集合（用于“未交互负采样”）
        u_np = train_dict["user"]
        i_np = train_dict["item"]
        self.user_pos_items = {}
        for u, it in zip(u_np, i_np):
            self.user_pos_items.setdefault(int(u), set()).add(int(it))

        # 用户基础属性（性别/职业），用于负样本行快速填充
        self.user_gender = {}
        self.user_occ = {}
        for idx in range(len(u_np)):
            u = int(u_np[idx])
            if u not in self.user_gender:
                self.user_gender[u] = int(train_dict["gender"][idx])
                self.user_occ[u] = int(train_dict["occupation"][idx])

        # 正样本索引（全部，因为全是1）
        self.pos_idx = np.arange(len(u_np), dtype=np.int64)

    def __len__(self):
        return len(self.pos_idx)

    def __getitem__(self, i):
        # 返回一个正样本的行号
        return int(self.pos_idx[i])

    def sample_neg_items_for_user(self, u: int, k: int):
        """从该用户未交互的物品集合中采K个负样本（稠密物品ID）"""
        pos_set = self.user_pos_items.get(int(u), set())
        picked = []
        tried = 0
        max_try = 10 * k + 100  # 简单防护，避免极端死循环
        while len(picked) < k and tried < max_try:
            need = k - len(picked)
            cand = self.rng.integers(low=0, high=self.n_items, size=need * 2)
            for it in cand:
                it = int(it)
                if it not in pos_set:
                    picked.append(it)
                    if len(picked) == k:
                        break
            tried += 1

        if len(picked) < k:
            # 兜底：构造未交互集合的列表（或全体有放回）
            fallback = [x for x in range(self.n_items) if x not in pos_set]
            if len(fallback) == 0:
                # 极端：用户看过所有物品 -> 全体有放回
                fallback = list(range(self.n_items))
            extra = self.rng.choice(fallback, size=k - len(picked), replace=True).tolist()
            picked += extra
        return picked


def neg_sampling_collate_implicit(batch_pos_indices, ds: PosOnlyWithNegSamplingImplicitDS):
    """
    collate_fn：对每个正样本 (u, i_pos)，附加 k 个未交互负样本 (u, i_neg)。
    """
    u_list, i_list, g_list, o_list, ge_list, y_list = [], [], [], [], [], []
    k = ds.n_neg
    for pos_row in batch_pos_indices:
        u = ds.u_pos[pos_row]
        i = ds.i_pos[pos_row]

        # 正例
        g = ds.gender_pos[pos_row]
        o = ds.occ_pos[pos_row]
        ge = ds.genre_pos[pos_row]
        u_list.append(u); i_list.append(i); g_list.append(g); o_list.append(o); ge_list.append(ge)
        y_list.append(torch.tensor(1.0))

        # 负例：未交互采样
        neg_items = ds.sample_neg_items_for_user(int(u), k)
        ug = torch.tensor(ds.user_gender[int(u)], dtype=torch.long)
        uo = torch.tensor(ds.user_occ[int(u)], dtype=torch.long)
        for it in neg_items:
            u_list.append(u)  # 同一用户
            i_list.append(torch.tensor(it, dtype=torch.long))
            g_list.append(ug)
            o_list.append(uo)
            ge_list.append(ds.item_genre_table[it])  # 直接按稠密ID索引
            y_list.append(torch.tensor(0.0))

    u = torch.stack(u_list)
    i = torch.stack(i_list)
    g = torch.stack(g_list)
    o = torch.stack(o_list)
    ge = torch.stack(ge_list)
    y = torch.stack(y_list)
    return u, i, y, g, o, ge


# =============================== 模型：双塔 ===============================
def MLP(in_dim: int, hidden: list, out_dim: int, dropout=0.0):
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout > 0: layers += [nn.Dropout(dropout)]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


class TwoTower(nn.Module):
    def __init__(self,
                 n_users: int, n_items: int,
                 n_occupation: int, n_genre: int,
                 id_emb_dim: int = 64,
                 gender_emb_dim: int = 8,
                 occupation_emb_dim: int = 16,
                 genre_proj_dim: int = 32,
                 user_hidden: list = [128],
                 item_hidden: list = [128],
                 tower_dim: int = 64,
                 dropout: float = 0.0,
                 init_std: float = 0.01):
        super().__init__()
        # Embeddings
        self.user_id_emb = nn.Embedding(n_users, id_emb_dim)
        nn.init.normal_(self.user_id_emb.weight, std=init_std)
        self.item_id_emb = nn.Embedding(n_items, id_emb_dim)
        nn.init.normal_(self.item_id_emb.weight, std=init_std)

        self.gender_emb = nn.Embedding(2, gender_emb_dim)
        nn.init.normal_(self.gender_emb.weight, std=init_std)
        self.occupation_emb = nn.Embedding(n_occupation, occupation_emb_dim)
        nn.init.normal_(self.occupation_emb.weight, std=init_std)

        self.genre_proj = MLP(n_genre, [64], genre_proj_dim, dropout=0.0)

        user_in_dim = id_emb_dim + gender_emb_dim + occupation_emb_dim
        item_in_dim = id_emb_dim + genre_proj_dim

        self.user_tower = MLP(user_in_dim, user_hidden, tower_dim, dropout=dropout)
        self.item_tower = MLP(item_in_dim, item_hidden, tower_dim, dropout=dropout)

    def forward(self, u, i, gender, occupation, genre_vec):
        ue = self.user_id_emb(u)
        ie = self.item_id_emb(i)
        ge = self.gender_emb(gender)
        oe = self.occupation_emb(occupation)
        gp = self.genre_proj(genre_vec)
        u_in = torch.cat([ue, ge, oe], dim=1)
        i_in = torch.cat([ie, gp], dim=1)
        uv = self.user_tower(u_in)
        iv = self.item_tower(i_in)
        logit = (uv * iv).sum(dim=1)  # 内积
        return logit


# =============================== 训练/评估循环 ===============================
def bce_epoch(model, loader, opt=None, l2=0.0):
    train_mode = opt is not None
    model.train() if train_mode else model.eval()

    bce_loss = nn.BCEWithLogitsLoss()
    loss_sum, n_ex = 0.0, 0
    all_logits, all_labels = [], []

    for u, i, y, gender, occupation, genre in loader:
        u, i, y = u.to(device), i.to(device), y.to(device)
        gender, occupation, genre = gender.to(device), occupation.to(device), genre.to(device)

        logits = model(u, i, gender, occupation, genre)

        reg = torch.tensor(0.0, device=device)
        if l2 and l2 > 0:
            reg_term = torch.tensor(0.0, device=device)
            for p in model.parameters():
                reg_term = reg_term + p.pow(2).sum()
            reg = l2 * reg_term

        loss = bce_loss(logits, y) + reg

        if train_mode:
            opt.zero_grad()
            loss.backward()
            opt.step()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n_ex += bs
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    mean_loss = float(loss_sum / max(1, n_ex))
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    auc = compute_auc_from_logits(logits, labels)
    return mean_loss, auc


def compute_auc_from_logits(logits, labels):
    scores = logits
    order = np.argsort(scores)  # 升序
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    pos = labels == 1
    neg = labels == 0
    n_pos = np.sum(pos)
    n_neg = np.sum(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_ranks_pos = np.sum(ranks[pos])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# =================================== 主函数 ===================================
def main():
    train, test, meta, train_df, test_df = load_data()
    n_users, n_items = meta["n_users"], meta["n_items"]
    n_occupation, n_genre = meta["n_occupation"], meta["n_genre"]
    item_genre_table = meta["item_genre_table"]

    # 负采样个数（可调）
    n_neg_train = 2    # 训练：每正样本配2个未交互负样本
    n_neg_eval = 50    # 测试：每正样本配更多负样本，AUC更稳定

    # 训练集（隐式负采样）
    train_ds = PosOnlyWithNegSamplingImplicitDS(
        train, n_items=n_items, item_genre_table=item_genre_table,
        n_neg=n_neg_train, seed=42
    )
    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True,
        collate_fn=lambda batch: neg_sampling_collate_implicit(batch, train_ds)
    )

    # 测试集（同样隐式负采样；否则AUC没有负类会为NaN）
    test_ds = PosOnlyWithNegSamplingImplicitDS(
        test, n_items=n_items, item_genre_table=item_genre_table,
        n_neg=n_neg_eval, seed=123
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        collate_fn=lambda batch: neg_sampling_collate_implicit(batch, test_ds)
    )

    # 模型
    model = TwoTower(
        n_users=n_users, n_items=n_items,
        n_occupation=n_occupation, n_genre=n_genre,
        id_emb_dim=64, gender_emb_dim=8, occupation_emb_dim=16,
        genre_proj_dim=32, user_hidden=[128], item_hidden=[128],
        tower_dim=64, dropout=0.0, init_std=0.01
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

    best_auc = 0.0
    for epoch in range(1, 8):
        tr_loss, tr_auc = bce_epoch(model, train_loader, opt=opt, l2=0.0)
        te_loss, te_auc = bce_epoch(model, test_loader, opt=None)

        best_auc = max(best_auc, te_auc if not np.isnan(te_auc) else 0.0)
        print(f"Epoch {epoch:02d} | "
              f"train LogLoss={tr_loss:.4f} AUC={tr_auc:.4f} | "
              f"test LogLoss={te_loss:.4f} AUC={te_auc:.4f} | best AUC={best_auc:.4f}")


if __name__ == "__main__":
    main()
