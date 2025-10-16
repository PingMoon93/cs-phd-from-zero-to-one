#在双塔模型的基础上加入负样本采样，采用简单粗暴的划分，将评分<3的定义为负样本；
#评估方法由回归改成分类，加入对数损失和AUC

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data  import Dataset, DataLoader


def load_data(path_rating="ml-100k/u.data",
              path_user = "ml-100k/u.user",
              path_item = "ml-100k/u.item",
              test_ratio = 0.2, seed = 42):
    ratings = pd.read_csv(
        path_rating, sep="\t",
        names=["user_id","item_id","rating","timestamp"],
        engine="python")[["user_id","item_id","rating"]]
    ratings["rating"] = ratings["rating"].astype(np.float32)

    
    users = pd.read_csv(
        path_user, sep="|",
        names=["user_id","age","gender","occupation","zip"],
        engine="python"
    )[["user_id","age","gender","occupation"]]
    users["gender_code"] = (users["gender"].astype(str).str.upper() == "F").astype(np.int64)
    users["occupation_code"] = users["occupation"].astype("category").cat.codes.astype(np.int64)
    n_occupation = users["occupation_code"].max() + 1

    
    item_raw = pd.read_csv(path_item, sep="|", header=None, engine="python", encoding="latin-1")
    genre_col = list(range(item_raw.shape[1]-19, item_raw.shape[1]))
    items = item_raw.iloc[:, [0]+genre_col].copy()
    items.columns = ["item_id"] + [f"g{j}" for j in range(19)]
    n_genre = 19

   
    df = ratings.merge(users[["user_id","gender_code","occupation_code"]], on="user_id", how="left")
    df = df.merge(items[["item_id"] + [f"g{j}" for j in range(19)]], on="item_id", how="left")

   
    ucat = df["user_id"].astype("category")
    icat = df["item_id"].astype("category")
    df["u"] = ucat.cat.codes.astype(np.int64)
    df["i"] = icat.cat.codes.astype(np.int64)
    n_users = ucat.cat.categories.size
    n_items = icat.cat.categories.size

    # 评分>=3为正样本，<3 为负样本，这个正负样本的思路有问题，后面再改变
    df["label"] = (df["rating"] >= 3).astype(np.float32)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df); n_train = int(n * (1 - test_ratio))
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    def pack(split_df):
        return {
            "user":      split_df["u"].to_numpy(np.int64),
            "item":      split_df["i"].to_numpy(np.int64),
            "rating":    split_df["rating"].to_numpy(np.float32),
            "label":     split_df["label"].to_numpy(np.float32),
            "gender":    split_df["gender_code"].to_numpy(np.int64),
            "occupation":split_df["occupation_code"].to_numpy(np.int64),
            "genre":     split_df[[f"g{j}" for j in range(19)]].to_numpy(np.float32),
        }, split_df

    train, train_df = pack(train_df)
    test,  test_df  = pack(test_df)

    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "n_occupation": n_occupation,
        "n_genre": n_genre
    }
    return train, test, meta, train_df, test_df

device = "cuda" if torch.cuda.is_available() else "cpu"

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

class PosOnlyWithNegSamplingDS(Dataset):
    
    def __init__(self, train_dict, n_neg=5, seed=42):
        self.rng = np.random.default_rng(seed)
        self.u_all = torch.as_tensor(train_dict["user"], dtype=torch.long)
        self.i_all = torch.as_tensor(train_dict["item"], dtype=torch.long)
        self.y_all = torch.as_tensor(train_dict["label"], dtype=torch.float32)
        self.gender_all = torch.as_tensor(train_dict["gender"], dtype=torch.long)
        self.occ_all = torch.as_tensor(train_dict["occupation"], dtype=torch.long)
        self.genre_all = torch.as_tensor(train_dict["genre"], dtype=torch.float32)

        # 正负样本
        self.pos_idx = np.where(train_dict["label"] == 1.0)[0]
        self.neg_idx = np.where(train_dict["label"] == 0.0)[0]
        self.n_neg = n_neg

        #用户负样本表
        self.user2negs = {}
        u_np = train_dict["user"]
        for idx in self.neg_idx:
            u = int(u_np[idx])
            self.user2negs.setdefault(u, []).append(idx)

        #全局负样本池
        self.neg_pool = self.neg_idx.tolist()

    def __len__(self):
        return len(self.pos_idx)

    def __getitem__(self, i):
        #返回一个正样本索引
        return int(self.pos_idx[i])

    def sample_negs_for_user(self, u, k):
        cand = self.user2negs.get(int(u), None)#存放每个用户的可用负样本池
        pool = cand if cand else self.neg_pool
        if len(pool) == 0:  #极端情况下
            return []
        if len(pool) >= k:
            return self.rng.choice(pool, size=k, replace=False).tolist()
        else:
            #不足则有放回补齐
            picked = pool.copy()
            need = k - len(pool)
            picked += self.rng.choice(pool, size=need, replace=True).tolist()
            return picked

def neg_sampling_collate(batch_pos_indices, ds: PosOnlyWithNegSamplingDS):
   
    u_list, i_list, g_list, o_list, ge_list, y_list = [], [], [], [], [], []
    k = ds.n_neg
    for pos_row in batch_pos_indices:
        # 正样本
        u = ds.u_all[pos_row]; i = ds.i_all[pos_row]
        g = ds.gender_all[pos_row]; o = ds.occ_all[pos_row]; ge = ds.genre_all[pos_row]
        u_list += [u]; i_list += [i]; g_list += [g]; o_list += [o]; ge_list += [ge]; y_list += [torch.tensor(1.0)]

        # 负样本（同一用户优先）
        neg_rows = ds.sample_negs_for_user(u, k)
        for nr in neg_rows:
            u_list += [ds.u_all[nr]]
            i_list += [ds.i_all[nr]]
            g_list += [ds.gender_all[nr]]
            o_list += [ds.occ_all[nr]]
            ge_list += [ds.genre_all[nr]]
            y_list += [torch.tensor(0.0)]

    u = torch.stack(u_list)
    i = torch.stack(i_list)
    g = torch.stack(g_list)
    o = torch.stack(o_list)
    ge = torch.stack(ge_list)
    y = torch.stack(y_list)
    return u, i, y, g, o, ge


def MLP(in_dim:int, hidden:list[int], out_dim:int, dropout = 0.0):
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout > 0: layers += [nn.Dropout(dropout)]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)

class TwoTower(nn.Module):
    def __init__(
        self,
        n_users:int,
        n_items:int,
        n_occupation:int,
        n_genre:int,
        id_emb_dim: int = 64,
        gender_emb_dim: int = 8,
        occupation_emb_dim: int = 16,
        genre_proj_dim: int = 32,
        user_hidden: list[int] = [128],
        item_hidden: list[int] = [128],
        tower_dim: int = 64,
        dropout: float = 0.0,
        init_std: float = 0.01
    ):
        super().__init__()
        # embeddings
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

        
        logit = (uv * iv).sum(dim=1)
        return logit

def bce_epoch(model, loader, opt=None, l2=0.0):
    train_mode = opt is not None
    if train_mode: model.train()
    else: model.eval()

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
    order = np.argsort(scores)#将评分按照位置排序
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)#创造长度数组，然后按照评分大小,评分大排后
    
    pos = labels == 1
    neg = labels == 0
    n_pos = np.sum(pos)
    n_neg = np.sum(neg)
   #考虑极端情况，需要考虑全是正样本或全是负样本的情况
    if n_pos == 0 or n_neg == 0:
        return float("nan")


    sum_ranks_pos = np.sum(ranks[pos])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def main():
    train, test, meta, train_df, test_df = load_data()
    n_users, n_items = meta["n_users"], meta["n_items"]
    n_occupation, n_genre = meta["n_occupation"], meta["n_genre"]

   
    n_neg = 2  #根据实验，=2时AUC最大
    train_ds = PosOnlyWithNegSamplingDS(train, n_neg=n_neg, seed=42)
    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True,
        collate_fn=lambda batch: neg_sampling_collate(batch, train_ds)
    )

    test_loader = DataLoader(RatingDSBinary(test), batch_size=512, shuffle=False)

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

    
