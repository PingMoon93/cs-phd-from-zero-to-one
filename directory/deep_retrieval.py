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
    #用户信息，除了id外，添加性别、年龄以及职业
    users = pd.read_csv(
        path_user, sep="|",
        names=["user_id","age","gender","occupation","zip"],
        engine="python"
    )[["user_id","age","gender","occupation"]]
    #性别编码
    users["gender_code"] = (users["gender"].astype(str).str.upper() == "F").astype(np.int64)
    #职业编码
    users["occupation_code"] = users["occupation"].astype("category").cat.codes.astype(np.int64)
    n_occupation = users["occupation_code"].max() + 1
    #item信息，添加item种类
    item_raw = pd.read_csv(
        path_item, sep="|", header=None, engine="python", encoding="latin-1"
    )
    genre_col = list(range(item_raw.shape[1]-19, item_raw.shape[1]))
    items = item_raw.iloc[:, [0] + genre_col].copy()
    items.columns = ["item_id"] + [f"g{j}" for j in range(19)]

    genre_mat = items[[f"g{j}" for j in range(19)]].astype(np.float32).values
    n_genre = genre_mat.shape[1]
    #合并，转换
    df = ratings.merge(users[["user_id","gender_code","occupation_code"]], on="user_id", how="left")
    df = df.merge(items[["item_id"] + [f"g{j}" for j in range(19)]], on="item_id", how="left")
    ucat = df["user_id"].astype("category")
    icat = df["item_id"].astype("category")
    df["u"] = ucat.cat.codes.astype(np.int64)
    df["i"] = icat.cat.codes.astype(np.int64)
    n_users = ucat.cat.categories.size
    n_items = icat.cat.categories.size

    #原始id连续编码
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df); n_train = int(n * (1 - test_ratio))
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    #切分
    def pack(split_df):
        return {
            "user": split_df["u"].to_numpy(np.int64),
            "item": split_df["i"].to_numpy(np.int64),
            "rating": split_df["rating"].to_numpy(np.float32),
            "gender": split_df["gender_code"].to_numpy(np.int64),
            "occupation": split_df["occupation_code"].to_numpy(np.int64),
            "genre": split_df[[f"g{j}" for j in range(19)]].to_numpy(np.float32),
        }

    train = pack(train_df)
    test = pack(test_df)

    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "n_occupation": n_occupation,
        "n_genre": 19
    }
    return train, test, meta

device = "cuda" if torch.cuda.is_available() else "cpu"

class RatingDS(Dataset):
    def __init__(self, data):
        self.u = torch.as_tensor(data["user"], dtype=torch.long)
        self.i = torch.as_tensor(data["item"], dtype=torch.long)
        self.r = torch.as_tensor(data["rating"], dtype=torch.float32)
        self.gender = torch.as_tensor(data["gender"], dtype=torch.long)
        self.occupation = torch.as_tensor(data["occupation"], dtype=torch.long)
        self.genre = torch.as_tensor(data["genre"], dtype=torch.float32)

    def __len__(self): return len(self.r)

    def __getitem__(self, idx):
        return (self.u[idx],
                self.i[idx],
                self.r[idx],
                self.gender[idx],
                self.occupation[idx],
                self.genre[idx])

class DeepRetrieval(nn.Module):
    def __init__(
        self,
        n_users:int,
        n_items:int,
        n_occupation:int,
        n_genre:int,
        id_emb_dim: int = 64,
        gender_emb_dim: int = 8,
        occupation_emb_dim: int = 16,
        token_dim: int = 64,#token应该和主要的embedding接近
        init_std: float = 0.01
    ):
        super().__init__()
        #embedding
        self.user_id_emb = nn.Embedding(n_users, id_emb_dim)
        nn.init.normal_(self.user_id_emb.weight, std=init_std)

        self.item_id_emb = nn.Embedding(n_items, id_emb_dim)
        nn.init.normal_(self.item_id_emb.weight, std=init_std)

        self.gender_emb = nn.Embedding(2, gender_emb_dim)
        nn.init.normal_(self.gender_emb.weight, std=init_std)

        self.occupation_emd = nn.Embedding(n_occupation, occupation_emb_dim)
        nn.init.normal_(self.occupation_emd.weight, std=init_std)

        self.genre_token = nn.Embedding(n_genre, token_dim)
        nn.init.normal_(self.genre_token.weight, std=init_std)

        #映射到统一 token 维度
        self.user_id_proj = nn.Linear(id_emb_dim, token_dim, bias=False)
        self.item_id_proj = nn.Linear(id_emb_dim, token_dim, bias=False)
        self.gender_proj  = nn.Linear(gender_emb_dim, token_dim, bias=False)
        self.occup_proj   = nn.Linear(occupation_emb_dim, token_dim, bias=False)

        # 显式评分回归用的缩放/偏置（仅评估时使用）
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _l2norm(x): 
        return x / (x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12))

    def _build_user_tokens(self, u, gender, occupation):
        u_id_tok = self.user_id_proj(self.user_id_emb(u))               # [B, D]
        g_tok    = self.gender_proj(self.gender_emb(gender))            # [B, D]
        o_tok    = self.occup_proj(self.occupation_emd(occupation))     # [B, D]
        U = torch.stack([u_id_tok, g_tok, o_tok], dim=1)                # [B, 3, D]
        return self._l2norm(U)

    def _build_item_tokens(self, i, genre_vec):
        B = i.size(0)
        i_id_tok = self.item_id_proj(self.item_id_emb(i))               # [B, D]
        G_table  = self.genre_token.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 19, D]
        I = torch.cat([i_id_tok.unsqueeze(1), G_table], dim=1)          # [B, 20, D]
        I = self._l2norm(I)
        mask_item = torch.cat(
            [torch.ones(B, 1, device=genre_vec.device, dtype=torch.bool),
             (genre_vec > 0.5)], dim=1
        )                                                               # [B, 20]
        return I, mask_item

    def _maxsim(self, U, I, mask_item):
        """
        U: [Bu, 3, D], I: [Bi, 20, D], mask_item: [Bi, 20] (bool)
        return: scores [Bu, Bi]
        """
        # 相似度张量：[Bu, Bi, 3, 20]
        sim = torch.matmul(U.unsqueeze(1), I.transpose(1, 2).unsqueeze(0))

        # mask 无效的物品 tokens
        # mask_item -> [1, Bi, 1, 20] 进行广播
        mask = mask_item.unsqueeze(0).unsqueeze(2)  # [1, Bi, 1, 20]
        neg_inf = torch.full_like(sim, float("-inf"))
        sim = torch.where(mask, sim, neg_inf)

        # 对每个用户 token 在物品 tokens 上取 MaxSim -> [Bu, Bi, 3]
        sim_max, _ = sim.max(dim=3)
        # 再对用户 tokens 求和 -> [Bu, Bi]
        scores = sim_max.sum(dim=2)
        return scores

    def score_matrix(self, u_rows, gender_rows, occup_rows, i_cols, genre_cols):
    
        U = self._build_user_tokens(u_rows, gender_rows, occup_rows)     # [Bu, 3, D]
        I, mask_item = self._build_item_tokens(i_cols, genre_cols)       # [Bi, 20, D], [Bi, 20]
        S = self._maxsim(U, I, mask_item)                                # [Bu, Bi]
        return S

    def forward(self, u, i, gender, occupation, genre_vec):
    
        U = self._build_user_tokens(u, gender, occupation)               # [B, 3, D]
        I, mask_item = self._build_item_tokens(i, genre_vec)             # [B, 20, D], [B, 20]
        S = self._maxsim(U, I, mask_item)                                # [B, B]
        diag = torch.arange(S.size(0), device=S.device)
        score = S[diag, diag]
        pred = self.alpha * score + self.beta
        return pred

def train_one_epoch(model, loader, opt, l2=0.0, tau: float = 0.7):
    model.train()
    ce = nn.CrossEntropyLoss()
    loss_sum, n_rows = 0.0, 0

    for u, i, r, gender, occupation, genre in loader:
        u = u.to(device); i = i.to(device); r = r.to(device)
        gender = gender.to(device); occupation = occupation.to(device); genre = genre.to(device)

        # 选出正样本行
        mask_pos = (r >= 4.0)
        if mask_pos.sum().item() == 0:
            continue

        rows_u = u[mask_pos]
        rows_g = gender[mask_pos]
        rows_o = occupation[mask_pos]

        cols_i = i
        cols_genre = genre

        # 打分矩阵 [Bp, B]
        S = model.score_matrix(rows_u, rows_g, rows_o, cols_i, cols_genre) / tau

        # 目标列索引（正样本所在的列）
        batch_idx = torch.arange(u.size(0), device=device)
        target_cols = batch_idx[mask_pos]  # [Bp], in [0..B-1]

        # L2 正则（可选）
        reg = 0.0
        if l2 and l2 > 0:
            reg_term = 0.0
            for p in model.parameters():
                reg_term = reg_term + p.pow(2).sum()
            reg = l2 * reg_term

        loss = ce(S, target_cols) + reg

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        loss_sum += loss.item() * S.size(0)
        n_rows += S.size(0)

    avg_loss = float(loss_sum / max(1, n_rows))
    return avg_loss

@torch.no_grad()
def evaluate(model, loader):
    """
    仍计算 RMSE（用 forward 的对角得分线性映射到评分）——可作为训练趋势参考。
    """
    model.eval()
    mse_sum, n = 0.0, 0
    for u, i, r, gender, occupation, genre in loader:
        u = u.to(device); i = i.to(device); r = r.to(device)
        gender = gender.to(device); occupation = occupation.to(device); genre = genre.to(device)

        pred = model(u, i, gender, occupation, genre)
        mse_sum += torch.sum((pred - r) ** 2).item()
        n += len(r)
    return float(np.sqrt(mse_sum / max(1, n)))

# -----------------------------
# Main
# -----------------------------
def main():
    train, test, meta = load_data()
    n_users, n_items = meta["n_users"], meta["n_items"]
    n_occupation, n_genre = meta["n_occupation"], meta["n_genre"]

    train_loader = DataLoader(RatingDS(train), batch_size=128, shuffle=True)
    test_loader = DataLoader(RatingDS(test), batch_size=128, shuffle=False)

    model = DeepRetrieval(
        n_users=n_users, n_items=n_items,
        n_occupation=n_occupation, n_genre=n_genre,
        id_emb_dim=64, gender_emb_dim=8, occupation_emb_dim=16,
        token_dim=64, init_std=0.01
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    best = 1e9
    for epoch in range(1, 20):
        tr = train_one_epoch(model, train_loader, opt, l2=0.0, tau=0.7)
        te = evaluate(model, test_loader)
        best = min(best, te)
        print(f"Epoch {epoch:02d} | contrastive loss={tr:.4f} | test RMSE={te:.4f} | best={best:.4f}")

if __name__ == "__main__":
    main()
