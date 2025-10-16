#在矩阵补充基础上，加入user和item的features，使用MLP，构成双塔模型
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
              n_occupation = users["occupation_code"].max() +1#计算职业信息的种类

              #item信息，添加item种类
              item_raw = pd.read_csv(
                   path_item, sep="|",header=None,engine="python",encoding="latin-1"    
              )
              genre_col = list(range(item_raw.shape[1]-19, item_raw.shape[1]))
              items = item_raw.iloc[:, [0]+ genre_col].copy()
              items.columns = ["item_id"] +[f"g{j}" for j in range(19)]

              genre_mat = items[[f"g{j}" for j in range(19)]].astype(np.float32).values
              n_genre = genre_mat.shape[1]

              df = ratings.merge(users[["user_id","gender_code","occupation_code"]],on="user_id",how="left")
              df = df.merge(items[["item_id"] + [f"g{j}" for j in range(19)]],on="item_id",how="left")
              ucat = df["user_id"].astype("category")
              icat = df["item_id"].astype("category")
              df["u"] = ucat.cat.codes.astype(np.int64)
              df["i"] = icat.cat.codes.astype(np.int64)
              n_users = ucat.cat.categories.size
              n_items = icat.cat.categories.size

              df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
              n = len(df); n_train = int(n * (1 - test_ratio))
              train_df = df.iloc[:n_train]
              test_df = df.iloc[n_train:]

              def pack(split_df):
                      return{
                              "user":split_df["u"].to_numpy(np.int64),
                              "item":split_df["i"].to_numpy(np.int64),
                              "rating":split_df["rating"].to_numpy(np.float32),
                              "gender":split_df["gender_code"].to_numpy(np.int64),
                              "occupation":split_df["occupation_code"].to_numpy(np.int64),
                              "genre":split_df[[f"g{j}" for j in range(19)]].to_numpy(np.float32),                      
                      }
              train = pack(train_df)
              test = pack(test_df)

              meta = {
                      "n_users":n_users,
                      "n_items":n_items,
                      "n_occupation":n_occupation,
                      "n_genre":n_genre
              }
              return train, test, meta
device = "cuda" if torch.cuda.is_available() else "cpu"

class RatingDS(Dataset):
        def __init__(self,data):
                self.u = torch.as_tensor(data["user"],dtype=torch.long)
                self.i = torch.as_tensor(data["item"],dtype=torch.long)
                self.r = torch.as_tensor(data["rating"],dtype=torch.float32)
                self.gender = torch.as_tensor(data["gender"],dtype=torch.long)
                self.occupation = torch.as_tensor(data["occupation"],dtype=torch.long)
                self.genre = torch.as_tensor(data["genre"],dtype=torch.float32)

        def __len__(self):return len (self.r)

        def __getitem__(self, idx):
                return (self.u[idx],
                        self.i[idx],
                        self.r[idx],
                        self.gender[idx],
                        self.occupation[idx],
                        self.genre[idx])
        
def MLP(in_dim:int, hidden:list[int], out_dim:int,dropout = 0.0):
        layers = []
        prev = in_dim
        for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                if dropout > 0:layers += [nn.Dropout(dropout)]
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
            dropout: float =0.0,
            init_std: float = 0.01
        ):
            super().__init__()
            #embeddings
            self.user_id_emb = nn.Embedding(n_users, id_emb_dim)
            nn.init.normal_(self.user_id_emb.weight, std=init_std)
            self.item_id_emb = nn.Embedding(n_items,id_emb_dim)
            nn.init.normal_(self.item_id_emb.weight, std=init_std)
            
            self.gender_emb = nn.Embedding(2, gender_emb_dim)
            nn.init.normal_(self.gender_emb.weight, std=init_std)

            self.occupation_emd = nn.Embedding(n_occupation,occupation_emb_dim)
            nn.init.normal_(self.occupation_emd.weight, std=init_std)

            self.genre_proj = MLP(n_genre, [64],genre_proj_dim, dropout=0.0)


            user_in_dim = id_emb_dim + gender_emb_dim +occupation_emb_dim
            item_in_dim = id_emb_dim + genre_proj_dim

            self.user_tower = MLP(user_in_dim, user_hidden,tower_dim,dropout=dropout)
            self.item_tower = MLP(item_in_dim, item_hidden,tower_dim,dropout=dropout)

        def forward(self, u, i, gender, occupation, genre_vec):
                ue = self.user_id_emb(u)
                ie = self.item_id_emb(i)

                ge = self.gender_emb(gender)
                oe = self.occupation_emd(occupation)
                gp = self.genre_proj(genre_vec)

                u_in = torch.cat([ue, ge, oe], dim=1)
                i_in = torch.cat([ie, gp],dim=1)

                uv = self.user_tower(u_in)
                iv = self.item_tower(i_in)

                dot = (uv * iv).sum(dim=1)
                pred = dot
                return pred
def train_one_epoch(model, loader, opt, l2=0.0):
        model.train()
        mse_sum, n= 0.0, 0
        for u, i, r, gender, occupation, genre in loader:
                u, i , r = u.to(device), i.to(device), r.to(device)
                gender, occupation, genre = gender.to(device), occupation.to(device), genre.to(device)

                pred = model(u,i,gender,occupation,genre)
                mse = torch.mean((pred - r) **2)

                reg = 0.0
                if l2 and l2 > 0:
                        reg_term = 0.0
                        for p in model.parameters():
                                reg_term = reg_term + p.pow(2).sum()
                        reg = l2 * reg_term

                loss = mse + reg
                opt.zero_grad()
                loss.backward()
                opt.step()

                mse_sum += mse.item() * len(r)
                n += len(r)
        return float(np.sqrt(mse_sum/n))
@torch.no_grad()
def evaluate(model, loader):
        model.eval()
        mse_sum, n =0.0, 0
        for u, i, r, gender, occupation, genre in loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                gender, occupation, genre = gender.to(device), occupation.to(device), genre.to(device)

                pred = model(u,i,gender,occupation,genre)
                mse_sum += torch.sum((pred - r) **2).item()
                n += len(r)
        return float(np.sqrt(mse_sum / n))

def main():
        train, test, meta = load_data()
        n_users, n_items = meta["n_users"], meta["n_items"]
        n_occupation, n_genre = meta["n_occupation"], meta["n_genre"]

        train_loader = DataLoader(RatingDS(train), batch_size=128, shuffle=True)
        test_loader = DataLoader(RatingDS(test), batch_size=128, shuffle=False)

        model = TwoTower(
                n_users=n_users, n_items=n_items,
                n_occupation=n_occupation, n_genre=n_genre,
                id_emb_dim=64, gender_emb_dim=8, occupation_emb_dim=16,
                genre_proj_dim=32, user_hidden=[128],item_hidden=[128],
                tower_dim=64, dropout=0.0, init_std=0.01
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

        best = 1e9
        for epoch in range(1,20):
                tr = train_one_epoch(model, train_loader, opt, l2=0.0)
                te = evaluate(model, test_loader)
                best = min(best, te)
                print(f"Epoch {epoch:02d} | train RMSE={tr:.4f} | test RMSE={te:.4f} | best={best:.4f}")
if __name__ == "__main__":
        main()
           
                        






            
       
       
       
       
