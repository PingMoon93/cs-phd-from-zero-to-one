
#矩阵补充，目前不常用:
# 仅用了用户id和物品id以及rating，没有使用用户和物品features;
# 负样本选取方式不对；使用内积作为相似度，不如余弦相似度，用平方损失函数，不如交叉熵损失，后续可根据此提高

#虽然但是，我认为这个模型可以用于股票推荐，但需要考虑features（主要是题材和板块）

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def load_data(path="ml-100k/u.data", test_ratio = 0.2, seed = 42):
       df = pd.read_csv(
        path, sep="\t",
        names=["user_id", "item_id", "rating","timestamp"],
        engine="python"
    )[["user_id","item_id","rating"]]#注意这里的取数逻辑，列出所有列的名称再取需要不会出错
       
       df["rating"] = df["rating"].astype(np.float32)    #rating是回归目标，int类型无法进行梯度传播   
       ucat = df["user_id"].astype("category")#user_id和item_id转成类别变量
       icat = df["item_id"].astype("category")
       df["u"] = df["user_id"].astype("category").cat.codes.astype(np.int64)#将原始的id映射成一个整数，并存到新列df['u']
       df["i"] = df["item_id"].astype("category").cat.codes.astype(np.int64)
       n_users = ucat.cat.categories.size#统计用户数量
       n_items = icat.cat.categories.size#统计物品总数
       
       
       df = df.sample(frac=1.0,random_state = seed).reset_index(drop=True) #frac是抽样比例，为1时表示抽样比例100%，打乱所有数据
       n = len(df)
       n_train = int(n * (1 - test_ratio))#设置训练集行数
       train_df = df.iloc[:n_train]#iloc按整数位数索引
       test_df = df.iloc[n_train:]#用行号切片划分数据

       train = {
              "user":train_df["u"].to_numpy(np.int64),
              "item":train_df["i"].to_numpy(np.int64),
              "rating":train_df["rating"].to_numpy(np.float32),
       }#转换为numpy数组
       test = {
        "user": test_df["u"].to_numpy(np.int64),
        "item": test_df["i"].to_numpy(np.int64),
        "rating": test_df["rating"].to_numpy(np.float32),
    }
       return train, test, n_users, n_items

device = "cuda" if torch.cuda.is_available() else "cpu"
class RatingsDS(Dataset):#定义pytorch数据集
    def __init__(self, data):
        self.u = torch.as_tensor(data["user"], dtype=torch.long)
        self.i = torch.as_tensor(data["item"], dtype=torch.long)
        self.r = torch.as_tensor(data["rating"], dtype=torch.float32)
    def __len__(self): return len(self.r)#返回评分个数
    def __getitem__(self, idx): return self.u[idx], self.i[idx], self.r[idx]

class MF(nn.Module):
    def __init__(self, n_users, n_items, d=64, init_std=0.01):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        #初始化权重
        nn.init.normal_(self.user_emb.weight, std=init_std)
        nn.init.normal_(self.item_emb.weight, std=init_std)
       
    def forward(self, u, i):
        pu = self.user_emb(u) #从用户embedding表中取出索引为u的行   
        qi = self.item_emb(i)             
        
        pred = (pu * qi).sum(1) 
        return pred
def train_one_epoch(model, loader, opt, l2=1e-5):
    
    model.train()
    mse_sum, n = 0.0, 0
    for u, i, r in loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        pred = model(u, i)
        #计算MSE
        mse = torch.mean((pred - r) ** 2)
        loss = mse 
        opt.zero_grad()#清空梯度
        loss.backward()#反向传播计算梯度
        opt.step()#优化器更新参数
        #注意这里的写法，最后一个batch的样本数可能不同
        mse_sum += mse.item() * len(r)
        n += len(r)
    return float(np.sqrt(mse_sum / n))

def evaluate(model, loader):
    model.eval()
    mse_sum, n = 0.0, 0
    with torch.no_grad():  # 禁用梯度计算
        for u, i, r in loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i)
            mse_sum += torch.sum((pred - r) ** 2).item()
            n += len(r)
    return float(np.sqrt(mse_sum / n))

def main():
    train, test, n_users, n_items = load_data(path="ml-100k/u.data")

    train_loader = DataLoader(RatingsDS(train), batch_size=128, shuffle=True)
    test_loader   = DataLoader(RatingsDS(test),   batch_size=128, shuffle=False)

    model = MF(n_users, n_items, d=64).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    best = 1e9
    for epoch in range(1, 20):
        tr = train_one_epoch(model, train_loader, opt, l2=1e-5)
        te = evaluate(model, test_loader)
        best = min(best, te)
        print(f"Epoch {epoch:02d} | train RMSE={tr:.4f} | test RMSE={te:.4f} | best={best:.4f}")

if __name__ == "__main__":
    main()