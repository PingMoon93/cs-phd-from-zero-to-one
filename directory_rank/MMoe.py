# MMoe两个任务：logit——lable用于预测是否喜欢，二分类；
# pred_rating回归任务，用于预测具体评分
# 专家网络根据视频一般用4或者8尝试，这里用4个
# 每个专家网络有两层MLP
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def load_data(path_rating = "ml-100k/u.data",   
              path_user = "ml-100k/u.user",
              path_item = "ml-100k/u.item",
              test_ratio = 0.2,random_seed = 42):
    ratings = pd.read_csv(
    path_rating,
    sep="\t",
    names=["user_id","item_id","rating","timestamp"],
    engine="python"
)

    ratings = ratings[["user_id","item_id","rating"]]
    #用户特征
    users = pd.read_csv(path_user,sep='|',
                        names = ['user_id','age','gender','occupation','zip'],
                        engine='python')[['user_id','age','gender','occupation']]
    users['gender_code'] = (users['gender'].astype(str).str.upper() == 'F').astype(np.int64)
    users['occupation_code'] = users['occupation'].astype('category').cat.codes.astype(np.int64)
    n_occupation = users['occupation_code'].max()+1 #????

    #物品特征
    items_raw = pd.read_csv(path_item,sep='|',header=None, engine='python',encoding='latin-1')
    genre_col = list(range(items_raw.shape[1]-19,items_raw.shape[1]))
    items = items_raw.iloc[:,[0]+genre_col].copy()
    items.columns = ['item_id'] + [f'g{j}' for j in range(19)]

    #合并
    df = ratings.merge(users[['user_id','gender_code','occupation_code']],on='user_id',how='left')
    df = df.merge(items[['item_id'] + [f'g{j}' for j in range(19)]],on='item_id',how='left')    
    #连续编码
    ucat = df['user_id'].astype('category')
    icat = df['item_id'].astype('category')
    df['u'] = ucat.cat.codes.astype(np.int64)
    df['i'] = icat.cat.codes.astype(np.int64)
    n_users = ucat.cat.categories.size
    n_items = icat.cat.categories.size  
    #构造二分类
    df['label'] = (df['rating'] >= 3.0 ).astype(np.float64)
    #打乱+切分
    df = df.sample(frac=1.0,random_state=random_seed).reset_index(drop=True)
    n = len(df); n_train = int(n*(1-test_ratio))
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_test = df.iloc[n_train:].reset_index(drop=True)

    def pack(split_df):
        return {
            'user': split_df['u'].to_numpy(np.int64),
            'item': split_df['i'].to_numpy(np.int64),
            'rating': split_df['rating'].to_numpy(np.float32),
            'label': split_df['label'].to_numpy(np.float64),
            'gender': split_df['gender_code'].to_numpy(np.int64),
            'occupation': split_df['occupation_code'].to_numpy(np.int64),
            'genre': split_df[[f'g{j}' for j in range(19)]].to_numpy(np.float32)
        }
    train_data = pack(df_train)
    test_data = pack(df_test)       
    #保存模型所需要的特征维度信息
    meta = {
        'n_users': n_users,
        'n_items': n_items,
        'n_occupation': n_occupation,
        'n_genre': 19
    }
    return train_data, test_data, meta
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

class RatingDS(Dataset):
    def __init__(self, data):
        self.u = torch.as_tensor(data['user'],dtype=torch.long)
        self.i = torch.as_tensor(data['item'],dtype=torch.long)
        self.r = torch.as_tensor(data['rating'],dtype=torch.float32)
        self.label = torch.as_tensor(data['label'],dtype=torch.float32)
        self.gender = torch.as_tensor(data['gender'],dtype=torch.long)
        self.occupation = torch.as_tensor(data['occupation'],dtype=torch.long)
        self.genre = torch.as_tensor(data['genre'],dtype=torch.float32) 

    def __len__(self):
        return len(self.r)
    def __getitem__(self, idx):
        return (self.u[idx], self.i[idx], self.r[idx], self.label[idx],
                self.gender[idx], self.occupation[idx], self.genre[idx])    
    
class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = nn.ReLU()
    def forward(self, x):
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        return h 
    #每个专家塔两层MLP
class TaskTower(nn.Module):
    def __init__(self, input_dim, hidden_dim=64,output_dim=1,final_activation=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.final_activation = final_activation
    def forward(self, x):
        h = self.act1(self.fc1(x))
        h = self.fc2(h)
        if self.final_activation is not None:
            h = self.final_activation(h)
        return h
class MMoe(nn.Module):
    def __init__(
            self,
            n_users:int,
            n_items:int,
            n_occupation:int,
            n_genre:int,
            id_emb_dim:int=64,
            gender_emb_dim:int=8,
            occupation_emb_dim:int=16,
            genre_token_dim:int=8,
            n_experts:int=4,#修改专家网络，acc基本没有变化
            expert_out:int=64,
            expert_hidden:int=128,
            tower_hidden:int=64,
            init_std:float=0.01
    ):
        super().__init__()
        #embedding层
        self.user_id_emb = nn.Embedding(n_users, id_emb_dim)
        nn.init.normal_(self.user_id_emb.weight, std=init_std)
        self.item_id_emb = nn.Embedding(n_items, id_emb_dim)
        nn.init.normal_(self.item_id_emb.weight, std=init_std)
        self.gender_emb = nn.Embedding(2, gender_emb_dim)
        nn.init.normal_(self.gender_emb.weight, std=init_std)
        self.occupation_emb = nn.Embedding(n_occupation, occupation_emb_dim)
        nn.init.normal_(self.occupation_emb.weight, std=init_std)
        self.genre_token = nn.Embedding(n_genre, genre_token_dim)
        nn.init.normal_(self.genre_token.weight, std=init_std)
        #将embedding拼接起来，做个线性投影到统一维度
        self.user_proj = nn.Linear(
            id_emb_dim +gender_emb_dim + occupation_emb_dim,
            64, bias=True
        )
        self.item_proj = nn.Linear(
            id_emb_dim + genre_token_dim,
            64, bias=True
        )
        #最终输入给MMoe的特征维度
        mmoe_input_dim = 128
        #专家网络
        self.experts = nn.ModuleList([
            MLPExpert(input_dim=mmoe_input_dim, hidden_dim=expert_hidden,output_dim = expert_out)
            for _ in range(n_experts)
        ])
        #回归塔，输出评分
        self.gate_rating = nn.Linear(mmoe_input_dim, n_experts)
        #分类塔，输出是否喜欢
        self.gate_label = nn.Linear(mmoe_input_dim, n_experts)
        
        self.tower_rating = TaskTower(
            input_dim = expert_out,
            hidden_dim = tower_hidden,
            output_dim = 2,
            final_activation = None
        )
        self.tower_label = TaskTower(
            input_dim = expert_out,
            hidden_dim = tower_hidden,
            output_dim = 1,
            final_activation = None
        )
    def _encode_user(self, u, gender, occupation):
        uid_vec = self.user_id_emb(u)
        gender_vec = self.gender_emb(gender)
        occupation_vec = self.occupation_emb(occupation)
        user_all = torch.cat([uid_vec, gender_vec, occupation_vec], dim=-1)
        user_feat = torch.relu(self.user_proj(user_all))
        return user_feat
    def _encode_item(self, i, genre_vec ):
        iid_vec = self.item_id_emb(i)
        g_table = self.genre_token.weight
        genre_wighted = torch.matmul(genre_vec, g_table)
        
        item_all = torch.cat([iid_vec, genre_wighted], dim=-1)
        item_feat = torch.relu(self.item_proj(item_all))
        return item_feat
    def _mmoe(self, fused_feat):
       expert_outputs = []
       for expert in self.experts:
        expert_outputs.append(expert(fused_feat))    # 每个 [B, expert_out]
        expert_outputs = torch.stack(expert_outputs, dim=0).transpose(0, 1)
        gate_r = torch.softmax(self.gate_rating(fused_feat), dim=-1)  # [B, n_experts]
        gate_n = torch.softmax(self.gate_label(fused_feat),  dim=-1)  # [B, n_experts]

        gate_r_exp = gate_r.unsqueeze(-1)  # [B, n_experts, 1]
        gate_n_exp = gate_n.unsqueeze(-1)  # [B, n_experts, 1]


        task_rating_feat = torch.sum(expert_outputs * gate_r_exp, dim=1)  # [B, expert_out]
        task_label_feat  = torch.sum(expert_outputs * gate_n_exp, dim=1)  # [B, expert_out]

        return task_rating_feat, task_label_feat
    
    def forward(self, u, i, gender, occupation, genre_vec):
        user_feat = self._encode_user(u,gender,occupation)
        item_feat = self._encode_item(i,genre_vec)
        fused = torch.cat([user_feat, item_feat], dim=-1)
        task_rating_feat, task_label_feat = self._mmoe(fused)
        pred_rating = self.tower_rating(task_rating_feat).squeeze(-1)
        logit_label = self.tower_label(task_label_feat).squeeze(-1)
        return pred_rating, logit_label
    
#train & eval
def train_one_epoch(model,loader, opt, l2=0.0, alpha=1.0, beta=1.0):
    model.train()
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    loss_sum, n_rows = 0.0, 0
    for u,i,r,label,gender,occupation,genre in loader:
        u = u.to(device)
        i = i.to(device)
        r = r.to(device)
        label = label.to(device)        
        gender = gender.to(device)
        occupation = occupation.to(device)
        genre = genre.to(device)
        pred_rating, logit_label = model(u,i,gender,occupation,genre)
        loss_r = mse_loss(pred_rating, r)
        loss_n = bce_loss(logit_label, label)
    reg = 0.0
    if l2 and l2 > 0.0:
        reg_term = 0.0
        for p in model.parameters():
            reg_term  = reg_term + p.pow(2).sum()
        reg = l2 * reg_term
    loss = alpha * loss_r + beta * loss_n + reg
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    opt.step()

    bs = u.size(0)
    loss_sum += loss.item() * bs
    n_rows += bs
    return float(loss_sum / max(1, n_rows)) 
@torch.no_grad()
def evaluate(model,loader):
    model.eval()
    mse_sum,n1 = 0.0,0
    correct,n2 = 0,0
    for u,i,r,label,gender,occupation,genre in loader:
        u = u.to(device)
        i = i.to(device)
        r = r.to(device)
        label = label.to(device)  
        gender = gender.to(device)
        occupation = occupation.to(device)
        genre = genre.to(device)
        pred_rating, logit_label = model(u,i,gender,occupation,genre)
        mse_sum += torch.sum((pred_rating - r)**2).item()
        n1 += len(r)
        prob = torch.sigmoid(logit_label)
        pred_label = (prob >= 0.5).float()
        correct += torch.sum((pred_label == label).float()).item()
        n2 += len(label)
    rmse_rating = np.sqrt(mse_sum / max(1,n1))   
    acc_label = correct / max(1,n2)
    return rmse_rating, acc_label   
def main():
    train, test, meta = load_data()
    n_users = meta['n_users']
    n_items = meta['n_items']
    n_occupation = meta['n_occupation']
    n_genre = meta['n_genre']           
    train_loader = DataLoader(
        RatingDS(train),
        batch_size=128,
        shuffle=True)
    test_loader = DataLoader(
        RatingDS(test),
        batch_size=128,
        shuffle=False)
    model = MMoe(
        n_users=n_users,
        n_items=n_items,
        n_occupation=n_occupation,
        n_genre=n_genre,
        id_emb_dim=64,
        gender_emb_dim=8,
        occupation_emb_dim=16,
        genre_token_dim=16,
        n_experts=4,
        expert_hidden=128,
        expert_out=64,
        tower_hidden=64,
        init_std=0.01       
    ).to(device)  

    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0) 

    best_rmse = 1e9
    best_acc = 1

    for epoch in range(1, 6):
        train_loss = train_one_epoch(model, train_loader, opt, l2=1e-6, alpha=1.0, beta=1.0)
        rmse_rating, acc_label = evaluate(model, test_loader)
        if rmse_rating < best_rmse:
            best_rmse = rmse_rating
        if acc_label > best_acc:
            best_acc = acc_label
        print(f"Epoch {epoch:02d}: Train Loss={train_loss:.4f}, "
              f"Test RMSE={rmse_rating:.4f} (Best={best_rmse:.4f}), "
              f"Test Acc={acc_label:.4f} (Best={best_acc:.4f})")    
if __name__ == "__main__":
    main()
  