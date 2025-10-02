
# User CF 的基本原理：兴趣相似的用户更可能喜欢相似的物品
from collections import defaultdict
import math
import pandas as pd
from operator import itemgetter

def load_data(path="ml-100k/u.data"):
    df = pd.read_csv(path, sep="\t", names=['user_id', 'item_id', 'rating', 'timestamp'])
    data = {}
    for row in df.itertuples(index=False):
        if row.rating >= 4:
            data.setdefault(row.user_id, set()).add(row.item_id)
    return data

train_data = load_data()

#以item为主键构建用户表，实现item到user的数据集
item_users = {}
for u, items in train_data.items():
    for i in items:
        item_users.setdefault(i, set()).add(u)

# 共现矩阵，替换掉item就是user
co_matrix = {}
user_degree = defaultdict(int)
for i, users in item_users.items():
    for u in users:
        user_degree[u] += 1#用户u喜欢的物品个数
    for u in users:
        items_u = train_data[u] #用户u喜欢的物品集合
        weight = 1.0 / math.log(1+len(users))#降低重度用户贡献太多的虚假相似度
        for v in users:
            if u == v:
                continue
            co_matrix.setdefault(u, defaultdict(float))
            co_matrix[u][v] += weight
userSim = {} #定义user的余弦相似度
for u, related in co_matrix.items():#这里的related是和物品i共现的其他物品
    userSim.setdefault(u, {})
    for v, cuv in related.items():
        denom = math.sqrt(user_degree[u] * user_degree[v])
        userSim[u][v] = 0.0 if denom == 0 else cuv / denom


# 给用户做推荐：取相似度最高的 K 个邻居，把他们的物品按相似度加权汇总
def recommend(user, N=5, K=10):
    if user not in train_data:
        return {}  # 冷启动：目标用户不在训练集
    interacted = train_data[user]
    neighbors = userSim.get(user, {})
    if not neighbors:
        return {}  # 没有相似邻居

    recs = defaultdict(float)
    for v, sim in sorted(neighbors.items(), key=itemgetter(1), reverse=True)[:K]:
        for j in train_data.get(v, ()):
            if j in interacted:
                continue
            recs[j] += sim

    return dict(sorted(recs.items(), key=itemgetter(1), reverse=True)[:N])

user = 69
print(recommend(user, N=10))

      
      
      
      