#Item CF的基本原理是：喜欢同一个物品的用户，会让这些物品更相似，如果给用户做推荐，从ta喜欢的物品出发，推荐与这些物品相似的物品
from collections import defaultdict
import math
import pandas as pd
from operator import itemgetter
def load_data(path="ml-100k/u.data"):
    df = pd.read_csv(path, sep="\t", names=['user_id', 'item_id', 'rating', 'timestamp'])
    data = {}
    for row in df.itertuples(index=False):
        if row.rating >= 3:
            data.setdefault(row.user_id, set()).add(row.item_id)
    return data
train_data = load_data()
#以item为主键构建用户表，实现item到user的数据集
item_users = {}
for u, items in train_data.items():
    for i in items:
        item_users.setdefault(i,set()).add(u)
#这一步变成{10：{1，3}}

co_matrix = {}#定义一个矩阵，这个矩阵存储每个item被多少个用户喜欢
item_degree = defaultdict(int)#
for i, users in item_users.items():
    item_degree[i] = len(users)#喜欢物品i的用户个数
    for u in users:
        items_u = train_data[u]#用户u喜欢的所有物品合集
        weight = 1.0 / math.log(1+len(items_u))#降低重度用户贡献太多的虚假相似度
        for j in items_u:
            if j == i:
                continue
            co_matrix.setdefault(i, defaultdict(float))#为什么这里需要用float，因为iuf模式可能出现浮点数
            co_matrix[i][j] += weight
itemSim = {} #定义item的余弦相似度
for i, related in co_matrix.items():#这里的related是和物品i共现的其他物品
    itemSim.setdefault(i, {})
    for j, cij in related.items():
        denom = math.sqrt(item_degree[i] * item_degree[j])
        itemSim[i][j] = 0.0 if denom == 0 else cij / denom
#给用户做推荐

def recommend(user, N=5, K=10):
    if user not in train_data:
        return{}#如果用户不在训练的列表里，返回空值
    interacted = train_data[user]
    recs = defaultdict(float)

    for i in interacted:
        if i not in itemSim:
            continue#跳过冷门产品，如果物品没有共有喜欢用户，跳过，但是这里不会造成损失吗？？
        for j, sim in sorted(itemSim[i].items(), key=itemgetter(1), reverse = True)[:K]:
            if j in interacted:
                continue#如果被用户喜欢过，也跳过，不做自我推荐
            recs[j] +=sim
    return dict(sorted(recs.items(), key=itemgetter(1), reverse=True)[:N])
user = 69
print(recommend(69, N=10))
