import math
from collections import defaultdict
from operator import itemgetter
import pandas as pd

def load_data(path="ml-100k/u.data"):
    df = pd.read_csv(path, sep="\t", names=['user_id', 'item_id', 'rating', 'timestamp'])
    data = {}
    for row in df.itertuples(index=False):
        if row.rating >= 4:
            data.setdefault(row.user_id, set()).add(row.item_id)
    return data
train_data = load_data()

item_users = {}
for u, items in train_data.items():
    for it in items:
        item_users.setdefault(it, set()).add(u)

c = {}
for it, users in item_users.items():
    users = list(users)
    for i in range(len(users)):
        u = users[i]
        c.setdefault(u, defaultdict(int))
        for j in range(i, len(users)):
            v = users[j]
            c[u][v] += 1
            if u != v:
                c.setdefault(v, defaultdict(int))
                c[v][u] += 1
SwingSim = {}
alpha = 1.0
for i, Ui in item_users.items():
    SwingSim.setdefault(i, defaultdict(float))
    for j, Uj in item_users.items():
        if i == j:
            continue
        score = 0.0
        for u in Ui:
            cu = c.get(u)
            if not cu:
                continue
            for v in Uj:
                cuv = cu.get(v, 0)
                if cuv > 0:
                    score += 1.0 / (alpha + cuv)
        if score > 0.0:
            SwingSim[i][j] = score

def recommend(user, N=5, K=10):
    if user not in train_data:
        return {}
    interacted = train_data[user]
    recs = defaultdict(float)

    for i in interacted:
        if i not in SwingSim:
            continue
        # 与i最相似的前 K 个物品
        for j, sim in sorted(SwingSim[i].items(), key=itemgetter(1), reverse=True)[:K]:
            if j in interacted:
                continue
            recs[j] += sim

    return dict(sorted(recs.items(), key=itemgetter(1), reverse=True)[:N])
user = 69
print(recommend(69, N=10))