# cs-phd-from-zero-to-one

## 矩阵补充的原理
根据已观测到的，有确实的矩阵，通过已知的部分数据，补全（预测）未知的矩阵元素。大前提是共同喜欢的item是相关的，根据已知的隐含因子（如评分或者交互行为）反推出所有的user—item的潜在关系，最终输出完整的矩阵，用于推荐或者预测。
## 为什么需要将ID转成连续整数索引再做embedding？
Id转为Embedding的通用操作是ID → 连续整数编码 → 送入 nn.Embedding 这是一种 通用、标准 的操作流程，几乎所有用到 embedding 的模型都会这么做，原因是原始ID一般是离散符号，而embedding输入要求是[0，n-1]的整数索引，适用于中小数据集的操作是Pandas category + .cat.codes，反映在其他任务如NLP中，则是将词典映射成整数，CTR中，将不同的id如用户id，广告id，设备id等各自独立映射，将离散特征转成向量计算。
## 正负样本采样思路
曝光且有点击的用户-物品成为正样本，但存在少部分物品占据大部分点击，导致正样本大多是热门物品，解决方法是过采样负样本，或者降采样热门物品。然而这个思路不适合电影推荐。我认为不能将用户打分过的所有电影都认为是正样本，只有评分高的算正样本，负样本需要随机采样。
## Deep Retrieval vs 双塔模型
经典的双塔模型将用户和物品表示成向量，分为用户塔和向量塔，即使加入了很多特征，最终也被塔网络混合成一个：
用户塔： u_vec = f(u)
物品塔： i_vec = g(i)
打分：   score = dot(u_vec, i_vec)
而Deep Retrieval将每个特征分别成一个token embedding，不拼接，最后每个user token与每个item token两两匹配:
用户编码： U = [u_token1, u_token2, ...]
物品编码： I = [i_token1, i_token2, ...]
打分：    score = Σ_q( max_d( u_token_q · i_token_d ) )
区别：双塔模型整合了用户和物品特征，但最后是单向量匹配；Deep Retrieval是多token晚交互，不把特征混合在编码阶段，而是保留到匹配阶段再计算细粒度相似度。
## AUC
<img width="1180" height="816" alt="image" src="https://github.com/user-attachments/assets/93289597-72b1-4d16-93a6-6d02947a3d2c" />

