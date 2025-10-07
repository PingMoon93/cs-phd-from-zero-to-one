# cs-phd-from-zero-to-one


# 为什么需要将ID转成连续整数索引再做embedding？
Id转为Embedding的通用操作是ID → 连续整数编码 → 送入 nn.Embedding 这是一种 通用、标准 的操作流程，几乎所有用到 embedding 的模型都会这么做，原因是原始ID一般是离散符号，而embedding输入要求是[0，n-1]的整数索引，适用于中小数据集的操作是Pandas category + .cat.codes，反映在其他任务如NLP中，则是将词典映射成整数，CTR中，将不同的id如用户id，广告id，设备id等各自独立映射，将离散特征转成向量计算。
