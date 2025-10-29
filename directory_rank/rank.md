## 关注业务知识
点击率 = 点击次数 / 曝光次数
点赞率 = 点赞次数 / 点击次数
收藏率 = 收藏次数 / 点击次数
转发率 = 转发次数 / 点击次数
排序模型预估点击率、点赞率、收藏率等多种分数，融合这些分数（根据业务规则），根据融合的分数做排序和截断。
## 多目标模型
小红书
用户特征：id,画像
物品特征：id，画像，作者信息, GeoHash,城市，笔记自带属性（标题、类目等）
统计特征：包括用户统计和物品统计信息，exp过去三十天曝光机会，多少点赞
<img width="1022" height="560" alt="image" src="https://github.com/user-attachments/assets/fc1033c6-1dea-4570-a0d4-2c95b4300f1f" />
<img width="1054" height="642" alt="image" src="https://github.com/user-attachments/assets/c635bf87-ea6d-4caf-88a2-89daa4effa6b" />
场景特征：当前时间，地理，节假日信息等
## 特征处理
离散特征：直接做embedding
包括用户id，物品id，作者id；
类别：关键词、城市、手机品牌；
连续特征：做分桶，变成离散特征
其他连续特征变化：做log（1+x）、转为为点击率、点赞率，并且做平滑。
还需要特征覆盖率
数据服务业务流：
<img width="1760" height="1010" alt="image" src="https://github.com/user-attachments/assets/06b55fc1-a45b-4a44-afd6-0238e87d728d" />

## 粗排和精排
需要考虑两个模型的单次推理代价，粗排需要单次推理代价小，预估准确性可以不高，精排则相反；
前期融合模型用于召回；后期融合模型用于排序；
粗排的三塔模型：
<img width="1294" height="728" alt="image" src="https://github.com/user-attachments/assets/74fba323-9162-4f75-868a-2000d222d126" />
介于前期融合与后期融合之间， 交叉塔中，每次推荐，用户塔只有一个用户，只做一次推理，物品塔较大，但是近乎静态，未命中缓存时需要做推理，交叉塔包含统计数据，该数据需要更新，必须做n次推理。



