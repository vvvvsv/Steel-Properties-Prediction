# 钢材性能预测项目

王思远 黄高翔 2022.06.09

题目详见`prob.zip`，报告详见`Steel-Report.pptx`

## 题目

基于机器学习算法，依据钢材的生产参数，预测成品钢材的性能（实验屈服值、实验抗拉值、实验伸长率）。其中，生产参数包括：

1. 基本信息：材料（含生产月份信息）、试批号、牌号、钢种

2. 工艺参数：出炉温度、加热时间、板坯厚度、中间坯厚度、粗轧压下率、精轧开轧温度、终轧温度、成品厚度、粗轧压缩比、精轧压缩比

3. 元素成分：C、Si、Mn、AlT、Nb、V、Ti、Ni、Cu、Cr、Mo、P、S

## 实验结果

|      | 实验屈服值  |        |            |        | 实验抗拉值  |        |            |        | 实验伸长率 |      |            |        |
| ---- | ------ | ------ | ---------- | ------ | ------ | ------ | ---------- | ------ | ----- | ---- | ---------- | ------ |
| 模型   | GBDT   | SVR    | Simple MLP | 多任务    | GBDT   | SVR    | Simple MLP | 多任务    | GBDT  | SVR  | Simple MLP | 多任务    |
| MSE  | 354.77 | 507.01 | 375.11     | 398.71 | 186.35 | 226.64 | 179.34     | 203.29 | 13.33 | 15.1 | 14.59      | 16.04  |
| RMSE | 18.84  | 22.52  | 19.37      | 19.97  | 13.65  | 15.05  | 13.39      | 14.26  | 3.65  | 3.89 | 3.82       | 4      |
| MAE  | 14.64  | 16.89  | 14.63      | 15.45  | 10.69  | 11.44  | 9.98       | 10.94  | 2.96  | 3.11 | 3.05       | 3.22   |
| R^2  | 0.49   | 0.27   | 0.46       | 0.43   | 0.37   | 0.23   | 0.39       | 0.3113 | 0.29  | 0.2  | 0.22       | 0.1475 |