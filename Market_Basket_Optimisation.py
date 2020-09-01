# 分析购物篮
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
        
# 数据加载,无头
baskets = pd.read_csv('./Market_Basket_Optimisation.csv', header = None)

pd.options.display.max_columns=40
# 进行行转列,并去掉索引，目的是为了设置头
stacks = baskets.stack().reset_index()
print(stacks)
# 设置头
stacks.columns = ['tId','tmpId', 'foodName']
#print(stacks.groupby(['tId','foodName'])['foodName'].count().unstack().reset_index().fillna(0).set_index('tId'))

# count()是为了计算一个transaction里有可能有重复的食物
# unstack，再进行列转行
# set_index，再一次设置索引
hot_encoded_df = stacks.groupby(['tId','foodName'])['foodName'].count().unstack().reset_index().fillna(0).set_index('tId')
# one_hot编码
hot_encoded_df = hot_encoded_df.applymap(encode_units)
# 计算频繁相集
frequent_itemsets = apriori(hot_encoded_df, min_support=0.02, use_colnames=True)
# 计算关联规则
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.2)
print("频繁项集：", frequent_itemsets)
print("关联规则：", rules[ (rules['lift'] >= 1) & (rules['confidence'] >= 0.2) ])
#print("关联规则2：", rules[ (rules['lift'] >= 1)])
#print("关联规则2：", rules)
