# coding:utf-8

# 1.合并数据
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,ElasticNet,BayesianRidge

# processed = pd.read_csv('./01-processed_data.csv')
processed = pd.read_csv('./04-normalize.csv')
score = pd.read_csv('./02-id_score.csv')
processed.head()

merged = pd.merge(processed,
                  score,
                  on='id',
                  how='left')

merged.to_csv('./03-merged.csv')

# merged = pd.read_csv('./04-normalize.csv')

# 获取所有包含缺失值的行和列
nan_id = merged[merged.isna().values == True]['id'].unique()
nan_row = merged.loc[merged['id'].isin(nan_id)]
nan_row.count()


# 丢掉为空值的部分
train_test = merged.dropna()


# 2.数据分割

columns = train_test.columns.values.tolist()

target_columns = 'score'
for x in ['id','score']:
    columns.remove(x)

input_columns = columns


X_tain,X_test,Y_train,Y_test = train_test_split(
    train_test[input_columns],
    train_test[target_columns],
    test_size=0.2
)
# 3.选择模型
# 线性回归模型
# model = LinearRegression()
# model = ElasticNet()
model = BayesianRidge()
model.fit(X_tain,Y_train)
y_pred = model.predict(X_test)


# 4.测试评价
print("Coefficiengts: \n",model.coef_)
from sklearn.metrics import mean_squared_error,r2_score
print("mean squeared error: %2.4f" % mean_squared_error(Y_test,y_pred))
print("r2 error: %2.4f" % r2_score(Y_test,y_pred))

# 5.画图解释