import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import clustering as clt
from node import *
import regression as rg

df = pd.read_csv('cnndata_adj_0610.csv')
# var_list = ['cd', 'as', 'cr', 'hg', 'pb', 'qtot', 'riverden', 'gdp', 'iwu', 'pop']
# factor_list = ['cd_group', 'as_group', 'cr_group', 'hg_group', 'pb_group', 'qtot_group', 'riverden_group',
#                'gdp_group', 'iwu_group', 'pop_group']
# soilhm_list = [0, 1, 2, 3, 4]
# water_list = [5, 6]
# farmer_list = [7, 8, 9]

# hidden_node_list = [soilhm_node, water_node, air_node, farmer_node]
hidden_node_list = [air_node]

# 求出显性变量名列表
var_list, factor_list = [], []

for node in hidden_node_list:
    var_list.extend(node.son_list)
    factor_list.extend(node.group)


# 依次筛选各个变量中的异常值，得到筛选后的df
for var in var_list:
    df = clt.data_except(df, var)

X = df[var_list].values
y = df['cd'].values
linear_rmse, linear_mae, linear_r2, b, k = rg.linear_regression(X, y)
ridge_rmse, ridge_mae, ridge_r2 = rg.ridge_regression(X, y)
print(f'线性回归rmse：{linear_rmse}')
print(f'线性回归mae：{linear_mae}')
print(f'线性回归r2：{linear_r2}')
print(f'岭回归rmse：{ridge_rmse}')
print(f'岭回归mae：{ridge_mae}')
print(f'岭回归r2：{ridge_r2}')

# 对变量离散化
# for var in var_list:
#     clt.data_group_cut(df, var)
#
# # 划分训练集和测试集
# X = df[factor_list].values
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
# print(len(X_train))
# print(len(X_test))
# X_soilhm_train = X_train[:, range(5)]
# X_soilhm_test = X_test[:, range(5)]
# X_water_train = X_train[:, range(5, 7)]
# X_water_test = X_test[:, range(5, 7)]
# X_air_train = X_train[:, range(7, 12)]
# X_air_test = X_test[:, range(7, 12)]
# X_farmer_train = X_train[:, range(12, 15)]
# X_farmer_test = X_test[:, range(12, 15)]
#
# cd_test = np.array(X_soilhm_test[:, 0])
#
# soilhm, soilhm_centroids, soilhm_delta_list, soilhm_error = clt.k_means_internal(X_soilhm_train, X_soilhm_test)
#
# water_delta_list, water_error = \
#     clt.k_means_transitive([X_water_train], [X_water_test], soilhm, soilhm_centroids, cd_test)
# air_delta_list, air_error = \
#     clt.k_means_transitive([X_air_train], [X_air_test], soilhm, soilhm_centroids, cd_test)
# farmer_delta_list, farmer_error = \
#     clt.k_means_transitive([X_farmer_train], [X_farmer_test], soilhm, soilhm_centroids, cd_test)
#
# # water_farmer_delta_list, water_farmer_error = \
# #     clt.k_means_transitive([X_water_train, X_farmer_train], [X_water_test, X_farmer_test], soilhm, soilhm_centroids, cd_test, coef=[0.5, 0.5])
# triple_delta_list, triple_error = \
#     clt.k_means_transitive([X_water_train, X_air_train, X_farmer_train], [X_water_test, X_air_test, X_farmer_test], soilhm, soilhm_centroids, cd_test, coef=[1 / 3, 1 / 3, 1 / 3])
#
#
# print(f'内部预测mae:{soilhm_error[0]}')
# print(f'内部预测rmse:{soilhm_error[1]}')
# print(f'内部预测r2:{soilhm_error[2]}')
# print(f'water预测mae:{water_error[0]}')
# print(f'water预测rmse:{water_error[1]}')
# print(f'water预测r2:{water_error[2]}')
# print(f'air预测mae:{air_error[0]}')
# print(f'air预测rmse:{air_error[1]}')
# print(f'air预测r2:{air_error[2]}')
# print(f'farmer预测mae:{farmer_error[0]}')
# print(f'farmer预测rmse:{farmer_error[1]}')
# print(f'farmer预测r2:{farmer_error[2]}')
# print(f'三者共同预测mae:{triple_error[0]}')
# print(f'三者共同预测rmse:{triple_error[1]}')
# print(f'三者共同预测r2:{triple_error[2]}')

