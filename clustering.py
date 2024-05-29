# 开发日期：2023/11/7
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import bar_graph_drawing as bgd


def data_except(sample, factor, show=False):
    """
    该函数采用绘制箱线图的方法对目标变量进行预处理，可以实现可视化，将异常的数据排除
    :param sample: 样本，以 DataFrame 的形式传入
    :param factor: 需要进行异常值处理的变量，一次只能传入一个变量
    :param show: 是否需要可视化数据分布，默认为不显示
    :return: 筛选异常值相应的列之后的样本
    """
    if show:
        # 绘制箱线图来可视化数据分布
        plt.boxplot(sample[factor])
        plt.show()

    # 使用箱线图来确定异常值的阈值，例如，这里选择超过箱线图上边缘的值为异常
    q1 = sample[factor].quantile(0.25)
    q3 = sample[factor].quantile(0.75)
    iqr = q3 - q1
    threshold1 = q1 - 1.5 * iqr
    threshold2 = q3 + 1.5 * iqr
    # 使用条件筛选来去除异常值
    sample_cleaned = sample[(threshold1 <= sample[factor]) & (sample[factor] <= threshold2)]

    return sample_cleaned


def data_group_cut(sample, factor, bins_num=10):
    """
    该函数实现了对变量进行预处理，使得各个变量的值离散化，离散后的结果存储在 DataFrame 的一个新列 factor_group 中
    :param sample: 样本，以 DataFrame 的形式传入
    :param factor: 需要进行离散化处理的变量，一次只能传入一个变量
    :param bins_num: 分组数量，默认为 10
    :return: None
    """
    sample[f'{factor}_group'] = pd.cut(sample[factor], bins=bins_num, labels=False)


def error_cal(abs_error, ssr, sst, num):
    """
    该函数实现了计算mae、rmse、r2的功能
    :param abs_error: 目标真实值减去预测值的绝对值求和
    :param ssr: 目标真实值减去预测值的平方求和
    :param sst: 目标真实值减去平均值的平方求和
    :param num: 目标样本的总数量
    :return: 依次返回mae、rmse、r2
    """
    mae = abs_error / num
    mse = ssr / num
    rmse = np.sqrt(mse)
    r2 = 1 - (ssr / sst)

    return mae, rmse, r2


def k_means_internal(x_train, x_test, k=10, target_index=0):

    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans.fit(x_train)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    delta_list = []
    mae_sum, mse_sum, sst = 0, 0, 0
    target_aver = np.sum(x_test[:, target_index]) / len(x_test)

    # 依次处理测试集中的每个样本
    for variable in x_test:
        # 将要排除的维度的值设置为0
        tmp = variable[target_index]
        variable[target_index] = 0

        # 计算 variable 与每个质心的欧氏距离，排除了指定维度
        distances = [np.linalg.norm(np.array(variable) - np.array(
            [centroid[i] if i != target_index else 0.0 for i in range(len(centroid))])) for centroid in
                     centroids]

        # 找到距离最近的质心
        nearest_centroid_index = np.argmin(distances)

        # 计算目标变量的预测值和真实值之间的差值
        target_predict = centroids[nearest_centroid_index][target_index]
        delta = tmp - target_predict                     # 差值定义为真实值减去预测值
        mae_sum += abs(delta)
        mse_sum += delta ** 2
        sst += (tmp - target_aver) ** 2
        delta_list.append(delta)

    error = error_cal(mae_sum, mse_sum, sst, len(x_test))

    return labels, centroids, delta_list, error


def k_means_transitive(x_train_list, x_test_list, next_labels, next_centroids, target, coef=(1,), k=10, target_index=0):

    if not (len(x_train_list) == len(x_test_list) == len(coef)):
        raise ValueError("输入的影响因素数量、权值数量不一致")

    if sum(coef) != 1:
        raise ValueError("各个权值之和不为1")

    # 获取kmeans模型列表和二维直方图列表
    model_list, bar2d_list = [], []
    for n in range(len(x_train_list)):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        kmeans.fit(x_train_list[n])
        model_list.append(kmeans)
        prior_labels = kmeans.labels_
        bar2d = bgd.bar2d_drawing(prior_labels, next_labels)
        bar2d_list.append(bar2d)

    delta_list = []
    mae_sum, mse_sum, sst = 0, 0, 0
    target_aver = np.sum(target) / len(target)

    j = 0

    # 依次处理测试集中的每个样本
    for sample in range(len(target)):
        target_predict = 0

        # 遍历每一种影响因素的测试样本
        for n in range(len(x_test_list)):
            variable = x_test_list[n][sample]
            variable = variable.reshape(1, -1)
            label = model_list[n].predict(variable)
            row_sum = np.sum(bar2d_list[n][label, :])

            # 计算目标变量的期望值
            for i in range(bar2d_list[n].shape[1]):
                target_predict += (next_centroids[i][target_index] * (bar2d_list[n][label, i] / row_sum)) * coef[n]

        delta = target[j] - target_predict                     # 差值定义为真实值减去预测值
        mae_sum += abs(delta)
        mse_sum += delta ** 2
        sst += (target[j] - target_aver) ** 2
        delta_list.append(delta)

        j += 1

    error = error_cal(mae_sum[0], mse_sum[0], sst, len(target))

    return delta_list, error


if __name__ == '__main__':
    pass

