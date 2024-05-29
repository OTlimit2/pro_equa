# 开发日期：2023/11/1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取整个CSV文件
df = pd.read_csv('cnndata_adj_0610.csv')

var_name = ('fvc', 'facden')


# 选择要读取的列
def bar_drawing(var_list=var_name, bin_count=10, show=True, save=True):
    """
    该函数实现了将对应列数据绘制成柱状图，显示出各个分组中样本数量所占百分比
    :param var_list: 传入需要绘制图像的变量名称，可以有多个，以元组的形式传入
    :param bin_count: 设置分组数量，默认为 10
    :param show: 设置是否显示图像，默认为显示
    :param save: 设置是否保存图像，默认为保存
    :return: None
    """
    for i in range(len(var_list)):
        col = df[f'{var_list[i]}']  # 通过列名选择列

        hist, bin_edges = np.histogram(col, bins=bin_count)

        # 计算各组的百分比
        percentage = (hist / len(col)) * 100

        # 绘制柱状图
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
        plt.bar(range(bin_count), percentage, width=0.8, align='center')

        # 在每个柱形上添加具体值（占比）
        for j, p in enumerate(percentage):
            plt.text(j, p, f'{p:.2f}%', ha='center', va='bottom')

        # 设置X、Y轴标签
        x_ticks = []
        for m in range(bin_count):
            if m == bin_count - 1:
                x_ticks.append(f'  组 {m+1} \n {bin_edges[m]:.2f}--{bin_edges[m+1]:.2f}')
            elif m == bin_count - 2:
                x_ticks.append(f'组 {m + 1} \n {bin_edges[m]:.2f} -    ')
            else:
                x_ticks.append(f'组 {m+1} \n {bin_edges[m]:.2f} ---  ')
        plt.xticks(range(bin_count), x_ticks, fontsize=6)
        plt.xlabel(f'变量{var_list[i]}分组')
        plt.ylabel('样本占比', fontsize=12)

        # 设置标题
        plt.title(f'变量{var_list[i]}分组情况直方图')

        # 保存图片
        if save:
            plt.savefig(f'figures\\变量{var_list[i]}分组情况直方图.png', format='png')

        # 显示柱状图
        if show:
            plt.show()


def bar2d_drawing(col1, col2, bin_count=10, show=False, save=False):
    """
    该函数实现了将两列数据绘制成二维柱状图，显示出各种不同分组情况下样本数量所占比例，即这两个变量的联合分布情况
    :param col1: 传入需要进行图像绘制的第一个维度变量
    :param col2: 传入需要进行图像绘制的第二个维度变量
    :param bin_count: 设置分组数量，默认为 10
    :param show: 设置是否显示图像，默认为显示
    :param save: 设置是否保存图像，默认为保存
    :return: 返回一个二维数组，该数组中每个元素表示出现该情况的样本数量
    """
    min_col1, max_col1 = col1.min(), col1.max()
    min_col2, max_col2 = col2.min(), col2.max()
    # 创建一个二维直方图，表示不同分组情况下的样本数量所占比例
    hist, x_edges, y_edges = np.histogram2d(col1, col2, bins=(bin_count, bin_count),
                                            range=[[min_col1, max_col1], [min_col2, max_col2]])

    # 计算比例
    percentage = (hist / hist.sum()) * 100
  
    # 创建图形
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
    plt.imshow(percentage, cmap='inferno', origin='lower', extent=(0, bin_count, 0, bin_count), interpolation='nearest')
    plt.colorbar(label='百分比')

    # 在每个方格中显示具体占比数值
    for i in range(bin_count):
        for j in range(bin_count):
            if percentage[i][j] >= np.max(percentage) - 5:      # 颜色过亮的时候将字体调为黑色
                plt.text(j + 0.5, i + 0.5, f'{percentage[i, j]:.3f}%', ha='center', va='center', color='black',
                         fontsize=7)
            else:
                plt.text(j + 0.5, i + 0.5, f'{percentage[i, j]:.3f}%', ha='center', va='center', color='white',
                         fontsize=7)

    # 设置轴标签
    x_labels = [f'组{n+1}' for n in range(10)]
    y_labels = [f'组{n+1}' for n in range(10)]
    plt.xticks(np.arange(bin_count) + 0.5, x_labels)
    plt.yticks(np.arange(bin_count) + 0.5, y_labels)
    plt.xlabel(f'变量col1分组')
    plt.ylabel(f'变量col2分组')

    plt.title(f'变量col1和col2分组二维直方图')

    # 保存图片
    if save:
        plt.savefig(f'figures\\变量col1和col2分组二维直方图.png', format='png')

    # 显示图形
    if show:
        plt.show()

    return hist


if __name__ == '__main__':
    # bar_drawing()
    # bar2d_drawing()
    pass


