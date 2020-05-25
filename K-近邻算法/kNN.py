import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import os


def creat_data_set():
    """
    构建数据集 与 标签
    :return:
    """
    data_set = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return data_set, labels


def classify0(inx, data_set, labels, k):
    """
    通过对未知向量 与已经判别lable的向量 计算欧式距离 来判断 未知向量的label
    :param inx: 未知向量
    :param data_set: 已经判别lable的向量列表
    :param labels:  向量列表对应labels 列表
    :param k:   挑选距离最近的前k个 向量labels
    :return:  未知向量所属label
    """
    data_set_size = data_set.shape[0]  # 获取当前数据个数
    base_inx_set = np.tile(inx, (data_set_size, 1))  # 通过构造inx 重复序列 从而方便计算 平方差的和
    diff2_set = (data_set - base_inx_set) ** 2
    # print(diff2_set)
    distance = diff2_set.sum(axis=1) ** 0.5
    distance_argsort = distance.argsort()  # 对不同的label 进行count 并且以dict形式存储
    # print(distance_argsort)
    size_count = {}
    for i in range(k):
        vote_label = labels[distance_argsort[i]]
        size_count[vote_label] = size_count.get(vote_label, 0) + 1

    # 对dict中的item 进行排序 排序方式为value 值的从大到小
    labels_sort_reverse = sorted(size_count.items(), reverse=True, key=lambda item: item[1])
    return labels_sort_reverse[0][0]


# 由于上面给出了knn算法基本内容，以下需要构造输入变量data_set, labels
def file2matrix(filename):
    """
    将文件中的数据转化为 矩阵 和labels列表
    :param filename: 文件名称 要求文件必须为数据文件并且不含有表头
    :return:  数据矩阵， labels列表
    """
    print('请确保输入文件为数据文件并且不含有表头')
    fr = open(filename, 'r')
    file_lines_list = fr.readlines()
    n_row = len(file_lines_list)
    n_col = len(file_lines_list[0].strip().split('\t'))
    # 通过n_row, n_col 构造 占位符形成的矩阵
    base_matrix = np.zeros((n_row, n_col - 1))

    labels_list = []
    index = 0
    for l in file_lines_list:
        ele_list = l.strip().split('\t')
        base_matrix[index, :] = ele_list[0:n_col - 1]
        labels_list.append(int(ele_list[-1]))
        index += 1
    return base_matrix, labels_list


# 特征中心化(归一化)处理
def auto_norm(data_set):
    """
    将data_set 进行
    :param data_set: 数据类型为 narray
    :return: norm_data_set 中心化数据，
    """
    min_vals = data_set.min(axis=0)
    max_vals = data_set.max(axis=0)
    ranges = max_vals - min_vals
    nrow, ncol = data_set.shape
    norm_data_set = (data_set - np.tile(min_vals, (nrow, 1))) / (np.tile(ranges, (nrow, 1)))
    return norm_data_set, min_vals, ranges


# 通过随机抽取测试集合计算 预测准确率
def random_data_test(random_ratio, data_set, labels):
    """
    通过随机抽取测试集合计算 预测准确率
    :param random_ratio: 默认随机抽取10%的数据
    :param data_set: 数据自变量集合
    :param labels: 数据因变量集合
    :return: error_ratio 错误率
    """
    nrow, ncol = data_set.shape
    # 测试数据行数
    n_test = int(nrow * random_ratio)
    # 数据中心化
    norm_data_set, min_vals, ranges = auto_norm(data_set)
    # 获取测试集 和 训练集 的index定位
    test_index_list = random.sample(range(nrow), n_test)
    training_index_list = [i for i in range(nrow) if i in test_index_list]
    # 定义训练集合 和
    training_data_set = norm_data_set[training_index_list]
    training_labels = np.array(labels)[training_index_list,]

    error_count = 0
    # print('选取测试行：', test_index_list)
    for i in test_index_list:
        predict_labels = classify0(inx=norm_data_set[i],
                                   data_set=training_data_set,
                                   labels=training_labels,
                                   k=3)
        if predict_labels != labels[i]:
            error_count += 1
            print('行数', i, '数据内容', data_set[i], '\t', '实际数值', labels[i], '=>', predict_labels)
    error_ratio = error_count / n_test
    return '%.8f' % error_ratio


if __name__ == '__main__':
    ##############################################
    # datingTestSet2 数据的初步展示
    # inx = [0, 0]
    # data_set, labels = creat_data_set()
    # k = 3
    # res = classify0(inx, data_set, labels, k)
    # print(res)

    ###############################################
    # datingTestSet2 2，3列的散点图绘制
    data_set, labels = file2matrix('datingTestSet2.txt')
    # print(data_set)
    # print(labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 111 为 1x1 窗格下的第一个图表
    ax.scatter(data_set[:, 1], data_set[:, 2], 15.0 * np.array(labels), 15.0 * np.array(labels))  # scatter 散点图
    plt.show()

    ###############################################
    # datingTestSet2 1，2列的三点图形绘制
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.scatter(data_set[:, 0], data_set[:, 1], 15 * np.array(labels), 15 * np.array(labels))
    plt.show()

    ###############################################
    # 对datingTestSet 进行绘制
    data_set, labels = file2matrix('datingTestSet.txt')
    fig3 = plt.figure()
    ax1 = fig3.add_subplot(221)
    ax1.scatter(data_set[:, 0], data_set[:, 1], 15.0 * np.array(labels), 15.0 * np.array(labels))
    ax2 = fig3.add_subplot(222)
    ax2.scatter(data_set[:, 1], data_set[:, 2], 15.0 * np.array(labels), 15.0 * np.array(labels))
    ax3 = fig3.add_subplot(223)
    ax3.scatter(data_set[:, 0], data_set[:, 2], 15.0 * np.array(labels), 15.0 * np.array(labels))
    plt.show()

    ###############################################
    # 通过Knn 对datingTestSet 进行预测 并且计算重复率
    data_set, labels = file2matrix('datingTestSet.txt')
    error_ratio = random_data_test(0.1, data_set, labels)
    print(error_ratio)

    ###############################################
    # 对于手写数据集 进行训练和计算
    # 定义空的训练数据 和 测试数据
    training_labels = []
    training_data_set = []
    test_labels = []
    test_data_set = []
    # 遍历文件夹中的文件 将文件名内容 和 文件内容 添加到 对应list 中
    for filename in os.listdir('./digits/trainingDigits'):
        n = int(filename.split('_')[0])
        training_labels.append(n)
        with open('./digits/trainingDigits/' + filename, 'r') as f:
            data_set_line = [int(i) for i in f.read().replace('\n', '').strip()]
            training_data_set.append(data_set_line)
            f.close()

    for filename in os.listdir('./digits/testDigits'):
        n = int(filename.split('_')[0])
        test_labels.append(n)
        with open('./digits/testDigits/' + filename, 'r') as f:
            data_set_line = [int(i) for i in f.read().replace('\n', '').strip()]
            test_data_set.append(data_set_line)
            f.close()
    training_labels = np.array(training_labels)
    training_data_set = np.array(training_data_set)
    test_labels = np.array(test_labels)
    test_data_set = np.array(test_data_set)

    # 获取测试数据集合长度
    nrow_test = len(test_labels)
    error_count = 0
    for i in range(nrow_test):
        predict_label = classify0(inx=test_data_set[i],
                                  data_set=training_data_set,
                                  labels=training_labels,
                                  k=3)

        if predict_label != test_labels[i]:
            error_count += 1
            print('{}=>{}'.format(test_labels[i], predict_label))
    error_ratio = error_count / nrow_test
    print('%.8f' % error_ratio)
