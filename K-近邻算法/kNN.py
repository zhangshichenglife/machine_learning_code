import numpy as np


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
    data_set_size = data_set.shape[0]   # 获取当前数据个数
    base_inx_set = np.tile(inx, (data_set_size, 1))    # 通过构造inx 重复序列 从而方便计算 平方差的和
    diff2_set = (data_set - base_inx_set) ** 2
    # print(diff2_set)
    distance = diff2_set.sum(axis=1) ** 0.5
    distance_argsort = distance.argsort()              # 对不同的label 进行count 并且以dict形式存储
    # print(distance_argsort)
    size_count = {}
    for i in range(k):
        vote_label = labels[distance_argsort[i]]
        size_count[vote_label] = size_count.get(vote_label, 0) + 1

    # 对dict中的item 进行排序 排序方式为value 值的从大到小
    labels_sort_reverse = sorted(size_count.items(), reverse=True, key=lambda item: item[1])
    return labels_sort_reverse[0][0]


if __name__ == '__main__':
    inx = [0, 0]
    data_set, labels = creat_data_set()
    k = 3
    res = classify0(inx, data_set, labels, k)
    print(res)

