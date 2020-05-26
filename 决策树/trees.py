import numpy as np
from math import log
from collections import Counter


def labels_index(data_set, ncol):
    """
    计算ncol 的标签 与 该标签 所在行数列表
    :param data_set: 数据集
    :param ncol: 指定列数 计算该列的标签
    :return: labels_index
    """
    labels_index_dict = {}
    nrow = data_set.shape[0]
    for m in range(nrow):
        value = data_set[m, ncol]
        if value not in labels_index_dict.keys():
            labels_index_dict[value] = []
        labels_index_dict[value].append(m)
    return labels_index_dict


# 计算香农熵
def canculate_shannon_ent(data_set):
    """
    参考链接 https://blog.csdn.net/theonegis/article/details/79890407
    计算香农 熵
    :param data_set: 数据集 array 类型 最后一列使用标签数据（分类变量）
    :return: 熵值 和 标签计数
    """
    nrow = data_set.shape[0]
    # 计算 香农 熵 数值
    shannon_ent = 0.0
    labels_count = Counter(data_set[:, -1]).values()
    for v in labels_count:
        p = v / nrow
        shannon_ent -= p * log(p, 2)

    return shannon_ent


# 根据 某一列的数值是否为value 来划分数据集
def split_data_col_by_value(data_set, ncol, value):
    split_data_set = []
    for row in data_set:
        if row[ncol] == str(value):
            list(row).pop(ncol)
            split_data_set.append(row)
    return split_data_set


# 通过计算熵值 选择出 最好的分类方式
def choose_best_feature_split(data_set):
    """
    通过对不同列的判断 判断那一列的那个标签 对应的数据集的熵最大， 从而确定 从那一列开始分类
    :param data_set:  自变量和因变量都有的数据集
    :return: 选择的列数 以及该列对应的 标签
    """
    nrow, ncol = data_set.shape

    # 计算原数据集因变量 香农熵
    base_entropy = canculate_shannon_ent(data_set)
    best_info_gains = 0
    best_col = -1
    print('best_info_gains =>', best_info_gains)
    # 遍历每一列从而获取 不同列的标签
    for n in range(ncol - 1):  # 不需要包含因变量列
        col_entropy = 0
        col_labels_index = labels_index(data_set, n)
        for l in col_labels_index.values():
            labels_data_set = data_set[l,]
            # print(labels_data_set)
            labels_entropy = canculate_shannon_ent(labels_data_set)
            col_entropy += len(l) / float(nrow) * labels_entropy
        col_info_gains = base_entropy - col_entropy
        print(str(n) + ' col info_gains =>', col_info_gains)
        if col_info_gains >= best_info_gains:
            best_info_gains = col_info_gains
            best_labels_index = col_labels_index
            best_col = n
    return best_col, best_labels_index


# 创建决策树

def creat_tree(data_set, col_name):
    """
    创建决策树 ，并且以dict方式存储
    :param data_set: narray 变量数据集合
    :param col_name: list or array 列名称
    :return: dict tree
    """

    print('data_set\n', data_set)
    # 该列labels 全部相同，则停止划分
    if len(set(data_set[:, -1])) == 1:
        return data_set[0, -1]
    # 自变量数据集合 被全部划分，则停止划分
    if data_set.shape[1] == 1:
        labels_index_dict = labels_index(data_set, ncol=1)
        values_list = [len(k) for k in labels_index_dict.values()]
        keys_list = labels_index_dict.keys()
        return dict(zip(keys_list, values_list))
    # 选择最好的分类方式
    n, col_labels_index = choose_best_feature_split(data_set)
    print(n)
    # 选择 该行数的col_name
    best_choose_col_name = col_name[n]
    tree = {best_choose_col_name: {}}

    for k, v in col_labels_index.items():
        tree[best_choose_col_name][k] = creat_tree(np.delete(data_set, n, axis=1)[v, ], np.delete(col_name, n, axis=0))

    return tree


if __name__ == '__main__':
    # #################################
    # # 测试计算指定列的熵值
    # data_set = np.array([[1, 1, 'yes'],
    #                      [1, 1, 'yes'],
    #                      [1, 0, 'no'],
    #                      [0, 1, 'no'],
    #                      [0, 1, 'no']])
    # ent = canculate_shannon_ent(data_set)
    # print(ent)
    #
    # data_set[0][-1] = 'maybe'
    # ent = canculate_shannon_ent(data_set)
    # print(ent)
    # ##################################
    # data_set = np.array([[1, 1, 'yes'],
    #                      [1, 1, 'yes'],
    #                      [1, 0, 'no'],
    #                      [0, 1, 'no'],
    #                      [0, 1, 'no']])
    # split_set_data = split_data_col_by_value(data_set, 0, 1)
    # print(split_set_data)
    #
    # ####################################
    # data_set = np.array([[1, 1, 'yes'],
    #                      [1, 1, 'yes'],
    #                      [1, 0, 'no'],
    #                      [0, 1, 'no'],
    #                      [0, 1, 'no']])
    # print(choose_best_feature_split(data_set))

    #######################################
    data_set = np.array([[1, 1, 'yes'],
                         [1, 1, 'yes'],
                         [1, 0, 'no'],
                         [0, 1, 'no'],
                         [0, 1, 'no']])
    col_name = ['no surfacing', 'flippers']
    res = creat_tree(data_set, col_name)
    print(res)
