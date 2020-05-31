import numpy as np
import matplotlib.pyplot as plt

INT_TO_COLOR = {0: 'red', 1: 'yellow', 2: 'blue', 3: 'green', 4: 'orange'}


# 定义sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 绘制sigmoid 函数图像
x = np.linspace(-5, 5, 500)
y = sigmoid(x)
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.show()


# 从testSet.txt中加载数据
def load_data_set():
    data_set = []  # 自变量数据集
    label_set = []  # 因变量数据集
    for l in open('testSet.txt', 'r').readlines():
        l_split_list = l.strip().split('\t')
        label_set.append(int(l_split_list[-1]))
        x0 = [1]
        x = [float(n) for n in l_split_list[:-1]]
        x0.extend(x)
        data_set.append(x0)
    return data_set, label_set


# 定义梯度上升路径
def grad_rise_method(data_set, label_set):
    data_mat = np.mat(data_set)
    label_mat = np.mat(label_set).transpose()
    m, n = data_mat.shape
    step_length = 0.001  # 初始化步长
    step_num = 500  # 初始化步数
    weights = np.ones((3, 1))  # 初始化权重
    weights_list = []  # weights_list 用于收集历史权重， 展示分类效果
    for i in range(step_num):
        # 计算与label_set 实际数值的差值
        error = label_mat - sigmoid(data_mat * weights)
        # 以error 方向对权重进行优化
        weights = weights + step_length * data_mat.transpose() * error
        weights_list.append(weights)
    return weights_list


# 作图展示历史分类效果
def plot_best_fit(data_set, label_set, weights_list=[1, 1, 1]):
    """
    对于每个weights 进行作 直线， 查看分类效果
    :param data_set: narray 自变量数据集
    :param label_set: list or array 因变量数据类性
    :param weights_list: 历史权重列表 列表中的weights 为 3x1矩阵
    :return: 输出图形
    """
    for weights in weights_list:
        print(weights)
        # 将label_set 进行分类 分类为 {'key1': [index1, index2], 'key2': ...}
        label_index_dict = {}
        nrow = len(label_set)
        for m in range(nrow):
            value = label_set[m]
            if value not in label_index_dict.keys():
                label_index_dict[value] = []
            label_index_dict[value].append(m)
        # 初始化画板
        fig = plt.figure()
        # 初始化画板 图片
        ax = fig.add_subplot(111)
        # 循环添加不同颜色的散点图形
        for label, label_indexs in label_index_dict.items():
            # print(np.array(label_set)[label_indexs, ])
            ax.scatter(data_set[label_indexs, 1], data_set[label_indexs, 2], s=20*(label+1), color=INT_TO_COLOR[label])

        # 添加分类线部分
        x = np.arange(-3.0, 3.0, 0.1)
        y = (-weights[0, 0] - weights[1, 0] * x) / weights[2, 0]
        ax.plot(x, y, linewidth=4)
        plt.xlim(min(data_set[:, 1]), max(data_set[:, 1]))
        plt.ylim(min(data_set[:, 2]), max(data_set[:, 2]))

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()


if __name__ == '__main__':
    data_set, label_set = load_data_set()
    # print(data_set, label_set)
    weights_list = grad_rise_method(data_set, label_set)
    plot_best_fit(np.array(data_set), label_set, weights_list)