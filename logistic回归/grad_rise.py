import numpy as np
import matplotlib.pyplot as plt


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
    data_set = []   # 自变量数据集
    label_set = []  # 因变量数据集
    for l in open('testSet.txt', 'r').readlines():
        l_split_list = l.strip().split('\t')
        label_set.append(float(l_split_list[-1]))
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
    step_length = 0.001      # 初始化步长
    step_num = 500           # 初始化步数
    weights = np.ones((3, 1))  # 初始化权重
    weights_list = []          # weights_list 用于收集历史权重， 展示分类效果
    for i in range(step_num):
        # 计算与label_set 实际数值的差值
        error = label_mat - sigmoid(data_mat * weights)
        # 以error 方向对权重进行优化
        weights = weights + step_length * data_mat.transpose() * error
        weights_list.append(weights)
    return weights_list


# 作图展示历史分类效果
def plot_best_fit(data_set, label_set, weights_list):
    pass



if __name__ == '__main__':
    data_set, label_set = load_data_set()
    # print(data_set, label_set)
    weights_list = grad_rise_method(data_set, label_set)
    print(weights_list[-1])
