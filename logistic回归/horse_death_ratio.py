import grad_rise
import numpy as np


def predict_house_death():

    # 读取训练数据文件 和 测试数据文件
    fr_train = open('horseColicTraining.txt', 'r')
    fr_test = open('horseColicTest.txt', 'r')

    # 定义训练自变量数据集 和 训练因变量数据标签
    data_set_train = []
    data_label_train = []

    for l in fr_train.readlines():
        l_list = l.strip().split('\t')
        data_set_train.append([float(i) for i in l_list[:-1]])
        data_label_train.append(int(float(l_list[-1])))

    # 计算权重
    train_weights = grad_rise.grad_rise_method(np.array(data_set_train), data_label_train)[-1]
    print(train_weights)
    error_count = 0
    num_test_data_set = 0
    for l in fr_test.readlines():
        num_test_data_set += 1
        l_list = l.strip().split('\t')
        data_set_test = np.array([float(i) for i in l_list[:-1]])
        if int(grad_rise.classify_probablity_to_bool(sum(data_set_test * train_weights))) != int(l_list[-1]):
            error_count += 1

    return float(error_count) / num_test_data_set


if __name__ == '__main__':
    error_ratio = predict_house_death()
    print(error_ratio)