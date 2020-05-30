import numpy as np
import re


# 加载测试型数据
def posting_list():
    posting_list = [
        'my dog has flea problems help please',
        'maybe not take him to dog park stupid',
        'my dalmation is so cute i love him',
        'stop posting stupid worthless garbage',
        'mr licks ate my steak how to stop him',
        'quit buying worthless dog food stupid'
    ]
    feelings_vector = [0, 1, 0, 1, 0, 1]
    return posting_list, feelings_vector


# 将语句转为 词库 集合
def word_set(data_set):
    """
    将语句列表 去除特殊符号后 切割成 单词 并存储到 集合中
    :param data_set: list 语句列表
    :return: 列表 因为 set具有无序性
    """
    words = set([])
    for sentence in data_set:
        # sentence = re.sub(r"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", " ", sentence)
        new_word_set = set(sentence.strip().split(' '))
        words = words | new_word_set
    return list(words)


# 将语句转化为01向量
def sentence_to_vector(word_list, input_sentence):
    """
    通过输入 词库信息 和 输入单个语句
    :param word_list: list 词库信息
    :param input_sentence: string 没有标点符号的单个语句
    :return: list 0/1向量
    """
    input_words = input_sentence.strip().split(' ')
    # 构造一个与词库等长的 占位向量
    sentence_vector = [0] * len(word_list)
    # 在占位向量的 部分位置 填充 1
    for word in input_words:
        if word in word_list:
            sentence_vector[word_list.index(word)] += 1
        else:
            print('word" %s is not in my word_list' % word)
    return sentence_vector


# 根据输入的 文档内容，计算每个词汇 的在侮辱性文档 和 非侮辱性文档中 出现的概率
def train_docs(train_sentence_matrix, train_sentence_labels):
    """
    根据输入的 文档内容，计算每个词汇的在侮辱性文档 和 非侮辱性文档中 出现的概率
    1 为 侮辱性语句 0 为 非侮辱性语句
    :param train_sentence_matrix: narray  输入为0，1矩阵(M x N), M 为语句数， N为词库长度
    :param train_sentence_labels: list 输入为0，1 向量 长度为 语句数
    :return:
    """
    nrow, ncol = train_sentence_matrix.shape
    # 寻找出 侮辱性语句 和 非侮辱性语句的index list
    feelings1_list = [index for (index, value) in enumerate(train_sentence_labels) if value == 1]
    feelings0_list = [index for (index, value) in enumerate(train_sentence_labels) if value == 0]
    # 找出 侮辱性语句 和 非侮辱性语句 的 矩阵
    feelings1_matrix = train_sentence_matrix[feelings1_list, ]
    feelings0_matrix = train_sentence_matrix[feelings0_list, ]

    # 矩阵行向量累加
    # feelings1_p_value = feelings1_matrix.sum(axis=0) / feelings1_matrix.sum()
    # feelings0_p_value = feelings0_matrix.sum(axis=0) / feelings0_matrix.sum()

    # 为了避免出现概率为 0 的情况  故使用以下矩阵行向量计算
    feelings1_p_value = (feelings1_matrix + 1).sum(axis=0) / (feelings1_matrix + 1).sum()
    feelings0_p_value = (feelings0_matrix + 1).sum(axis=0) / (feelings0_matrix + 1).sum()

    all_p_value = sum(train_sentence_labels) / nrow

    # return feelings0_p_value, feelings1_p_value, all_p_value
    # 为避免出现 由于过多 概率 乘积 过小 被 python 认为为0的情况， 我们选用log
    return np.log(feelings0_p_value), np.log(feelings1_p_value), all_p_value


if __name__ == '__main__':
    # 将语句转入词库
    data_set, feelings_vector = posting_list()
    word_list = word_set(data_set)
    print(word_list)
    print(sentence_to_vector(word_list, data_set[0]))

    #####################################
    # 生成语句0/1矩阵
    train_sentences_matrix = []
    for i in range(len(data_set)):
        sentence_vector = sentence_to_vector(word_list, data_set[i])
        train_sentences_matrix.append(sentence_vector)
    train_sentences_matrix = np.array(train_sentences_matrix)
    feelings0_p_value, feelings1_p_value, all_p_value = train_docs(train_sentences_matrix, feelings_vector)
    print(feelings0_p_value)
    print(feelings1_p_value)
    print(all_p_value)

    ######################################
    # 对上述分类模型进行测试
    test_data = ['love my dalmation', 'stupid garbage']
    for sentence in test_data:
        sentence_vector = sentence_to_vector(word_list, sentence)
        p1 = sum(sentence_vector * feelings1_p_value) + np.log(all_p_value)
        p0 = sum(sentence_vector * feelings0_p_value) + np.log(1 - all_p_value)
        print(p1 > p0)

    ########################################
