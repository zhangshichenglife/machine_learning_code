import numpy as np
import bayes_word_bag
import re
import random

# 仅提取文件中的单词 并且 去除长度小于等于2的单词
def text_parse(input_sentence):
    """
    语句切割
    :param input_sentence: string
    :return: 单词列表
    """
    word_list = re.split(r'\W+', input_sentence)
    return [word.lower() for word in word_list if len(word) > 2]


# 垃圾邮件测试函数 其中 spam 为垃圾邮件文件夹，ham为 非垃圾邮件文件夹
def spam_test():
    # doc_list 为输入文件内容剔除 特殊符号的语句列表
    doc_list = []
    # spam_vector = 为输入是否为垃圾邮件
    spam_vector = []
    # 读取文件, 并且存入到词库中
    # 如果在读取文件中出现编码错误；请查看
    # https://blog.csdn.net/wiki347552913/article/details/88060582
    # https://blog.csdn.net/liy010/article/details/79504006
    for i in range(1, 26):  # 这里的1:26为 spam文件夹下的文件名称
        spam_doc = open('email/spam/%d.txt' % i, 'r').read()
        print(spam_doc)
        doc_word_list = text_parse(spam_doc)
        print(doc_word_list)
        doc_list.append(' '.join(doc_word_list))
        spam_vector.append(1)
    for i in range(1, 26):  # 这里的1:26为 ham文件夹下的文件名称
        ham_doc = open('email/ham/%d.txt' % i, 'r').read()
        doc_word_list = text_parse(ham_doc)
        doc_list.append(' '.join(doc_word_list))
        spam_vector.append(0)
    # 将doc_list 语句列表转化为词库
    print(doc_list)
    word_list = bayes_word_bag.word_set(doc_list)
    print(word_list)

    # 从doc_list 随机选取10个作为测试集test_set，所有的 或者一部分为训练集data_set
    test_set_index = random.sample(range(50), 50)
    print(test_set_index)
    data_set = doc_list[:50]

    # 生成语句0/1矩阵
    train_sentences_matrix = []
    for i in range(len(data_set)):
        sentence_vector = bayes_word_bag.sentence_to_vector(word_list, data_set[i])
        train_sentences_matrix.append(sentence_vector)
    train_sentences_matrix = np.array(train_sentences_matrix)
    feelings0_p_value, feelings1_p_value, all_p_value = bayes_word_bag.train_docs(train_sentences_matrix, spam_vector)
    print(feelings0_p_value)
    print(feelings1_p_value)
    print(all_p_value)

    # 对测试集 进行预测
    for i in test_set_index:
        sentence = data_set[i]
        sentence_vector = bayes_word_bag.sentence_to_vector(word_list, sentence)
        p1 = sum(sentence_vector * feelings1_p_value) + np.log(all_p_value)
        p0 = sum(sentence_vector * feelings0_p_value) + np.log(1 - all_p_value)
        print(i, int(p1 > p0) == spam_vector[i])


if __name__ == '__main__':
    spam_test()