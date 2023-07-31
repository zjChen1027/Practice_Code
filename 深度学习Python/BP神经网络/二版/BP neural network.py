import numpy as np
import matplotlib.pyplot as plt  # 图像绘制库

"""
    bp神经网络示例(可编辑多层神经网络)
    Logistic 回归模型
    
    Author : Chen
    Data   : 2023.7.11 程序整体与sigmoid激活函数部分完成
"""


def parameter_initializer(n1, n2, n3):
    """
    参数初始化
    :param n1: n_feature  样本特征值
    :param n2: n_neurone  各层神经元个数
    :param n3: n_label    标签维度
    :return para: wi bi   神经网络参数
    """
    n_nd = len(n2)  # 神经网络的深度
    para = {}  # 各层参数
    para.setdefault('w1', np.random.normal(scale=(2.0 / n1) ** 0.5, size=(n2[0], n1)))
    para.setdefault('b1', np.zeros(shape=(n2[0], 1)))
    if n_nd > 1:
        for i in range(n_nd - 1):
            para.setdefault('w' + str(i + 2), np.random.normal(scale=(2.0 / n2[i]) ** 0.5, size=(n2[i + 1], n2[i])))
            para.setdefault('b' + str(i + 2), np.zeros(shape=(n2[i + 1], 1)))
    # 最后一层神经网络参数
    para.setdefault('w' + str(n_nd + 1),
                    np.random.normal(scale=(2.0 / n2[n_nd - 1]) ** 0.5, size=(n3, n2[n_nd - 1])) * 0.001)
    para.setdefault('b' + str(n_nd + 1), np.zeros(shape=(n3, 1)))
    return para


def active_fun(z, type):
    """
    激活函数
    :param z:           线性输入
    :param type:        激活函数选择
    :return   a:        激活函数输出值
    """
    if type == 'sigmoid':  # sigmoid 激活函数
        a = 1 / (1 + np.exp(-z))
    elif type == 'tanh':  # tanh(反正切)激活函数
        a = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    elif type == 'relu':  # relu(线性修正单元)激活函数
        a = np.maximum(0, z)
    return a


def loss(pred_y, y):
    """ 损失函数 """
    cross_entropy = -((1 - y) * np.log(1 - pred_y) + y * np.log(pred_y))  # logistic回归函数的损失函数
    cost = np.mean(np.sum(cross_entropy, axis=0, keepdims=True))
    return cost


def forward_propagate(x, para, n_neurone, actfun):
    """
    前向传播
    :param x:                  样本值
    :param para:               神经网络参数
    :param n_neurone:          隐藏层各层神经单元个数
    :return: a:                中间值(层激活函数输出值)
     y_predict:                预测标签
    """
    n_nd = len(n_neurone)
    a = {'a0': x}
    for i in range(n_nd):
        z = np.dot(para['w' + str(i + 1)], a['a' + str(i)]) + para['b' + str(i + 1)]
        a['a' + str(i + 1)] = active_fun(z, actfun[0])
    # 最后一层神经网络
    z = np.dot(para['w' + str(n_nd + 1)], a['a' + str(n_nd)]) + para['b' + str(n_nd + 1)]
    y_predict = active_fun(z, actfun[1])
    return a, y_predict


def back_propagate(para, a, y_predict, y, learning_rate):
    """
    反向传播
    :param para:             神经网络参数
    :param a:                各层激活函数输出值
    :param y_predict:        预测值
    :param y:                样本标签
    :param learning_rate:    学习率
    :return para:            神经网络参数
    """
    # 反向传播
    m = y.shape[1]  # 样本个数
    k = len(para)  # 变量数
    # 输出层梯度
    para_bp = {}
    para_bp.setdefault('dz' + str(k // 2), y_predict - y)  # dw
    para_bp.setdefault('dw' + str(k // 2),
                       1 / m * np.dot(para_bp['dz' + str(k // 2)], a['a' + str(k // 2 - 1)].T))  # dw
    para_bp.setdefault('db' + str(k // 2), 1 / m * np.sum(para_bp['dz' + str(k // 2)], axis=1, keepdims=True))  # db
    # 输出层梯度下降
    para['w' + str(k // 2)] = para['w' + str(k // 2)] - learning_rate * para_bp['dw' + str(k // 2)]
    para['b' + str(k // 2)] = para['b' + str(k // 2)] - learning_rate * para_bp['db' + str(k // 2)]
    for i in range(k // 2 - 1, 0, -1):
        # 隐藏层参数更新
        para_bp.setdefault('dz' + str(i),
                           np.dot(para['w' + str(i + 1)].T, para_bp['dz' + str(i + 1)]) * ((1 - a['a' + str(i)]) * a[
                               'a' + str(i)]))  # dz
        para_bp.setdefault('dw' + str(i), 1 / m * np.dot(para_bp['dz' + str(i)], a['a' + str(i - 1)].T))  # dw
        para_bp.setdefault('db' + str(i), 1 / m * np.sum(para_bp['dz' + str(i)], axis=1, keepdims=True))  # db
        # 反向传播
        para['w' + str(i)] = para['w' + str(i)] - learning_rate * para_bp['dw' + str(i)]
        para['b' + str(i)] = para['b' + str(i)] - learning_rate * para_bp['db' + str(i)]
    return para


def bpnn(x, y, n_neurone, epochs, learning_rate, actfun):
    """
    反向传播神经网络
    :param x:                   样本
    :param y:                   样本标签
    :param n_neurone:           各层隐藏层神经元数
    :param epochs:              训练次数
    :param learning_rate:       学习率
    :return para:               神经网络参数
        cost_all:               各次训练代价
    """
    n_feature = x.shape[0]  # 特征维度(样本特征数)
    n_label = y.shape[0]  # 标签维度
    cost_all = np.zeros(shape=(1, epochs))
    # 参数初始化
    para = parameter_initializer(n_feature, n_neurone, n_label)
    for i in range(epochs):
        # 前向传播
        a, y_predict = forward_propagate(x, para, n_neurone, actfun)
        # 损失
        cost = loss(y_predict, y)
        cost_all[0, i] = cost
        print("{}epoch, cost is {}".format(i + 1, cost))
        # 反向传播
        para = back_propagate(para, a, y_predict, y, learning_rate)
    return para, cost_all


if __name__ == '__main__':
    # 训练次数
    epochs = 3000
    # 学习率
    learning_rate = 0.3
    # 激活函数
    actfun = ['sigmoid', 'sigmoid']  # 注: actfun[0] 表示隐藏层激活函数  actfun[1] 表示输出层激活函数
    # 各层神经元数量(注: 最后一层网络以确定，仅配置中间隐藏层神经元数量)
    n_neurone = np.array([5, 4, 3])

    # 好瓜、坏瓜模型  分别对应0-2
    # 色泽(青绿、乌黑、浅白)、根蒂(蜷缩、稍缩、硬挺)、敲声(清脆、浑浊、沉闷)、纹理(稍糊、清晰)、脐部(凹陷、稍凹、平坦)、触感(硬滑、软粘)
    x = np.array([[0, 1, 1, 0, 2, 0, 1, 1, 1, 0, 2, 2, 0, 2, 1, 2, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0, 0],
                  [1, 2, 1, 2, 1, 1, 1, 1, 2, 0, 0, 1, 1, 2, 1, 1, 2],
                  [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 1, 2, 1],
                  [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]])
    # 好坏(坏瓜、好瓜)
    y = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    print('输入x的尺寸{}'.format(x.shape))
    para, cost_all = bpnn(x, y, n_neurone, epochs, learning_rate, actfun)
    # 损失图绘制
    plt.plot(np.arange(1, epochs + 1), cost_all[0, :])
    plt.title('loss image')
    plt.show()

    # 测试
    x_predict1 = np.array([[0, 0, 1, 1, 0, 0]]).T  # 样本好瓜
    x_predict2 = np.array([[1, 1, 2, 0, 1, 0]]).T  # 样本坏瓜
    a, y_predict = forward_propagate(x_predict2, para, n_neurone, actfun)
    print('测试结果为{}'.format(y_predict))
