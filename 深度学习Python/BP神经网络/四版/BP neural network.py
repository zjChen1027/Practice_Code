import numpy as np
import matplotlib.pyplot as plt  # 图像绘制库

"""
    bp神经网络示例(可编辑多层神经网络)
    四种激活函数的bpnn示例
    隐藏层和输出层激活层可分别选择
    Logistic 回归模型
    
    Author : Chen
    Data   : 2023.7.11 程序整体与sigmoid激活函数部分完成
             2023.7.12 全部完成
             2023.7.18 正则化项、输入归一化
"""


def initializer_fun(n1, n2, type):
    """
    参数w初始化模式
    :param n1:        本层特征维度(神经单元个数)
    :param n2:        下一层特征维度(神经单元个数)
    :param type:      初始化类型
    :return   w:      初始化值
    """
    if type == 'Sigmoid':
        # 针对于sigmoid、tanh等激活函数初始化
        w = np.random.randn(n2, n1) * 0.01
    elif type == 'tanh':
        # 针对于tanh
        w = np.random.randn(n2, n1) * np.sqrt(1/n1)
        # w = np.random.randn(n2, n1) * np.sqrt(2/(n1+n2))
    elif type == 'ReLU':
        # 针对于ReLU等
        w = np.random.randn(n2, n1) * np.sqrt(2/n1)
    else:
        # 初始化类型错误
        print("初始化选择错误，默认模式")
        w = np.random.normal(scale=(2.0 / n1) ** 0.5, size=(n2, n1))
    return w


def parameter_initializer(n1, n2, n3, type):
    """
    参数初始化
    :param n1: n_feature  样本特征值
    :param n2: n_neurone  各层神经元个数
    :param n3: n_label    标签维度
    :param type:          初始化类型
    :return para: wi bi   神经网络参数
    """
    para = {}  # 各层参数
    n2 = [n1] + n2
    n_nd = len(n2)  # 神经网络的深度
    # 隐藏层参数初始化
    for i in range(n_nd - 1):
        para.setdefault('w' + str(i + 1), initializer_fun(n2[i], n2[i+1], type[0]))
        para.setdefault('b' + str(i + 1), np.zeros(shape=(n2[i+1], 1)))
    # 输出层参数初始化
    para.setdefault('w' + str(n_nd), initializer_fun(n2[n_nd - 1], n3, type[1]))
    para.setdefault('b' + str(n_nd), np.zeros(shape=(n3, 1)))
    return para


def active_fun(z, actfun):
    """
    激活函数
    :param z:           线性输入
    :param actfun:      激活函数选择
    :return a:          激活函数输出值
    """
    if actfun == 'sigmoid':  # sigmoid 激活函数
        a = 1 / (1 + np.exp(-z))
    elif actfun == 'tanh':  # tanh(反正切)激活函数
        a = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    elif actfun == 'relu':  # relu(线性修正单元)激活函数
        a = np.maximum(0, z)
    elif actfun == 'leaky_relu':  # 泄露的ReLU激活函数
        a = np.maximum(0.001 * z, z)
    else:
        print('激活函数选择错误，默认使用sigmoid')
        a = 1 / (1 + np.exp(-z))
    return a


def loss(pred_y, y, reg):
    """ 损失函数 """
    cross_entropy = -((1 - y) * np.log(1 - pred_y) + y * np.log(pred_y))  # logistic回归函数的损失函数
    cost = np.mean(np.sum(cross_entropy, axis=0, keepdims=True) + reg)
    return cost


def normalization(x):
    """
    输入归一化
    :param x:      样本值
    :return x:     归一化后的样本值
    """
    m = x.shape[1]
    # 零均值化
    mu = 1/m * np.sum(x, axis=1, keepdims=True)
    x = x - mu

    # 方差归一化
    var = 1/m * np.sum(x**2, axis=1, keepdims=True)
    x = x / np.sqrt(var)
    return x


def forward_propagate(x, para, n_neurone, actfun):
    """
    前向传播
    :param x:                  样本值
    :param para:               神经网络参数
    :param n_neurone:          隐藏层各层神经单元个数
    :param actfun:             隐藏层与输出层激活函数类型
    :return: a:                中间值(层激活函数输出值)
     y_predict:                预测标签
    """
    n_nd = len(n_neurone)
    a = {'a0': x}
    # 隐藏层神经网络
    for i in range(n_nd):
        # z = w[i].Ta[i] + b[i]
        z = np.dot(para['w' + str(i + 1)], a['a' + str(i)]) + para['b' + str(i + 1)]
        # a[i+1] = g(z)  g(x)为激活函数
        a['a' + str(i + 1)] = active_fun(z, actfun[0])
    # 输出层神经网络
    z = np.dot(para['w' + str(n_nd + 1)], a['a' + str(n_nd)]) + para['b' + str(n_nd + 1)]
    # 预测标签
    y_predict = active_fun(z, actfun[1])
    return a, y_predict


def back_propagate(para, a, y_predict, y, learning_rate, actfun, regularization):
    """
    反向传播
    :param para:             神经网络参数
    :param a:                各层激活函数输出值
    :param y_predict:        预测值
    :param y:                样本标签
    :param learning_rate:    学习率
    :param actfun:           隐藏层与输出层激活函数类型
    :param regularization:   正则化参数
    :return para:            神经网络参数
    """
    # 反向传播
    m = y.shape[1]  # 样本个数
    k = len(para)  # 变量数
    para_bp = {}
    # 预测标签梯度值
    dy_predict = -(y / y_predict) + (1 - y) / (1 - y_predict)  # dL/dy_predict

    # 输出层梯度
    if actfun[1] == 'sigmoid':
        dy_predict_dz = y_predict * (1 - y_predict)
    elif actfun[1] == 'tanh':
        dy_predict_dz = 1 - y_predict * y_predict
    elif actfun[1] == 'relu':
        dy_predict_dz = np.where(y_predict >= 0, 1, 0)
    else:
        dy_predict_dz = np.where(y_predict >= 0, 1, 0.001)

    # 正则化项
    if regularization[0, 0] == 1:
        # 添加正则化项
        w = para['w' + str(k // 2)]
        if regularization[0, 1] == 1:
            # L1正则化
            reg = regularization[0, 2]
        elif regularization[0, 1] == 2:
            # L2正则化
            reg = regularization[0, 2] * w
        else:
            # 默认L2正则化
            reg = regularization[0, 2] * w
    else:
        reg = 0
    # dz
    para_bp.setdefault('dz' + str(k // 2), dy_predict * dy_predict_dz)
    # dw
    para_bp.setdefault('dw' + str(k // 2),
                       1 / m * (np.dot(para_bp['dz' + str(k // 2)], a['a' + str(k // 2 - 1)].T) + reg))
    # db
    para_bp.setdefault('db' + str(k // 2), 1 / m * np.sum(para_bp['dz' + str(k // 2)], axis=1, keepdims=True))
    # 输出层梯度更新
    para['w' + str(k // 2)] = para['w' + str(k // 2)] - learning_rate * para_bp['dw' + str(k // 2)]
    para['b' + str(k // 2)] = para['b' + str(k // 2)] - learning_rate * para_bp['db' + str(k // 2)]
    # 隐藏层反向传播
    for i in range(k // 2 - 1, 0, -1):
        # 正则化项
        if regularization[0, 0] == 1:
            # 添加正则化项
            w = para['w' + str(i)]
            if regularization[0, 1] == 1:
                # L1正则化
                reg = regularization[0, 2]
            elif regularization[0, 1] == 2:
                # L2正则化
                reg = regularization[0, 2] * w
            else:
                # 默认L2正则化
                reg = regularization[0, 2] * w
        else:
            reg = 0
        # 激活函数
        if actfun[0] == 'sigmoid':
            dadz = ((1 - a['a' + str(i)]) * a['a' + str(i)])
        elif actfun[0] == 'tanh':
            dadz = 1 - (a['a' + str(i)] * a['a' + str(i)])
        elif actfun[0] == 'relu':
            dadz = np.where(a['a' + str(i)] > 0, 1, 0)
        else:
            dadz = np.where(a['a' + str(i)] > 0, 1, 0.001)
        # dz
        para_bp.setdefault('dz' + str(i), np.dot(para['w' + str(i + 1)].T, para_bp['dz' + str(i + 1)]) * dadz)
        # dw
        para_bp.setdefault('dw' + str(i), 1 / m * (np.dot(para_bp['dz' + str(i)], a['a' + str(i - 1)].T) + reg))
        # db
        para_bp.setdefault('db' + str(i), 1 / m * np.sum(para_bp['dz' + str(i)], axis=1, keepdims=True))
        # 隐藏层梯度更新
        para['w' + str(i)] = para['w' + str(i)] - learning_rate * para_bp['dw' + str(i)]
        para['b' + str(i)] = para['b' + str(i)] - learning_rate * para_bp['db' + str(i)]
    return para


def bpdnn(x, y, n_neurone, epochs, learning_rate, actfun, initialize_type, regularization):
    """
    反向传播深度神经网络
    :param x:                   样本
    :param y:                   样本标签
    :param n_neurone:           各层隐藏层神经元数
    :param epochs:              训练次数
    :param learning_rate:       学习率
    :param actfun:              隐藏层与输出层激活函数类型
    :param initialize_type:     参数初始化模式(隐藏层参数与输出层参数单独选择)
    :param regularization:      正则化参数
    :return para:               神经网络参数
        cost_all:               各次训练代价
    """
    n_feature = x.shape[0]  # 特征维度(样本特征数)
    n_label = y.shape[0]  # 标签维度
    cost_all = np.zeros(shape=(1, epochs))
    # 参数初始化
    para = parameter_initializer(n_feature, n_neurone, n_label, initialize_type)
    # 输入归一化
    x = normalization(x)
    for i in range(epochs):
        # 前向传播
        a, y_predict = forward_propagate(x, para, n_neurone, actfun)
        # 正则化项
        if regularization[0, 0] == 1:
            reg = 0
            if regularization[0, 1] == 1:
                # L1 正则化
                for j in range(len(para) // 2):
                    reg += np.sum(np.abs(para['w' + str(j+1)]))
            else:
                # L2 正则化
                for j in range(len(para) // 2):
                    reg += np.sqrt(np.sum(para['w' + str(j+1)] ** 2))
            reg = reg * regularization[0, 2]/2
        else:
            reg = 0
        # 损失
        cost = loss(y_predict, y, reg)
        cost_all[0, i] = cost
        print("{}epoch, cost is {}".format(i + 1, cost))
        # 反向传播
        para = back_propagate(para, a, y_predict, y, learning_rate, actfun, regularization)
    return para, cost_all


if __name__ == '__main__':
    "================配置参数================"
    # 训练次数
    epochs = 500
    # 学习率
    learning_rate = 0.05
    # 激活函数
    actfun = ['relu', 'sigmoid']  # 注: actfun[0] 表示隐藏层激活函数  actfun[1] 表示输出层激活函数
    # 隐藏层和输出层参数初始化类型
    # 注: Sigmoid  tanh  ReLU
    initialize_type = ['ReLU', 'Sigmoid']
    # 各层神经元数量(注: 最后一层网络以确定，仅配置中间隐藏层神经元数量)
    n_neurone = [5, 6, 8, 11]
    # 正则化参数
    regularization = np.array([[1, 2, 0.0001]])  # 正则化参数 [0,0]:正则化使能位; [0,1]:正则化模式选择; [0,2]:lambda
    # 测试模式
    pattern = 1
    """
        模式1为选定隐藏层激活函数和输出层激活函数，bpnn网络示例
        模式2为隐藏层分别为相同训练次数四种激活函数的训练损失比较
    """
    "======================================="
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
    "======================================="
    print('输入样本的维度{}'.format(x.shape))
    match pattern:
        case 1:
            # 反向传播神经网络
            para, cost_all = bpdnn(x, y, n_neurone, epochs, learning_rate, actfun, initialize_type, regularization)
            # 损失图绘制
            plt.plot(np.arange(1, epochs + 1), cost_all[0, :])
            plt.title('loss image')
            plt.xlabel("Training Number")
            plt.ylabel("Costs")
            plt.show()

            # 测试
            x_predict1 = np.array([[0, 0, 1, 1, 0, 0]]).T  # 样本好瓜
            x_predict2 = np.array([[1, 1, 2, 0, 1, 0]]).T  # 样本坏瓜
            x_test = np.array([[1, 1, 2, 0, 1, 1]]).T  # 测试坏瓜
            a, y_predict = forward_propagate(x_predict2, para, n_neurone, actfun)
            print('测试结果为{}'.format(y_predict))
        case 2:
            # 各层激活层
            layer = {'hid_lay1': 'sigmoid',    'act_lay1': 'sigmoid',
                     'hid_lay2': 'tanh',       'act_lay2': 'sigmoid',
                     'hid_lay3': 'relu',       'act_lay3': 'sigmoid',
                     'hid_lay4': 'leaky_relu', 'act_lay4': 'sigmoid'
                     }
            # 标题
            hide_title = ['sigmoid', 'tanh', 'relu', 'leaky relu']
            # 各激活函数最后的损失值
            last_cost = np.zeros(shape=(1, 4))
            # 创建子图
            fig, ax = plt.subplots(2, 2, figsize=(15, 15))
            fig.tight_layout(h_pad=4, w_pad=4)
            fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
            for i in range(4):
                actfun = [layer['hid_lay' + str(i + 1)], layer['act_lay' + str(i + 1)]]
                # 反向传播神经网络
                para, cost_all = bpdnn(x, y, n_neurone, epochs, learning_rate, actfun, initialize_type, regularization)
                # 保存最终达到的损失值
                last_cost[0, i] = cost_all[0, -1]
                # 图像位置确定
                if i < 2:
                    x_axis = 0
                else:
                    x_axis = 1
                if i % 2 == 0:
                    y_axis = 0
                else:
                    y_axis = 1
                # 损失图绘制
                ax[x_axis, y_axis].plot(np.arange(1, epochs + 1), cost_all[0, :])
                ax[x_axis, y_axis].set_title('Hidden layer:' + hide_title[i])
                ax[x_axis, y_axis].set_xlabel("Training Number")
                # ax[x_axis, y_axis].xaxis.set_label_coords(0.5, 0.01)
                ax[x_axis, y_axis].set_ylabel("Costs")
                # ax[x_axis, y_axis].yaxis.set_label_coords(0.01, 0.5)
            plt.show()
            print('4中隐藏层激活函数迭代{}次，最后损失值:{}、{}、{}、{}'.format(epochs, last_cost[0, 0], last_cost[0, 1],
                                                                            last_cost[0, 2], last_cost[0, 3]))
