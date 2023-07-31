import numpy as np                       # Numpy库
import matplotlib.pyplot as plt          # 图像绘制库
'BP神经网络实现'


def parameter_initializer(n1, n2, n3):
    """ 初始化权值 """
    w1 = np.random.normal(scale=(2.0/n1)**0.5, size=(n1, n2))
    b1 = np.zeros(shape=(1, n2))
    w2 = np.random.normal(scale=(2.0/n2)**0.5, size=(n2, n3))
    b2 = np.zeros(shape=(1, n3))
    return w1, b1, w2, b2


def sigmoid(z):
    """ Sigmoid 激活函数 """
    a = 1/(1+np.exp(-z))
    return a


def relu(z):
    a = np.fmax(0, z)
    return a


def forward_propagate(x, w1, b1, w2, b2):
    """ 构建三层神经网络 -- 前向传播 """
    """ x.shape = (m,n1)
        y.shape = (m, n3)
        w1.shape = (n, n2)
        b1.shape = (1, n2)
        w2.shape = (n2, n3)
        b2.shape = (1, n3)
    """
    z_1 = np.dot(x, w1) + b1
    a_1 = sigmoid(z_1)
    z_2 = np.dot(a_1, w2) + b2
    pred_y = sigmoid(z_2)
    return a_1, pred_y


def loss(pred_y, y):
    """ 损失函数 """
    cross_entropy = -((1-y)*np.log(1-pred_y) + y * np.log(pred_y))  # logistic回归函数的损失函数
    cost = np.mean(np.sum(cross_entropy, axis=1))
    return cost


def back_propagate(w1, b1, w2, b2, a_1, pred_y, y, learning_rate):
    """ 反向传播 """
    m = y.shape[0]
    dz_2 = pred_y - y  # dL/dz_2
    dw2 = 1/m * np.dot(a_1.T, dz_2)  # dL/dw_2
    db2 = 1/m * np.sum(dz_2, axis=0, keepdims=True)  # dL/db 即dL/dz_2

    # 梯度下降
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2

    dz_1 = np.dot(dz_2, w2.T) * ((1-a_1)*a_1)  # sigmoid(z_1)的导数  # (m, n2)
    dw1 = 1/m * np.dot(x.T, dz_1)  # (n1, n2)
    db1 = 1/m * np.sum(dz_1, axis=0, keepdims=True)  # (1, n2)

    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1

    return w1, b1, w2, b2


def bpnn(x, y, epochs, learning_rate):
    w1, b1, w2, b2 = parameter_initializer(4, 3, 2)
    cost_all = np.zeros(shape=(1, epochs))
    for epoch in range(epochs):
        a_1, pred_y = forward_propagate(x, w1, b1, w2, b2)
        cost = loss(pred_y, y)
        print("{} epoch, cost is {}".format(epoch, cost))
        cost_all[0, epoch] = cost
        w1, b1, w2, b2 = back_propagate(w1, b1, w2, b2, a_1, pred_y, y, learning_rate)
    return w1, b1, w2, b2, cost_all


if __name__ == "__main__":
    x = np.arange(0, 20).reshape(5, 4)
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], dtype=float).reshape(5, 2)
    epochs = 1500
    w1, b1, w2, b2, cost_all = bpnn(x, y, epochs, 0.1)
    print('训练参数分别为:\n w_1:\n{}\n b_1:\n{}\n w_2:\n{}\n b_2:\n{}'.format(w1, b1, w2, b2))

    # 图像绘制
    plt.plot(np.arange(1, epochs+1), cost_all[0, :])
    plt.title('loss image')
    plt.show()

    # 测试效果
    x_test = np.array([[1, 3, 2, 4], [4, 5, 7, 4], [7, 9, 12, 7]])
    output = forward_propagate(x_test, w1, b1, w2, b2)

    print(output)
