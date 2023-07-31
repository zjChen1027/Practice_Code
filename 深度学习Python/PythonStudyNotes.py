"""
    Python学习笔记

"""

import numpy as np               # Numpy矩阵库
import numpy.linalg as nla       # Numpy线性代数模块
import numpy.random as npr       # Numpy随机数模块
import matplotlib.pyplot as plt  # 图像绘制库

if __name__ == '__main__':
    subject = 3
    match subject:
        case 1:
            """Python Broadcasting(广播)实例"""
            matrix = np.array([[56, 0, 4.4, 68],
                               [1.2, 104, 52, 8],
                               [1.8, 135, 99, 0.9]])
            sum_matrix = matrix.sum(axis=0, keepdims=True)
            print('Total heat:', sum_matrix)
            percentage = 100*matrix/sum_matrix.reshape(1, 4)  # reshape来确保矩阵尺寸合适
            print(percentage)
            """
                Python广播详解: 任意(m*n)矩阵与(1*n)或(m*1)计算，Python将自动对其进行扩展，变为(m*n)的矩阵进行运算
                适用于 + - * /  (Matlab中类似函数 bsxfun )
            """

        case 2:
            """编程技巧"""
            # N.1
            temp = npr.randn(3)
            print(temp.shape)
            '注: 此时的数组结构为未知状态，即Python中秩为1的数组。此时它即不为行向量，也不为列向量。'
            print(np.dot(temp, temp.T))
            '此时若为外积应为矩阵，内积为数字，在后续处理中造成歧义。'
            '正确做法:'
            temp = npr.randn(3, 1)
            print(temp.shape)
            print(np.dot(temp, temp.T))
            """
                当不确定一个向量的具体维度时，采用assert声明定义。确保该向量为列/行向量。
                因为某些原因得到秩为1的数组时，可采用reshape调整数组。
            """
            assert(temp.shape == (3, 1))
            temp = temp.reshape(3, 1)

        case 3:
            a = 1
