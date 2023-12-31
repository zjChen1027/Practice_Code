import numpy as np
'''
    Numpy库使用方法
    
    import numpy.linalg as nla  # 线性代数模块
    import numpy.random as npr  # 随机数模块
    
    
    
    Author : Chen
    Data   : 2023.7.4
'''
# 基本计算方式
'==============================================================='
x = np.array([1.2, 54, 23, 2.3])  # 生成np数组
# 最值操作
y = np.argmax(x)  # 数组x中最大值得的下标(多个最值时返回第一个最值的下标)
print('最大值对应的下标', y, type(y))

# 取整操作
z = np.ceil(x[0])  # 向上取整
k = np.floor(x[0])  # 向下取整
print('向上取整', z, type(z))
print('向下取整', k, type(k))
'==============================================================='
# np算数运算
'==============================================================='
x = np.array([[1, 3, 5], [7, 9, 11]])  # 创建多维数组
y = np.array([[2, 4, 6], [8, 10, 12]])
# 元素对应加法
print('矩阵加法:', x + y)
# 元素对应乘法
print('矩阵对应元素乘法:', x * y)
# 元素对应除法
print('矩阵对应元素除法:', x / y)
# 矩阵乘法
x = np.array([1, 2, 3])
y = np.array([[1], [2], [3]])
print('矩阵乘法:', np.dot(x, y))  # 1*1 + 2*2 + 3*3
'dot的应用神经网络的应用'
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)
print(Y)
'==============================================================='
# np 访问元素
'==============================================================='
# 元素访问(元素索引从0开始)
X = np.array([[11, 12], [23, 24], [53, 54]])
print('元素访问:', X[1, 1])
# 转换为一维数组
X = X.flatten()
print('转换为一维数组:', X)
# 获取对应位置元素
print('对应元素获取:', X[np.array([0, 2, 4])])
# 获取满足一定条件的元素
print('获取条件元素:', X[X > 23])
'==============================================================='
# 多维数组
'==============================================================='
# 数组维度
# ndim返回维度
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Y = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
print('ndim返回维度')
print('数组X:', X, '\n', '获取数组维度', X.ndim)  # 2维数组
print('数组Y:', Y, '\n', '获取数组维度', Y.ndim)  # 3维数组
# shape返回维度
print('shape返回维度')
print('数组X:', X, '\n', '获取数组维度', X.shape)  # 2维数组
print('数组Y:', Y, '\n', '获取数组维度', Y.shape)  # 3维数组
# 数组基本索引与切片
print('数组切片:')
print(X[1:3, 0:2])
print(Y[0:2, 0:1, 0:1])
# 使用布尔代数索引
X = np.array([1, 2, 3, 4, 1, 5, 2])
Y = np.array([True, False, True, False, False, True, True])
print('布尔代数索引:')
print(X[Y == False])
print(X[Y])
'==============================================================='
# 创建数组
'==============================================================='
# 创建全1数组
X = np.ones((5, 5))
# 创建全0数组
Y = np.zeros((5, 5))
# 创建单位数组
Z = np.eye(5)
print('创建全1数组:\n', X, type(X), '\n',
      '创建全0数组:\n', Y, type(Y), '\n',
      '创建单位数组:\n', Z, type(Z), '\n')
'==============================================================='
# 通用函数
'==============================================================='
'''
 abs、fabs                            计算绝对值，对于复数fabs更快
 sqrt                                 计算各元素平方根
 square                               计算各元素平方
 exp                                  计算各元素指数
 log、log10、log2、log1p
 sign                                 计算各元素正负号(1：正数；0：零；-1：负数)
 ceil                                 向上取整
 floor                                向下取整
 cos、cosh、sin、sinh、tan、tanh        普通型和双曲型三角函数
 sum
 mean
 std、var                              标准差和反差，自由度可调(默认为n)
 min、max                              最大值与最小值
 argmin、argmax                        最大值与最小值索引
'''
X = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]],
              [[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]])
print('通用函数:')
print('数组X的维度:', X.shape)
print('最大值(函数):', np.max(X))
print('最大值(索引):', X[3, 2, 1])
print('对每行求均值:', X.mean(axis=1))  # 对每行求平均值
print('对每行求均值:', X.mean(axis=0))  # 对每列求平均值
# 注: [[(1+7+13+19)/4, (2+8+14+20)/4], ...] 注意X的维度为 4*3*2，因此求每列平均为X[1,1,1]+X[2,1,1]+...
'==============================================================='
# numpy线性代数
import numpy.linalg as nla  # 线性代数模块

# 其他调用方法: numpy.linalg.xxx
'==============================================================='
'''
 diag                                  将一维序列转换为对角矩阵或将返回方阵的对角线(位于numpy库内)
 dot                                   矩阵乘法
 trace                                 方阵的迹
 det                                   计算记者行列式
 eig                                   计算方阵的特征值与特征向量
 inv                                   计算方阵逆矩阵
 pinv                                  计算伪逆矩阵
 qr                                    QR分解
 svd                                   奇异值分解
 solve                                 求解线性方程组Ax=b
 istsq                                 求解最小二乘解
'''
x = np.array([1, 2, 3])
y = np.diag(x)
print('生成对角矩阵:', y)
print('矩阵对角元素:', np.diag(y))
print('特征值，特征向量求解:', nla.eig(y))
'==============================================================='
# numpy随机数生成
import numpy.random as npr  # 随机数模块

# 其他调用方法: numpy.random.xxx
'==============================================================='
'''
 seed                                  确定随机数生成器的种子
 shuffle                               对一个序列就地排列
 rand                                  生成均匀分布序列
 randint                               给定上下限范围内随机选取整数
 randn                                 产生正态分布
 uniform                               产生[0, 1]均匀分布
 normal                                产生正态(高斯)分布
 binomial                              产生二项分布
 normal、binomial参数:
 n: int型或者int型数组(向下取整);          p: 概率; 
 size: 输出值的大小                      return: 返回值
'''
X = npr.randint(0, 2, size=10)
print('十次抛硬币伯努利实验:', X)
print('正面次数:', (X > 0).sum())
# 分布函数举例:
# 十个样本，每个样本发生概率为0.5，重复实验10次
print('二项分布:', npr.binomial(10, 0.5, 10))
print('正态分布:', npr.normal(10, 0.5, 10))
'==============================================================='
# 技巧:
""" keepdims=True 的作用"""
# example:
y = np.arange(1, 10).reshape(3, 3)
print('方阵y为:\n', y)
print('按行求和(错误):\n', np.sum(y, axis=1))
print('按行求和(正确):\n', np.sum(y, axis=1, keepdims=True))
""" 直接使用sum求和时，使用axis会按着该维度进行求和，但同时也会消除该维度。使用
    keepdims=True 即可保持保持维度的正确性 """
'==============================================================='
