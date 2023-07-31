close all; clear; clc;
% 采用梯度下降算法的线性回归算法
% 
% Batch梯度下降法(Batch Gradient Descent)
% 
% 梯度下降算法的问题：容易陷入局部最优解。
% 线性回归算法的特点：代价函数总是弓状函数(凸函数)，即无局部最优解，只有全局最优解。
% 
% 修改日期：2023.1.29
%===============================================================Author:Chen
%% 样本初始化
m = 40;   % 样本个数
% 生成样本集合
y = 10*rand(1,m);   % 生成m个0-10之间的随机数(纵坐标)
x = 10*rand(1,m);   % 生成m个0-10之间的随机数(横坐标)
% 绘制样本点集合
figure(NumberTitle="off",Name='样本点图')
scatter(x,y,'k'); title("观测样本点图"); hold on;
%% 线性回归
syms Theta0 Theta1 X;  % 定义模型参数与变量
F = Theta0 + Theta1*X; % 单变量线性回归模型
% 预测模型函数
H = subs(F,X,x);  % 将样本点的横坐标带入模型

% 采用平方误差函数作为代价函数
J = (1/2/m)*sum((H - y).^2); % 代价函数

% 假设模型参数 Theta0 与 Theta1 的初始值为 3,-2
orig_0 = 3; orig_1 = -2;
orig_F = subs(subs(F,Theta0,orig_0),Theta1,orig_1);

% 初始直线绘制
plot(x,subs(orig_F,X,x),'r')

%% 梯度下降算法
% 通过梯度下降算法求线性回归代价函数 J(Theta0,Theta1) 的最小值
Theta_d = [diff(J,Theta0),diff(J,Theta1)];   % 计算偏导数

% 进行 k 次迭代，学习率为 α
% α 过大导致发散，α 过小导致收敛速度过慢
k = 100; alpha = 0.05;
for i = 1:k   % 梯度下降算法
    temp0 = orig_0 - alpha*subs(subs(Theta_d(1),Theta0,orig_0),Theta1,orig_1);
    temp1 = orig_1 - alpha*subs(subs(Theta_d(2),Theta0,orig_0),Theta1,orig_1);
    orig_0 = temp0; orig_1 = temp1; % 更新迭代值
end

% 绘制线性回归后的直线方程
LR_F = subs(subs(F,Theta0,orig_0),Theta1,orig_1);
plot(x,subs(LR_F,X,x),'g')
legend('','初始拟合直线','拟合后的直线')






