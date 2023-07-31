close all; clear; clc;
% 感知机函数
% eg.1 and eg.2
% 
% 修改时间：2023.2.3
eg_n = 1;
hyperplane = sym(zeros(1,2));
iteration_n = zeros(1,2);
%==============================================================Author：Chen
%% 书P40页例题 eg.1=========================================================
X = [3,3; 4,3; 1,1]';
Y = [1 1 -1];
Eta = 1;
ori_w = [0;0]; ori_b = 0;
%=================================================
P_P = find(Y == +1);  % 获得正类别点的坐标
N_P = find(Y == -1);  % 获得负类别点的坐标
% 样本图像绘制
figure(Name="eg.1样本点图")

scatter(X(1,P_P),X(2,P_P),'g');  % 正类标点
hold on;
scatter(X(1,N_P),X(2,N_P),'r');  % 负类标点

% 计算超平面
[hyperplane(1), iteration_n(1)] = perceptron(X,Y,Eta,ori_w,ori_b);

fprintf("例%d迭代次数；%d\n",eg_n,iteration_n(1));

% 绘制超平面
para = symvar(hyperplane(1));  % 得到超平面参数
F = subs(hyperplane(1),para(1),0:4);  % 带入样本特征值1
for i = 1:5
    F(i) = solve(F(i),para(2));    % 计算样本特征值2
end

plot(0:4,F,'k')  % 绘制超平面
title('二分类线性模型');  xlabel('特征值1');  ylabel('特征值2')
axis([0 4 0 4]); legend('正分类点','负分类点','超平面')
%% eg.2====================================================================
eg_n = eg_n+1;
% 样本初始化
m = 30;     % 样本个数
Eta = 1;    % 学习率
char_n = 2; % 样本特征个数；
ori_b = 0;  % 偏置初值
ori_w = zeros(char_n,1);  % 权向量初值
%=================================================
% 产生随机样本
X = randi(m,char_n,m);  % 生成样本特征向量
% X(1,:)表示横坐标，X(2,:)表示纵坐标
Y = zeros(m,1);  % 初始化类别矩阵
% 位于 y = -x+m 直线上半部的点为 +1类别；直线下半部的点为 -1类别。
for i =1:m  % 设置点类别
    if X(1,i)+X(2,i) >= 30
        Y(i) = 1;
    else
        Y(i) = -1;
    end
end

P_P = find(Y == +1);  % 获得正类别点的坐标
N_P = find(Y == -1);  % 获得负类别点的坐标
% 样本图像绘制
figure(Name="eg.2样本点图")
scatter(X(1,P_P),X(2,P_P),'g');  % 正类标点
hold on;
scatter(X(1,N_P),X(2,N_P),'r');  % 负类标点

% 计算超平面
[hyperplane(2), iteration_n(2)] = perceptron(X,Y,Eta,ori_w,ori_b);

fprintf("例%d迭代次数；%d\n",eg_n,iteration_n(2));

% 绘制超平面
para = symvar(hyperplane(2));  % 得到超平面参数
F = subs(hyperplane(2),para(1),0:m);  % 带入样本特征值1
for i = 1:m+1
    F(i) = solve(F(i)==0,para(2));    % 计算样本特征值2
end

plot(0:m,F,'k')  % 绘制超平面
title('二分类线性模型');  xlabel('特征值1');  ylabel('特征值2')
axis([0 m 0 m]); legend('正分类点','负分类点','超平面')








