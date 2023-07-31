close all; clear; clc;
% 感知机函数编写前框架
% 
% 修改时间：2023.2.3
%==============================================================Author：Chen
%% 样本初始化
m = 30;  % 样本个数
X = randi(m,m,2);  % 生成样本点横纵坐标，X(:,1)表示横坐标，X(:,2)表示纵坐标
Y = zeros(1,m);    % 初始化类别矩阵
% 位于 y = -x+m 直线上半部的点为 +1类别；直线下半部的点为 -1类别。
for i =1:m  % 设置点类别
    if X(i,1)+X(i,2) >=30
        Y(i) = 1;
    else
        Y(i) = -1;
    end
end
P_P = find(Y == +1);  % 获得正类别点的坐标
N_P = find(Y == -1);  % 获得负类别点的坐标
% 样本图像绘制
figure(Name="样本点图")
scatter(X(P_P,1),X(P_P,2),'g');  % 正类标点
hold on;
scatter(X(N_P,1),X(N_P,2),'r');  % 负类标点
% plot(1:m,-(1:m)+m,'k')  % y = -x+m
%% 书 P40 页例题
% m = 3;
% X = [3,3; 4,3; 1,1];
% Y = [1 1 -1];
% figure(Name="样本点图")
% scatter(X(:,1),X(:,2));  hold on;
%% 随机梯度下降法
ori_w = 0;  ori_b = 0;
Eta = 1;  % 学习率

syms b w;  % 定义参数
k = 0;     % 定义迭代次数

w = ori_w;  b = ori_b;  % 带入初值
flag = true;  % 循环退出标志
while flag
    for i = 1:m
        if Y(i)*(sum(w.*X(i,:)) + b) <= 0  % 当前点为误分类点
            % 更新参数值
            w = w + Eta*Y(i)*X(i,:);
            b = b + Eta* Y(i);
            k = k + 1;  % 记录迭代次数
            break;
        elseif i == m && Y(i)*(sum(w.*X(i,:)) + b) > 0
            % 所有点均被正确分类
            flag = false;  % 退出循环
        end
    end
end
%% 函数验证
syms x y;
y = -(w(1)*x + b)/w(2);

fprintf('迭代次数%d\n',k)
plot(0:m,subs(y,x,0:m))
legend('正样本点','负样本点','分类超平面')


