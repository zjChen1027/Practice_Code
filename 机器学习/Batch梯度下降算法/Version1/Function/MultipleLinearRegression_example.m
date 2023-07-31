close all; clear; clc;
format short
% Batch梯度下降算法
% MultipleLinearRegression()函数使用例程
% 
% 修改日期：2023.1.30
%===============================================================Author:Chen

% 样本生成（房屋假设模型）
m = 3; n = 40;   % 40个样本，每个样本包含3个特征
X = zeros(n,m);
X(:,1) = randi([80 130],n,1);   % 房屋面积
X(:,2) = randi([3 5],n,1);      % 房间数
X(:,3) = randi([1 32],n,1);     % 楼层
Y = X(:,1).*X(:,2).*(abs(X(:,3)-16)+1)+3000;   % 价格
Z = [X,Y];

Th_Orig = zeros(1,m+1);   % 参数初始值
alpha = 0.00011;          % 学习率
k = 50;   % 迭代次数

% 线性回归假设方程
syms Theta0;
F_in = Theta0;
for i = 1:m
     eval(['syms' ' ' 'X' num2str(i) ';']);
     eval(['syms' ' ' 'Theta' num2str(i) ';']);
     F_in = F_in + eval(['X' num2str(i) '*' 'Theta' num2str(i) ';']);
end

% Batch 梯度下降算法
[F_out, symF_out] = MultipleLinearRegression(X,Y,k,F_in,Th_Orig,alpha);

% 验证
X_Ver = [120 5 32];
disp(double(subs(F_out,symF_out,X_Ver)))
%%
% J(theta)曲线绘制 学习率固定为 alpha
point_n = 15;
y_plot = zeros(1,point_n);
% J(theta) 图
for i = 1:point_n
    sum_temp = 0;
    [F_out, symF_out] = MultipleLinearRegression(X,Y,i*2,F_in,Th_Orig,alpha);
    for j = 1:n
        sum_temp = sum_temp + (subs(F_out,symF_out,X(j,:)) - Y(j)).^2;
    end
    y_plot(i) = (1/2/n)*sum_temp;
end

title_name = sprintf('学习为：%d J(θ)图',alpha);
figure(NumberTitle="off",Name='学习率J(θ)图')
plot(2*(1:point_n),y_plot);













