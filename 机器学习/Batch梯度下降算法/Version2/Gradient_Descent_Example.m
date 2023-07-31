close all; clear; clc;
% 梯度下降法
% Gradient_Descent_algorithm 函数实例
% 
% 修改日期：2023.2.2
%==============================================================Author：Chen
syms theta0 theta1 theta2  theta3 theta4 X0;
% 参数初始化
m = 40;  % 样本个数
Epsilion = 1e-6;  % 误差精度
Eta = 0.0035;     % 学习率
% 目标方程
F = theta0 + theta1*X0;
% 样本初始化
X = 1:m;  Y = sort(rand(1,m));

figure(NumberTitle="off",Name="样本")
scatter(X,Y); hold on;
title("随机样本")

F = subs(F,X0,X);   % 样本特征带入目标函数

% 批量梯度下降
Cost_F = (1/2/m)*sum((F - Y).^2);   % 代价函数

F_Sym = symvar(Cost_F);    % 获得代价函数的参数
[~, temp_n] = size(F_Sym); % 代价函数参数个数
Theta_ori = zeros(1,temp_n);   % 参数初始值

[temp_theta, N_cyc] = Gradient_Descent_algorithm(Cost_F, Eta, Epsilion, Theta_ori);

% 函数验证
F = subs(F,F_Sym,temp_theta);
fprintf('循环次数：%d\n',N_cyc);
plot(X,F)  % 绘制目标函数
legend('样本点','线性回归直线')



