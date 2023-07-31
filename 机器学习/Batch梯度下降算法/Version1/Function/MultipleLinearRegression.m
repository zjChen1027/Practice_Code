function [F_out, symF_out] = MultipleLinearRegression(X,Y,k,F_in,Th_Orig,alpha)
% 多变量线性回归模型
% Batch梯度下降算法
% 
%   输入：
%       X：样本特征值（横表示某一样本，纵表示某一特征值 （n*m维数组））
%           eg：X = [1 10; 2 20; 3 30];                  % 三个样本，每个样本有两个特征
%       Y：样本的观测值（n*1维数组）            n：样本个数；m：样本特征值个数
%       k：迭代次数
%       F_in：线性回归假设方程
%           eg：F_in = Theta0 + Theta1*X1 + Theta2*X2;   % 三参数，两变量预测方程
%       Th_Orig：参数的初始值
%       alpha：学习率
%   输出：
%       F_out：确定参数后的预测方程
%       symF_out：确定参数后的预测方程的变量
% 
% 
% 
% 修改日期：2023.1.30
%===============================================================Author:Chen
%% 参数初始化
[X_Sn, X_Cn] = size(X);   % 得到样本个数与特征个数
symF = symvar(F_in);      % 得到线性回归函数中的符号变量
[~,X_Pn] = size(symF);
X_Pn = X_Pn - X_Cn;       % 线性回归模型参数个数
% symF 前 X_Pn 个为参数，后 X_Cn 为变量(特征)
%% 特征缩放
% 存在错误，问题未解决
% 对样本的特征值进行均值归一化并进行特征值缩放
% for i = 1:X_Sn
%     for j = 1:X_Cn
%         X(i,j) = (X(i,j) - mean(X(:,j)))/max(abs(X(:,j)));
%     end
%     Y(i) = (Y(i) - mean(Y))/max(abs(Y));
% end
%% 梯度下降算法前预处理
h_Theta = sym(zeros(X_Sn,1));   % 初始化预测函数矩阵
for i = 1:X_Sn
    for j = 1:X_Cn   % 带入样本数据
        h_Theta(i) = h_Theta(i) + subs(F_in, symF(X_Pn+j), X(i,j));
    end
    h_Theta(i) = h_Theta(i) - (X_Cn-1)*F_in; % 减去每次下循环多乘的 X_Cn-1 个 F
end

% 代价函数
J = (1/2/X_Sn)*sum((h_Theta-Y).^2);

% 代价函数对 X_Pn 个参数的偏导
Theta_d = sym(zeros(1,X_Pn));
for i = 1:X_Pn
    Theta_d(i) = diff(J,symF(i));
end
%% 梯度下降算法
for i = 1:k
    % 改变初始点的参数值
    Th_Orig = Th_Orig - alpha*subs(Theta_d,symF(1:X_Pn),Th_Orig);
end
% 迭代后的 Theta值带入假设方程
F_out = subs(F_in,symF(1:X_Pn),Th_Orig);

% 获得假设方程的变量
symF_out = symvar(F_out);


