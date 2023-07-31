function [hyperplane, iteration_n]= perceptron(X,Y,Eta,ori_w,ori_b)
% 基于随机梯度下降的感知机
%   输入：
%       X：实例特征向量（每个样本为列向量）
%       Y：类别
%       Eta：学习率
%       ori_w：权值向量
%       ori_b：偏置
% 
%   输出：
%       hyperplane ：求得的超平面
%       iteration_n: 迭代次数
% 
% Example：
% X = [3,3; 4,3; 1,1]';
% Y = [1 1 -1];
% Eta = 1;
% ori_w = [0;0]; ori_b = 0;
% [hyperplane, iteration_n]= perceptron(X,Y,Eta,ori_w,ori_b)
% 修改时间：2023.2.3
%==============================================================Author：Chen
syms b w;  % 定义参数
iteration_n = 0;     % 定义迭代次数

w = ori_w;  b = ori_b;  % 带入初值
m = length(Y);% 样本个数
flag = true;  % 循环退出标志
while flag
    for i = 1:m
        if Y(i)*(dot(w,X(:,i)) + b) <= 0  % 当前点为误分类点
            % 更新参数值
            w = w + Eta*Y(i)*X(:,i);
            b = b + Eta* Y(i);
            iteration_n = iteration_n + 1;  % 记录迭代次数
            break;
        elseif i == m
            % 所有点均被正确分类
            flag = false;  % 退出循环
        end
    end
end
para = sym('X',[1 length(w)]);  % 初始化样本特质变量
hyperplane = dot(w,para)+b;  % 得到超平面
end