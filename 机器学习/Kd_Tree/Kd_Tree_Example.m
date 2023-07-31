close all; clear; clc;
% 生成树实例
% 
% 
% 修改时间：2023.2.5
%===============================================================Author:Chen
%% 生成样本

% eg.1
data = [2 5 9 4 8 7; 3 4 6 7 1 2];  % 书P55例题
data(3,:) = 1:size(data,2);  % 为样本添加编号

% eg.2
% data = [3 5 1 3 2 8; 2 6 4 7 9 3];
% data(3,:) = 1:size(data,2);  % 为样本添加编号

% 生成Kd树
Kd_tree = Kd_tree_create(data,"var");

t_point = [2 4.5]; n = 3;
% 查找Kd树
nn_p = Kd_tree_search_knn(Kd_tree, t_point, n);