close all; clear; clc;
% 生成树实例
% 
% 参考：https://blog.csdn.net/john_xia/article/details/107563005（存在错误）
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

t_point = [2 4.5]; n = 4;
% 查找Kd树
nn_p = Kd_tree_search_knn(Kd_tree, t_point, n);

%%====================================================================================================================================================
%% Kd树 方差/顺序递增法生成
function [tree] = Kd_tree_create(data,M,forment_n)
% 生成Kd树
% Input：
%   Data：训练数据集（每个样本为列向量）
%       注：Data训练数据集的最后一行需为数据 id 编号
%   M：构建规则（方差/特征序列递增构建；默认方差最大）
%       var：按方差最大的维度进行分割
%       increase：按特征值序号递增进行分割
% 
% Output：
%   tree：Kd树
% 
% 修改时间：2023.2.5
%==============================================================Author：Chen
%% 判断Kd树生成方式
if nargin==1 || M=="var"
    %% 按特征方差最大进行分割
    [dimension, num] = size(data);  % 获得训练数据集维度
    % 计算特征值的方差
    data_var = zeros(1,dimension-1);
    for i = 1: dimension-1
        data_var(i) = var(data(i,:));
    end
    [~, max_dim] = max(data_var);  % 获得训练数据集维度方差最大索引
    data = (sortrows(data', max_dim))';  % 以序号为 max_dim 维度进行排序

    present_dim = max_dim;  % 当前节点id值
    method_n = 'var';  % Kd树生成方式

    forment_n = false;
elseif (nargin==2 && M == "increase") || (nargin==3 && M == "increase")
    %% 按特征序号递增进行分割
    if nargin==2 && M == "increase"  % 用户不指定选择的初始维度时
        forment_n = 1;
    end
    [dimension, num] = size(data);  % 获得训练数据集维度
    n_dim = forment_n;
    
    data = (sortrows(data', n_dim))';  % 以序号为 max_dim 维度进行排序

    present_dim = forment_n;  % 当前节点id值
    method_n = 'increase';  % Kd树生成方式

    % 更新排序维度序号
    if forment_n+1 > dimension-1
        forment_n = 1;
    else
        forment_n = forment_n+1;
    end
else
    %% 参数输入错误
    disp('函数输入错误，请检查输入参数。')
    tree = false;
end
%% Kd tree
% 获得当前节点信息（num若为偶数，选择中位数右侧点）
tree.id = data(dimension,round((num+1)/2));  % 当前节点 id
tree.node = data(1:dimension-1,round((num+1)/2));  % 当前节点特征向量
tree.dim = present_dim;  % 排序时所使用的维度序号
% 初始化分支
tree.parent = [];  % 当前节点的父节点
tree.left  = [];   % 当前节点的做分支序号
tree.right = [];   % 当前节点的右分支序号

% 递归生成左右子树
% 生成当前节点左子树
Ltree = [];
if round((num+1)/2) > 1  % 存在左子树
    leftdata = data(:,1:round((num+1)/2)-1);  % 得到小于当前特征值的训练数据集
    Ltree = Kd_tree_create(leftdata,method_n,forment_n);  % 递归得到下一节点Kd树
    for i = 1:size(Ltree, 1)
        % 判断当前节点是否为递归得到Kd树的父节点
        if isempty(Ltree(i).parent)
            Ltree(i).parent = tree.id;
            tree.left = Ltree(i).id;
        end
    end
end

% 生成当前节点右子树
Rtree = [];
if round((num+1)/2) < num  % 存在右子树
    rightdata = data(:,round((num+1)/2)+1:end);  % 得到大于当前特征值的训练数据集
    Rtree = Kd_tree_create(rightdata,method_n,forment_n);  % 递归得到下一节点Kd树
    for i = 1:size(Rtree, 1)
        % 判断当前节点是否为递归得到Kd树的父节点
        if isempty(Rtree(i).parent)
            Rtree(i).parent = tree.id;
            tree.right = Rtree(i).id;
        end
    end
end

% 保存当前节点的左右分支
tree = [tree; Ltree];
tree = [tree; Rtree];
end
%%====================================================================================================================================================
%% Kd树 KNN算法查询
function [nn_p] = Kd_tree_search_knn(Kd_tree, t_point, n)
% Kd树查询（KNN算法--k-Nearest Neighbor）
% Input：
%   Kd_tree：Kd树
%       注：Kd树node值为行列皆可
%   t_point：目标点
%       注：t_point 值为行列皆可
%   n：查找最近点个数
% 
% Output：
%   nn_p：最近邻点集合
%       注：按行排列。与目标点由近到远排列
% 
% 修改时间：2023.2.6
%==============================================================Author：Chen
% 寻找最靠近目标点的节点
closest = Kd_tree(1);  % 获得根节点
% 将closest.node重构为行向量
closest.node = reshape(closest.node,1,length(closest.node));
% 将t_point重构为行向量
t_point = reshape(t_point,1,length(t_point));

%% 寻找目标点所在的叶节点超矩形区域
while(1)
    if closest.node(closest.dim) >= t_point(closest.dim) && ~isempty(closest.left)
        % 目标点位于当前节点左侧且当前节点左侧分支非空
        closest = Kd_tree([Kd_tree.id]==closest.left);
        closest.node = reshape(closest.node,1,length(closest.node));  % 将closest.node重构为行向量
    elseif closest.node(closest.dim) <= t_point(closest.dim) && ~isempty(closest.right)
        % 目标点位于当前节点右侧且当前节点右侧分支非空
        closest = Kd_tree([Kd_tree.id]==closest.right);
        closest.node = reshape(closest.node,1,length(closest.node));  % 将closest.node重构为行向量
    else
        break;
    end
end
%% 计算目标点与当前最近叶节点距离
Kd_tree([Kd_tree.id]==closest.id).done = 1;
nn_p = closest.node;
[max_dis, max_idx] = max(sum((nn_p - t_point).^2, 2));
max_dis = max_dis(1);
max_idx = max_idx(1);

%% 从当前节点向上回溯
node_now = closest;
while(1)
    % 回溯到根节点就break
    if find([Kd_tree.id]==node_now.id) == 1
        break
    end
    % 回溯到父节点，如果父节点的点符合要求就添加进去
    node_now = Kd_tree([Kd_tree.id]==node_now.parent);
    node_now.node = reshape(node_now.node,1,length(node_now.node));  % node_now.node重构为行向量
    Kd_tree([Kd_tree.id]==node_now.id).done = 1;
    if size(nn_p, 1) < n  % 最近邻点 nn_p 个数<所需点
        nn_p(end+1, :) = node_now.node;  % 将当前节点保存在 nn_p 内
    elseif sum((node_now.node - t_point).^2) < max_dis
        nn_p(max_idx, :) = node_now.node;
        % 计算 nn_p 与目标点的欧几里得距离
        [max_dis, max_idx] = max(sum((nn_p - t_point).^2, 2));  % 按行求和得到当前最大欧几里得距离
        max_dis = max_dis(1);  % 得到最大距离 
        max_idx = max_idx(1);  % 最大距离对应的 nn_p 点序号
    end
    
    
    closest_temp = [];
    % 检查目标点 t_point 到当前父节点的欧几里得距离，如果距离小于当前最大距离，则可能在另一侧有符合要求的点
    if sum((t_point - node_now.node).^2) < max_dis
        if ~isempty(node_now.left)  % 当前父节点左分支非空时
            node_temp = Kd_tree([Kd_tree.id]==node_now.left);  % 获得当前父节点左分支叶节点
            if isempty(node_temp.done)  % 检查节点是否判断过
                % 把左子节点调到第一行，作为根节点，递归调用
                Kd_tree_temp = Kd_tree;
                Kd_tree_temp(1) = node_temp;
                Kd_tree_temp([Kd_tree.id]==node_temp.id) = Kd_tree(1);
                % 得到的最近点按行排列
                closest_temp = [closest_temp; Kd_tree_search_knn(Kd_tree_temp, t_point, n)];
            end
        end
        if ~isempty(node_now.right)  % 当前父节点右分支非空时
            node_temp = Kd_tree([Kd_tree.id]==node_now.right);  % 获得当前父节点右分支叶节点
            if isempty(node_temp.done)  % 检查节点是否判断过
                % 把右子节点调到第一行，作为根节点，递归调用
                Kd_tree_temp = Kd_tree;
                Kd_tree_temp(1) = node_temp;
                Kd_tree_temp([Kd_tree.id]==node_temp.id) = Kd_tree(1);
                % 得到的最近点按行排列
                closest_temp = [closest_temp; Kd_tree_search_knn(Kd_tree_temp, t_point, n)];
            end
        end
    end
    
    if ~isempty(closest_temp)  % 存在分支节点
        closest_temp_dis = sum((closest_temp - t_point).^2, 2);  % 计算分支节点与目标点的欧几里得距离
        % 比较现有的nn_p集
        for i = 1: size(closest_temp_dis, 1)
            if closest_temp_dis(i) < max_dis  % 点closest_temp距离目标点更近
                % 检查nn_p空间是否已满
                if size(nn_p, 1) < n  % nn_p 还存在剩余空间
                    nn_p(end+1, :) = closest_temp;  % 将当前节点保存在 nn_p 内
                else  % 将离目标点最远的叶节点替换为 closest_temp
                    nn_p(max_idx, :) = closest_temp(i, :);
                    % 计算nn_p集更新后的最大距离点
                    [max_dis, max_idx] = max(sum((nn_p - t_point).^2, 2));
                    max_dis = max_dis(1);
                    max_idx = max_idx(1);
                end
            end
        end
    end
end
%% 对集合  nn_p 进行排序
if size(nn_p,1)>1  % nn_p集合中至少含有两个点时
    % 冒泡排序
    for i = 1:size(nn_p,1)-1
        for j = 1:size(nn_p,1)-1
            % 计算相邻两点到目标点的欧氏距离
            temp_dist1 = sum((nn_p(j  ,:) - t_point).^2, 2);
            temp_dist2 = sum((nn_p(j+1,:) - t_point).^2, 2);
            if temp_dist1 > temp_dist2
                temp_var = nn_p(j,:);  nn_p(j,:) = nn_p(j+1,:);  nn_p(j+1,:) = temp_var;
            end
        end
    end
end
end

