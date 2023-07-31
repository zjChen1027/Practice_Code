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