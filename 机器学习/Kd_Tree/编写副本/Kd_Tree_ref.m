clear;
close all;
clc;
% 生成数据
data = [2 3;
    5 4;
    9 6;
    4 7;
    8 1;
    7 2];
% 给数据标号
for i = 1: size(data,1)
    data(i,3) = i;
end
% 建立Kd树
Kd_tree = Kd_tree_create(data);
% 利用Kd树进行kNN查询
closest = Kd_tree_search_knn(Kd_tree, [2 4.5], 5);

%% 使用data建立Kd树
function [tree] = Kd_tree_create(data)
% 生成Kd树，每次分割以方差最大的维度进行分割
[num, dimension] = size(data);
dimension = dimension - 1;
for i = 1: dimension
    data_var(i) = var(data(:,i));
end
[~, choose_dim] = max(data_var);
data = sortrows(data, choose_dim);
tree.id = data(round(num/2),end);
tree.node = data(round(num/2),1:end-1);
tree.dim = choose_dim;
tree.parent = [];
tree.left = [];
tree.right = [];

% 递归生成左右子树
lefttree = [];
righttree = [];
if round(num/2) > 1
    leftdata = data(1:(round(num/2)-1), :);
    lefttree = Kd_tree_create(leftdata);
    for i = 1: size(lefttree, 1)
        if isempty(lefttree(i).parent)
            lefttree(i).parent = tree.id;
            tree.left = lefttree(i).id;
        end
    end
end
if round(num/2) < num
    rightdata = data((round(num/2)+1):end, :);
    righttree = Kd_tree_create(rightdata);
    for i = 1: size(righttree, 1)
        if isempty(righttree(i).parent)
            righttree(i).parent = tree.id;
            tree.right = righttree(i).id;
        end
    end
end
tree = [tree; lefttree];
tree = [tree; righttree];
end


%% 利用Kd树进行kNN查询
function [closest_point] = Kd_tree_search_knn(Kd_tree, data, n)
% 从根节点开始一直查询到叶节点，找到和data在一个区域的叶节点
closest = Kd_tree(1);
while(1)
    if closest.node(closest.dim) >= data(closest.dim) && ~isempty(closest.left)
        closest = Kd_tree(find([Kd_tree.id]==closest.left));
    elseif closest.node(closest.dim) <= data(closest.dim) && ~isempty(closest.right)
        closest = Kd_tree(find([Kd_tree.id]==closest.right));
    else
        break
    end
end

Kd_tree(find([Kd_tree.id]==closest.id)).done = 1;
closest_point = closest.node;
[max_dis, max_idx] = max(sum((closest_point - data).^2, 2));
max_dis = max_dis(1);
max_idx = max_idx(1);

% 从当前节点向上回溯
node_now = closest;
while(1)
    % 回溯到根节点就break
    if find([Kd_tree.id]==node_now.id) == 1
        break
    end
    
    % 回溯到父节点，如果父节点的点符合要求就添加进去
    node_now = Kd_tree(find([Kd_tree.id]==node_now.parent));
    Kd_tree(find([Kd_tree.id]==node_now.id)).done = 1;
    if size(closest_point, 1) < n
        closest_point(end+1, :) = node_now.node;
    elseif sum((node_now.node-data).^2) < max_dis
        closest_point(max_idx, :) = node_now.node;
        [max_dis, max_idx] = max(sum((closest_point - data).^2, 2));
        max_dis = max_dis(1);
        max_idx = max_idx(1);
    end
    
    % 检查data点到父节点的分割线的距离，如果距离小于当前最大距离，则可能在另一侧有符合要求的点
    closest_temp = [];
    if (data(node_now.dim)-node_now.node(node_now.dim))^2 < max_dis
        if ~isempty(node_now.left)
            node_temp = Kd_tree(find([Kd_tree.id]==node_now.left));
            if isempty(node_temp.done)
                % 把左子节点调到第一行，作为根节点，递归调用
                Kd_tree_temp = Kd_tree;
                Kd_tree_temp(1) = node_temp;
                Kd_tree_temp(find([Kd_tree.id]==node_temp.id)) = Kd_tree(1);
                closest_temp = [closest_temp; Kd_tree_search_knn(Kd_tree_temp, data, n)];
            end
        end
        if ~isempty(node_now.right)
            node_temp = Kd_tree(find([Kd_tree.id]==node_now.right));
            if isempty(node_temp.done)
                % 把右子节点调到第一行，作为根节点，递归调用
                Kd_tree_temp = Kd_tree;
                Kd_tree_temp(1) = node_temp;
                Kd_tree_temp(find([Kd_tree.id]==node_temp.id)) = Kd_tree(1);
                closest_temp = [closest_temp; Kd_tree_search_knn(Kd_tree_temp, data, n)];
            end
        end
    end
    
    if ~isempty(closest_temp)
        closest_temp_dis = sum((closest_temp - data).^2, 2);
        for i = 1: size(closest_temp_dis, 1)
            if closest_temp_dis(i) < max_dis
                closest_point(max_idx, :) = closest_temp(i, :);
                [max_dis, max_idx] = max(sum((closest_point - data).^2, 2));
                max_dis = max_dis(1);
                max_idx = max_idx(1);
            end
        end
    end
end
end
