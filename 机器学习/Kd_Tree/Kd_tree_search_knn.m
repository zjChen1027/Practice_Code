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

