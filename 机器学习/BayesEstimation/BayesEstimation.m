function y = BayesEstimation(X,Y,x,lambda)
% 贝叶斯估计函数
% 
% 输入：
%   X：训练数据集
%       注：训练数据集样本为列向量
%   Y：类别
%   x：实例
%   lambda：贝叶斯参数
%       注: lambda=0 => 极大似然估计
%           lambda=1 => 拉普拉斯平滑
% 
% 输出：
%   y：估计类别
%       注：当存在多个预测类别时，按行输出可能的类别值
% 
% 注：lambda=0;时，会出现估计的概率值为0的问题，影响到后验概率的计算。
% 
% 修改日期：2023.2.8
%==============================================================Author：chen
Y = reshape(Y,length(Y),1);  % Y 重构为列向量
[char_n, n] = size(X);  % 获得训练集样本特征个数与样本数
% 特征向量处理
Sj = zeros(1,char_n);
for i = 1:char_n
   temp = tabulate(X(i,:)');  % 统计每个特征的取值以及取值个数
   Sj(i) = length(temp(:,1));  % 保存每个特征取值的个数
end

% 类向量处理
C = tabulate(Y);  % 统计类的取值以及取值个数
% 获得类值
if iscell(C)  % 判断C是否为元组
    c_name = cell2mat(C(:,1));
else
    c_name = C(:,1);
end
[C_n, ~] = size(C);  % 类取值个数

%% 贝叶斯估计
% lambda = 0：极大似然估计；lambda = 1：拉普拉斯平滑。
% char_n：样本特征的个数；Sj：每个特征的取值个数
Py = zeros(1,C_n);
for i = 1:C_n  % 分别计算每个类别的朴素贝叶斯概率
    P_cond = 1;
    crood_ck = find(Y == c_name(i));  % 获得Ck类的坐标序号
    % 先验概率的贝叶斯估计
    P_Ck = (length(crood_ck)+lambda)/(n+C_n*lambda);
    % 条件概率的贝叶斯估计
    for j = 1:char_n
        P_cond = (length(find(X(j,crood_ck) == x(j))) + lambda)/(length(crood_ck)+Sj(j)*lambda)*P_cond;
    end
    Py(i) = P_Ck*P_cond;  % 计算类别C_n的朴素贝叶斯概率
end

[~, max_Py] = max(Py);
max_Py = Py == Py(max_Py);  % 检查 Py(max_Py) 是否是唯一的最大值
y = c_name(max_Py);  % 得到实例所属类别
end

