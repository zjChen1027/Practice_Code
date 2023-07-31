function [temp_theta, N_cyc] = Gradient_Descent_algorithm(Cost_F, Eta, Epsilion, Theta_ori)
% 梯度下降算法
%   InPut：
%       Cost_F：代价函数（损失函数）
%       Eta   ：学习率（步长）
%       Epsilion ：计算精度
%       Theta_ori：参数初值
%   OutPut：
%       temp_theta：参数值
%       N_cyc：循环次数
% 
%   Example：
%       syms theta0 theta1 X0;
%       m = 40;  Epsilion = 1e-6;  Eta = 0.0035;   
%       F = theta0 +theta1*X0;
%       X = 1:m;  Y = sort(rand(1,m));      % 初始化样本
%       F = subs(F,X0,X);          % 样本带入目标方程
%       Cost_F = (1/2/m)*sum((F - Y).^2);   % 代价函数
%       F_Sym = symvar(Cost_F);    % 获得代价函数的参数
%       Theta_ori = [0 0];         % 参数初始值
%       [temp_theta, N_cyc] = Gradient_Descent_algorithm(F_in, Eta, Epsilion, Theta_ori);
% 
% 修改时间：2023.2.1
%==============================================================Author：Chen
%% 预处理
F_Sym = symvar(Cost_F);    % 获得代价函数的参数
[~, temp_n] = size(F_Sym); % 代价函数参数个数
F_d = sym(zeros(1,temp_n));% 初始化梯度矩阵
for i = 1:temp_n
    F_d(i) = diff(Cost_F,F_Sym(i));
end
%% 梯度下降算法
N_cyc = 0;  % 定义循环次数
temp_theta = sym(zeros(1,temp_n));  % 初始化参数矩阵
while true
    for i = 1:temp_n  % 计算 Theta_i+1
        temp_theta(i) = Theta_ori(i) - Eta*subs(F_d(i),F_Sym,Theta_ori);
    end
    % 计算 F(Theta^(k+1)) - F(Theta^(k)) 的 Frobenius 范数
    % 两种精度计算方式
    nor = norm(subs(Cost_F,F_Sym,temp_theta) - subs(Cost_F,F_Sym,Theta_ori));
    % nor = norm(temp_theta - Theta_ori);
    N_cyc = N_cyc+1;   % 循环次数自增
    if nor < Epsilion  % 满足最小精度停止条件
        break;
    elseif N_cyc==30 &&  nor>1   % 学习率过大导致参数发散
        fprintf('学习率过大，参数发散\n请减小学习率')
        break;
    elseif N_cyc >= 150          % 学习率过低 or 误差精度过高导致参数收敛过慢
        fprintf('学习率过小/容许的误差精度过高\n请增大学习率或降低误差精度')
        break;
    end
    Theta_ori = temp_theta;  % 更新初始值
end



