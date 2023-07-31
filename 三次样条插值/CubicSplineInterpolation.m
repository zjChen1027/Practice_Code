function S = CubicSplineInterpolation(Original_Coord,Original_Amp,Desire_Coord)
% 函数功能：计算三次样条插值函数对应的函数值
%
% OutPut：
%   Si：表示输入坐标轴 Original_Coord 所对应的函数值
%
% InPut：
%   Original_Coord：样本点的坐标
%   Original_Amp  ：样本值
%   Desire_Coord  ：待求解插值函数后的坐标
%       注：输入都为行向量
%
% 采用 CubicSplineInterpolation_coefficient(X,Y)求解系数Mi
% 修改日期：2022.11.22
%==============================================================Author：Chen
% 求解三次样条插值系数 M、每段插值区间的长度 h、插上表 A。
[h,A,M] = CubicSplineInterpolation_coefficient(Original_Coord,Original_Amp);
% 计算原信号样本点数 与 三次样条插值后样本点数
n = length(Original_Coord); m = length(Desire_Coord);
% 初始化插值后矩阵
S = zeros(1,m);
% 计算Si（i = 1:m）
for i = 1:m
    for j = 1:n-1
        % 计算位于第 j 段内的插值点 Si 的函数值
        if (Desire_Coord(i)<=Original_Coord(j+1)) && (Desire_Coord(i)>=Original_Coord(j))
            % 分别计算 Si 的四个参数
            P1 = M(j,1)*(Original_Coord(j+1)-Desire_Coord(i))^3/(6*h(j));
            P2 = M(j+1,1)*(Desire_Coord(i)-Original_Coord(j))^3/(6*h(j));
            P3 = (A(j,1)-M(j,1)/6*(h(j))^2)*(Original_Coord(j+1)-Desire_Coord(i))/h(j);
            P4 = (A(j+1,1)-M(j+1,1)/6*(h(j))^2)*(Desire_Coord(i)-Original_Coord(j))/h(j);
            S(i) = P1+P2+P3+P4; % 得到位于 j 段内的第 i 个参数幅值
            break;
        else
            S(i)=0; % 不位于原样本区间的点数，设置为0
       end
    end
end
end


function [h,A,M] = CubicSplineInterpolation_coefficient(X,Y)
% 函数功能：计算三次样条插值系数 M
% 自然边界条件的三次样条函数(第二种边界条件--两端点的二阶导数值相等且为0)
%
% OutPut：
%   D：系数矩阵
%   h：插值宽度(相邻两个样本点之间的距离)
%       注：length(h)表示区间段的数量
%   A：差商表
%       注：A(:,1) 表示Y'；A(:,2)表示相邻点之间的一阶差商；A(:,3)表示相邻三点之间的二阶差商；
%          以此类推。
%   M：三次样条插值系数
%
% InPut：
%   X：样本点的坐标
%   Y：样本值
%       注：输入都为 1×n 维行向量
% 
% 扩展知识；自变量之差于应变量之差之比称为差商，一介差商也称做平均变化率。
% 
%==============================================================Author：Chen
N = length(X); % 样本点的个数
% 矩阵初始化
A = zeros(N,N); h = zeros(1,N-1); u = zeros(1,N-2);
D = zeros(N-2,N-2); g = zeros(N-2,1);
A(:,1) = Y';
% 计算差商表A
for  j = 2:N
    for i = j:N
        A(i,j)=(A(i,j-1)- A(i-1,j-1))/(X(i)-X(i-j+1));
    end
end
% 计算相邻样本点的距离表h
for i = 1:N-1
    h(i) = X(i+1) - X(i);
end

for i = 1:N-2
    D(i,i) = 2;
    g(i,1) = (6/(h(i+1)+h(i)))*(A(i+2,2)-A(i+1,2));
end

for i = 2:N-2
    u(i) = h(i)/(h(i)+h(i+1));
    N(i-1) = h(i)/(h(i-1)+h(i));
    D(i-1,i) = N(i-1);
    D(i,i-1) = u(i);             
end
% 计算 inv(D)*g
M = D\g;     % 求解系数 M
M = [0;M;0]; % 边界系数置零，使其满足自然边界条件
end



