function [Imf,Residual] = Own_EMD(x,M)
% Empiricial Mode Decomposition
% EMD分解
% 采用三次样条插值对包络进行拟合
% InPut：
%       x：待EMD分解的信号。(x为1×N维行向量)
%       M：需要的IMF个数(可省)
%       注：当给定M值时，输出为满足带分解信号不为单调函数前提下分解的EMD信号
% OutPut：
%       Imf：分解得到的IMF分量
%       Residual：残差
% 
% 
% 本EMD分解采用当待分解信号为单调函数时停止分解
% 修改日期：2022.11.28
%==============================================================Authon：Chen
%% EMD分解函数
N = length(x); % 得到待处理矩阵长度
imf = [];      % 初始化IMF矩阵

% 其他停止标准：
% 1、标准偏差(Standard Deviation ,SD)
% 2、S Number准则
% 3、阈值法等
switch nargin
    case 1
        % 当待分解的信号为单调函数时结束分解
        while ~if_monotonic(x)
            imf1 = Sub_EMD(x); % 得到x的一个IMF分量
            imf = [imf imf1];  % 保存分解得到的IMF分量
            x = x - imf1;      % 原始信号-IMFx，作为新的原始信号
        end
    case 2
        M = M+1;  % 添加残差数量
        % 满足信号为单调函数的前提下得到M个IMF分量
        while M && ~if_monotonic(x)
            imf1 = Sub_EMD(x); % 得到x的一个IMF分量
            imf = [imf imf1];  % 保存分解得到的IMF分量
            x = x - imf1;      % 原始信号-IMFx，作为新的原始信号
            M = M-1;
        end
end
% 对IMF进行重构，得到IMF矩阵
imf = reshape(imf,N,length(imf)/N);
[~,imf_R] = size(imf);
Imf = imf(:,1:imf_R-1);  % 得到 IMF分量
Residual = imf(:,imf_R); % 得到分解后残差
%% 子函数声明
%==========================================================================
% 得到x的一个IMF分量
function imf = Sub_EMD(x)
N = length(x);  % 得到待处理矩阵长度
JudCriteria = 0.05;  % 定义判断标志
dispose_signal = x;
while true
    %======================================================== 步骤1
    max_peak_n = Find_Peak(dispose_signal);   % 获得x极大值坐标
    min_peak_n = Find_Peak(-dispose_signal);  % 获得x极小值坐标
    max_peak = dispose_signal(max_peak_n); % 获得x极大值幅值
    min_peak = dispose_signal(min_peak_n); % 获得x极小值幅值
    % 根据原信号极值点绘制上下包络线
    max_envelope = CubicSplineInterpolation(max_peak_n,max_peak,1:N);
    min_envelope = CubicSplineInterpolation(min_peak_n,min_peak,1:N);
    %======================================================== 步骤2
    % 计算均值包络线
    mean_envelope = (max_envelope+min_envelope)/2;
    %======================================================== 步骤3
    % 原信号-均值包络线--得到中间信号
    middle_signal = dispose_signal - mean_envelope;
    %======================================================== 步骤4
    % 判断 middle_signal 是否为IMF
    max_mid_peak_n = Find_Peak(middle_signal);   % 获得中间信号极大值坐标
    min_mid_peak_n = Find_Peak(-middle_signal);  % 获得中间信号极小值坐标
    max_mid_peak = middle_signal(max_mid_peak_n);  % 获得中间信号极大值幅值
    min_mid_peak = middle_signal(min_mid_peak_n);  % 获得中间信号极小值幅值
    % 根据原信号极值点绘制上下包络线
    max_mid_envelope = CubicSplineInterpolation(max_mid_peak_n,max_mid_peak,1:N);
    min_mid_envelope = CubicSplineInterpolation(min_mid_peak_n,min_mid_peak,1:N);
    mean_mid_envelope = (max_mid_envelope+min_mid_envelope)/2; % 中间信号的均值包络
    % 计算上下包络的平均值
    mean_mid = sum(mean_mid_envelope)/N;
    
    % 判断中间信号的均值包络 是否满足IMF的两个条件
    if mean_mid <= JudCriteria && if_zero_peak_n(dispose_signal)
        break;% middle_signal 是IMF
    end
    % middle_signal不是IMF，以该信号为基础继续进行IMF分解
    dispose_signal = middle_signal;
end
% 保存每次分解且满足IMF条件的中间包络
imf = middle_signal;
%==========================================================================
% 判断函数是否单调
function d = if_monotonic(x)
u = length(Find_Peak(x))*length(Find_Peak(-x)); % 判断函数极大极小值点是否同时存在
if u > 0  % 同时存在极大与极小值
    d = 0;
else
    d = 1;
end
%==========================================================================
% 判断过零点个数与极值点个数是否相同
function d = if_zero_peak_n(x)
% 错位相乘判断过 0 点个数
zero_n = sum(x(1:end-1).*x(2:end) < 0) ;                % 过零点个数
peak_n = length(Find_Peak(x)) + length(Find_Peak(-x)); % 极值点个数
if abs(zero_n - peak_n)>2 % 极值点个数和过零点个数相差不大于1时满足 IFM 条件
    % 注：由于端点问题，这里由不小于1改为2（存在问题）
    d = 0; % 过零点数与极值点数相差＞1
else
    d = 1;
end
%==========================================================================
% 寻找极值点
function n = Find_Peak(x)
n = find(diff(diff(x)>0)<0);
n = n+1;
%==========================================================================
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
% 
% 详细参考 CubicSplineInterpolation_Example
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
%==========================================================================
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
%==========================================================================




