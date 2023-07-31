clear; close all; clc;
% 三次样条插值函数实例：
% 实例内容：生成一个 T 秒离包含 N 点的散数据，幅值 0-1 随机分布的序列
% 对序列采用三次样条插值进行 10*N 点插值拟合。
% 
% 实例内函数声明：
% CubicSplineInterpolation_coefficient 求解插值系数 M
% CubicSplineInterpolation 求解插值后的信号
% 
% 修改日期：2022.11.22
%==============================================================Author：Chen
% 参数初始化
N = 50; T = 20;
t = linspace(0,T,N);% 原始坐标区间
y = rand(1,N);      % 待插值信号

% 原图像绘制
figure(NumberTitle="off",Name="待插值信号绘制")
plot(t,y)
xlabel('时间 S'); ylabel('幅值'); title('未三次样条插值前');

% 插值后的坐标轴
n = linspace(0,T,10*N); % 在原 N-1 段内每段增加 10N/(N-1) 个数据拟合图像
% 采用三次样条插值
S = CubicSplineInterpolation(t,y,n);
% 图像绘制
figure(NumberTitle="off",Name="原信号三次样条插值后图像")
stem(t,y,'r'); hold on;% 绘制原图像做对比
plot(n,S,'k');
xlabel('时间 S'); ylabel('幅值'); title('采用三次样条插值后');
legend('原信号','三次样条插值拟合信号');


