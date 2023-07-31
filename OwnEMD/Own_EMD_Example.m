close all; clear; clc;
% Own_EMD函数参考实例
% 
% 
% 
% 修改时间：2022.11.28
%===============================================================Author:Chen
%% 参数定义
fc = 10; N = 1024; T = 2; fs = N/T;
t = linspace(0,T,N); f = (-N/2:N/2-1)*fs/N;
% 参考信号
x = 2*cos(2*pi*fc*t) + 0.2*cos(2*pi*2.5*fc*t) + 0.5*cos(2*pi*5*fc*t);

% 图像绘制
figure(NumberTitle="off",Name="emd函数实例")
plot(t,x,'k'); hold on;
%% Own_EMD
% 假设分解6个IMF信号
dec_n = 6;
[Imf, Residual] = Own_EMD(x,dec_n);
figure(NumberTitle="off",Name="Own_EMD函数分解")
for i= 1:dec_n
    subplot(dec_n,2,i*2-1)
    plot(t,Imf(:,i))
    subplot(dec_n,2,i*2)
    plot(f,fftshift(2*abs(fft(Imf(:,i))/N)))
end

% MATLAB 自带EMD
[imf,residual] = emd(x);
[imf_l, imf_r] = size(imf);
figure(NumberTitle="off",Name="Matlab EMD函数分解")
for i = 1:imf_r
    subplot(imf_r,2,i*2-1)
    plot(t,imf(:,i))
    subplot(imf_r,2,i*2)
    plot(f,fftshift(2*abs(fft(imf(:,i))/N)))
end
% 结论：Matlab 自带EMD端点效应小，Own_EMD端点效应严重，随着分解层数的增加，信号误差越来越大
% MATLAB 自带的EMD更好！！

