close all; clear; clc;
% 离散小波实例
% 对含有高斯白噪声的信号进行 level尺度分解、重构、降噪，并计算信噪比比较了不同降噪模型软降噪的性能差异。
% 降噪模型：缺省阈值模型、Penalty模型
% 
%==============================================================author：Chen
%% 参考信号定义
% 参数定义==================================================================
T = 1; B = 100;
N = 2048; fs = N/T;
Num_plot = 1; % 图像编号
%==========================================================================
% 坐标轴初始化
t = linspace(0,T,N);
f = linspace(-fs/2,fs/2,N);

% 参考信号：chirp波
St = 2*exp(1i*pi*(B/T)*t.^2);  % St = 2*cos(2*pi*0.1*B*t) + 3*cos(2*pi*0.5*B*t);
% 参考信号图像绘制
Fig_name = sprintf("参考信号时频图  Num %d",Num_plot); Num_plot = Num_plot+1;
figure(NumberTitle="off",Name=Fig_name)
subplot(2,1,1)
plot(t,real(St)); xlabel('时间 s'); ylabel('幅度'); title('参考信号时域');
subplot(2,1,2)
plot(f,fftshift(abs(2*fft(St)/N))); xlabel('频率 Hz'); ylabel('幅度'); title('参考信号频域');
%% 添加噪声
gauss = 0.1*randn(1,N);
% 添加高斯白噪声
Yt = St + gauss;
% 含噪声信号绘制
Fig_name = sprintf("含噪声参考信号  Num %d",Num_plot); Num_plot = Num_plot+1;
figure(NumberTitle="off",Name=Fig_name)
subplot(2,1,1) % 含噪声参考信号时域
plot(t,real(Yt)); xlabel('时间 s'); ylabel('幅度'); title('含噪声参考信号时域');
subplot(2,1,2) % 含噪声参考信号频域
plot(f,fftshift(2*abs(fft(Yt))/N)); xlabel('频率 Hz'); ylabel('幅度'); title('含噪声参考信号频域');
%% 小波分解
% 参数定义==================================================================
level = 5; WaveName = 'db5';
% 注：dbN （Daubechies小波 N 表示阶数）db1等同于Haar小波
% 多级分解时需要注意下采样是否满足奈奎斯特的问题；
% Morlet 小波不能做离散小波变换和正交小波变换（不具备正交性，仅满足连续小波的允许条件）
%==========================================================================
[C, L] = wavedec(Yt, level, WaveName);
% 获得尺度 level 的细节系数
for i = 1:level
    eval(['cd' num2str(i) '= detcoef(C, L, i)', ';']);     % 命名为 cdx (x = 1:level)
end
% 获得尺度 level 的近似系数
eval(['ca' num2str(level) '= detcoef(C, L, level)', ';']); % 命名为 cax (x = level)

% 小波分解图像绘制
Fig_name = sprintf("小波分解时域图像绘制  Num %d",Num_plot); Num_plot = Num_plot+1;
figure(NumberTitle="off",Name=Fig_name) % 时域图像绘制
for i = 1:level            % 细节系数时域绘制
    subplot(level+1,1,i)
    plot(1:L(level +2-i),real(eval(['cd' num2str(i)])));
    title(sprintf('细节系数：cd%d 时域',i));
end
subplot(level+1,1,level+1) % 近似系数时域绘制
plot(1:L(1),real(eval(['ca' num2str(level)])));
title(sprintf('近似系数：ca%d 时域',level));

Fig_name = sprintf("小波分解频域图像绘制  Num %d",Num_plot); Num_plot = Num_plot+1;
figure(NumberTitle="off",Name=Fig_name) % 频域图像绘制
% 细节系数频域绘制
for i = 1:level
    process_signal = eval(['cd' num2str(i)]); % 待处理信号
    N_P = length(process_signal);             % 待处理信号长度
    % cDx 真实频域（x = 1:level）
    fft_ProcSign = 2*abs(fft(process_signal))/N_P;
    subplot(level+1,1,i)
    %fw = linspace(-L(level +2-i)/2,L(level +2-i)/2,L(level +2-i)/fs*N);
    fw = (0:N_P/2)*(fs/N); % 绘制正频部分
    stem(fw,fft_ProcSign(1:N_P/2+1));
    title(sprintf('cd%d 频域',i));
end
% 近似系数频域
process_signal = eval(['ca' num2str(level)]); % 待处理信号
N_P = length(process_signal);                 % 待处理信号长度
% cAx 真实频域（x = level）
fft_ProcSign = 2*abs(fft(process_signal))/N_P;
fw = (0:N_P/2)*(fs/N); % 绘制正频部分
subplot(level+1,1,level+1)
stem(fw,fft_ProcSign(1:N_P/2+1));
title(sprintf('近似系数：ca%d 频域',level));
%% 小波重构
s_rec = waverec(C, L, WaveName);
Fig_name = sprintf("小波重构  Num %d",Num_plot); Num_plot = Num_plot+1;
figure(NumberTitle="off",Name=Fig_name)
subplot(2,1,1)
plot(t,real(s_rec))
xlabel('时间 s'); ylabel('幅度'); title('小波重构')
subplot(2,1,2)
plot(t,abs(s_rec - Yt))
xlabel('时间 s'); ylabel('幅度'); title('重构信号与原信号差值')
%% 小波降噪
% 采用软降噪

% 通过第1层细节系数估算信号的噪声强度
sigma = wnoisest(C, L, 1);

% 缺省的阈值模型
[thr1, sorh, keepapp] = ddencmp('den', 'wv', Yt);
% 重建噪声信号
xd1 = wdencmp('gbl', C, L, WaveName, level, thr1, 's', 1);
xd1_fft = fftshift(2*abs(fft(xd1))/N);
% penalty阈值模型
alpha = 2; % 选择参数：α = 2
thr2 = wbmpen(C, L, sigma, alpha);
xd2 = wdencmp('gbl', C, L, WaveName, level, thr2, 's', 1);
xd2_fft = fftshift(2*abs(fft(xd2))/N);

% 两种重构方法图像绘制
Fig_name = sprintf("小波去噪  Num %d",Num_plot); Num_plot = Num_plot+1;
figure(NumberTitle="off",Name=Fig_name)
% 缺省阈值模型
subplot(2,2,1)
plot(t,real(xd1)); xlabel('时间 s'); ylabel('幅度'); title('缺省阈值模型重构时域')
subplot(2,2,2)
plot(f,xd1_fft); xlabel('频率 Hz'); ylabel('幅度'); title('缺省阈值模型重构频域');
% penalty阈值模型
subplot(2,2,3)
plot(t,real(xd2)); xlabel('时间 s'); ylabel('幅度'); title('penalty阈值模型重构时域')
subplot(2,2,4)
plot(f,xd2_fft); xlabel('频率 Hz'); ylabel('幅度'); title('penalty阈值模型重构频域');

% 信噪比求解
for i = 1:2
    % 定义信号功率 SigPowerX   = sum(abs(xdX).^2)/N;   (X = 1,2)
    eval(['SigPower' num2str(i) '= sum(abs(eval([''xd'' num2str(i)])).^2)/N', ';']);
    % 定义噪声功率 noisePowerX = sum(abs(Yt-xdX).^2)/N (X = 1,2)
    eval(['NoisePower' num2str(i) '= sum(abs(Yt - eval([''xd'' num2str(i)])).^2)/N', ';']);
    % 求解信噪比
    eval(['SNR' num2str(i) '= 10*log10(eval([''SigPower'' num2str(i)]) / eval([''NoisePower'' num2str(i)]))',';']);
end
% 实际信噪比求解
SigPower = sum(abs(St).^2)/N;  NoisePower = sum(abs(Yt-St).^2)/N;
SNR = 10*log10(SigPower/NoisePower);
fprintf("缺省阈值模型信噪比为：%f，penalty阈值模型信噪比为：%f\n实际参考信号信噪比为：%f \npenalty阈值模型降噪效果良好",SNR1,SNR2,SNR)








