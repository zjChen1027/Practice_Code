close all; clear; clc;
% LPF 线性预测编码函数实例
% 
% 一个平稳信号 与 一个非平稳采用AR模型的预测效果
% 结论：AR模型对平稳信号有较强的预测，非平稳信号尽可预测少量近似点
%==============================================================author：Chen
%% 参数定义
T = 10e-6; % 信号持续时间
B = 10e6;  % 信号带宽
fs = 10*B; % 采样频率
K = B/T;   % chirp 斜率
N = round(T*fs);% 采样点数
% 坐标轴绘制
t = linspace(0,T,N);
f = linspace(-fs/2,fs/2,N);
%% 实例函数定义(无噪声)
signal_1 = 10*cos(2*pi*4e6*t) + 9*sin(2*pi*5e6*t); % 实例信号 1
signal_2 = 10*exp(1i*pi*K*t.^2);                   % 实例信号 2

% gauss_noise = randn(1,N);% 产生高斯白噪声
% 添加gauss白噪声
% signal_1 = signal_1 + 0.5*gauss_noise;
% signal_2 = signal_2 + 0.5*gauss_noise;

% 原始信号绘制
figure(NumberTitle="off", Name="待预测原信号")
subplot(2,1,1)
plot(t*1e6,real(signal_1))
xlabel('时间 μs'); ylabel('幅度'); title('signal 1')
subplot(2,1,2)
plot(t*1e6,real(signal_2))
xlabel('时间 μs'); ylabel('幅度'); title('signal 2')
%% LPC模型
n_predict = 200;% 预测长度 对应时间 n_predict/fs
p = 500;        % 模型阶数

% 存放预测的数据组合
Predict_data_set = zeros(2,n_predict + length(signal_1));
for n = 1:2
    % 获得待处理信号变量
    disp_signal = eval(['signal_',mat2str(n)]);
    ai = arburg(disp_signal,p);      % 求解预测系数
    singal_len = length(disp_signal);% 获得信号长度
    
    % 创建预测矩阵
    pridect_signal = zeros(1,n_predict + singal_len);
    pridect_signal(1:singal_len) = disp_signal;
    % AR模型
    for i = 1:n_predict
        pridect_signal(singal_len + i) = - sum(fliplr(ai(2:end)) .* pridect_signal(singal_len-p +i:singal_len+ i-1));
    end

    Predict_data_set(n,:) = pridect_signal;% 保存预测之后的数据
    % 图像绘制：
    % 新的坐标尺度
    t_new = linspace(0,T + n_predict/fs,N + n_predict);
    f_new = linspace(-fs/2,fs/2,N + n_predict);

    Fig_name = sprintf("Signal %d, 模型阶数 %d, 预测 %d 点参考图",n,p,n_predict);
    figure(NumberTitle="off",Name=Fig_name)
    % 预测信号与原始信号时域
    subplot(2,1,1)
    plot(t_new*1e6,real(pridect_signal),t*1e6,real(disp_signal),'r')
    title_name = sprintf("Signal %d 时域",n);
    xlabel('时间 μs'); ylabel('幅度'); title(title_name); legend('向后预测信号','原始信号')
    
    disp_fft = fftshift(abs(fft(disp_signal)/N));
    pridect_sig_fft = fftshift(abs(fft(pridect_signal)/(N + n_predict)));
    % 预测信号与原始信号频域
    subplot(2,1,2)
    plot(f_new*1e-6,pridect_sig_fft,f*1e-6,disp_fft,'r')
    title_name = sprintf("Signal %d 频域",n);
    xlabel('频率 MHz'); ylabel('幅度'); title(title_name); legend('向后预测信号','原始信号')
end

%% 算法比较
% 注：未合并原始数据，仅计算不同算法预测后的数据 时域、频域、时域差
%{
y = zeros(2,n_predict);% 矩阵初始化
for n = 1:2
    % 读取原信号
    disp_signal = eval(['signal_',mat2str(n)]);
    y(n,:) = (for_predictm(disp_signal,n_predict,p))'; % 算法2向后预测

    t_compare = linspace(0,n_predict/fs,n_predict);    % 初始化刻度
    f_compare = linspace(-fs/2,fs/2,n_predict);

    Fig_name = sprintf("Signal %d 算法比较",n);
    figure(NumberTitle="off", Name=Fig_name)
    subplot(3,1,1)       % 算法时域比较
    plot(t_compare*1e6,real(y(n,:)));  hold on;
    plot(t_compare*1e6,real(Predict_data_set(n,N+1:end)),'r');
    title_name = sprintf("算法 1、2时域");
    xlabel('时间 μs'); ylabel('幅度'); title(title_name); legend('算法2','本例程算法')

    y_fft = fftshift(abs(fft(y(n,:)/length(y(n,:)))));
    disp_fft = fftshift(abs(fft(Predict_data_set(n,N+1:end))/n_predict));
    subplot(3,1,2)       % 算法频域比较
    plot(f_compare*1e-6,y_fft,'k',f_compare*1e-6,disp_fft,'r')
    title_name = sprintf("算法 1、2频率");
    xlabel('频率 MHz'); ylabel('幅度'); title(title_name); legend('算法2频域','本例程频域')

    % 比较两算法差异
    diff_predict = real(y(n,:) - Predict_data_set(n,N+1:end));
    n1 = find(diff_predict >= 0);  % 找出大于或等于0的元素的序号
    n2 = find(diff_predict  < 0);  % 找出小于0的元素的序号
    subplot(3,1,3)       % 算法差异
    plot(t_compare*1e6,diff_predict,'k');  hold on;
    plot(n1/fs*1e6,diff_predict(n1),'r*') % 红色星号表示算法2≥本例程算法的值
    plot(n2/fs*1e6,diff_predict(n2),'g*') % 绿色星号表示算法2<本例程算法的值
    title_name = sprintf("算法 1、2差异");
    xlabel('时间 μs'); ylabel('幅度'); title(title_name); legend('','算法2≥本例程算法','算法2<本例程算法')
end
%}



