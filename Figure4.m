%% FIGURE 4
clear all;close all;
set(groot,'defaulttextinterpreter','latex');
%% Lorenz observer
cTrain = [100 100 190]/255;
cTest = [190 100 100]/255;
sigma = 10;
rho = 28;
beta = 8/3;
x0 = [-0.8032; -0.1800; 19.4488]; 
dt = 0.05;
total_T = 80;
[~, data_original] = ode45(@(t,x) lorenz_system(x, sigma, rho, beta), 0:dt:total_T, x0);
N_transitT = 10/dt;

data_original = data_original(N_transitT+1:end,:);
data_max = max(max(data_original));
data_min = min(min(data_original));
data_range = data_max-data_min;
exp_lorenz_observer_path = '...\exp_results\exp_data_lorenz_observer.mat';
load(exp_lorenz_observer_path);

test_T = 401:400+800;
x_show_test = test_T*dt;
test_input_unnorm = test_input * data_range + data_min;
test_GT_unnorm = test_GT * data_range + data_min;
output_test_unnorm = output_test * data_range + data_min;
figure('color','w');
subplot(3,1,1); plot(x_show_test,test_input_unnorm(:,1),'color',cTrain);
subplot(3,1,2); plot(x_show_test,test_input_unnorm(:,2),'color',cTrain);
subplot(3,1,3);plot(x_show_test,test_GT_unnorm,'color',cTrain); 
hold on; plot(x_show_test,output_test_unnorm,'color',cTest);
legend('z of GT','z of prediction');
error = calculateNRMSE(output_test_unnorm, test_GT_unnorm', 'element-wise');
%% KS results
exp_path1 = '...\exp_results\KS_observer_7.mat';
load(exp_path1);
dt = 0.25;
lambda_max = 0.043;
LL = 22;
num_inputs = 64;
test_T = 2000; 
tStart = 10001;
t = (tStart:1:tStart+test_T-1)*dt*lambda_max;
ss = (1:1:num_inputs).*LL/num_inputs;
error_test_full = output_test_full - test_GT_full;
figure('color','w');
tttt = tiledlayout(3,1,'TileSpacing','Compact');
h(1) = nexttile(tttt);
imagesc(t,ss,test_GT_full');
xticklabels({});
h(2) = nexttile(tttt);
imagesc(t,ss,output_test_full');
xticklabels({});
h(3) = nexttile(tttt);
imagesc(t,ss,error_test_full');
colormap('jet');
ks_corr = corr2(test_GT_full,output_test_full);
%% 
c_observer = '#1D2088';
c_spline = '#C30D23';
ks_observer_corr = [0.9042,0.8738,0.9318,0.9838,0.9982];
ks_spline_corr = [0.3668, 0.5665, 0.7436, 0.8051, 0.9283];
ks_corr = [ks_observer_corr;ks_spline_corr];
x = [4,5,6,7,8];
figure('color','w');
b = bar(x,ks_corr','histc');
b(1).FaceColor = c_observer;
b(2).FaceColor = c_spline;
% axis padded
xlim([3.8 8.74]);
ylim([0.2 1.04])