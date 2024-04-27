%% FIGURE 2
clear all;close all;
set(groot,'defaulttextinterpreter','latex');
cTrain = [100 100 190]/255;
cTest = [190 100 100]/255;
%% lorenz 
sigma = 10;
rho = 28;
beta = 8/3;
x0 = [-0.8032; -0.1800; 19.4488]; % initial conditions
dt = 0.025;
total_T = 210;
N_transitT = 10/dt;
N_trainT = 4000;
N_testT = 400;
[~, data_original] = ode45(@(t,x) lorenz_system(x, sigma, rho, beta), 0:dt:total_T, x0);
data_original = data_original(N_transitT+1:end,:);

x_show = (1:400)*dt;
figure('color','w');subplot(3,1,1);plot(x_show, data_original(1:400,1)','color',cTrain);xticklabels({});%legend('x');
subplot(3,1,2);plot(x_show, data_original(1:400,2)','color',cTrain);xticklabels({});%legend('y');
subplot(3,1,3);plot(x_show, data_original(1:400,3)','color',cTrain);ylim([5,45]);%legend('z');

data_max = max(max(data_original));
data_min = min(min(data_original));
data_range = data_max-data_min;
%% reservoir dynamics
trace_path = '...\exp_results\LorenzReservoirTrace.mat';
load(trace_path);
time_span = 1:400;
x_show = (1:400)*dt;
figure('color','w');
for i=1:10
    plot(x_show,reservoir_trace_train(time_span,i)/55);
    hold on
end
%% Short-term prediction
exp_data_lorenz_ST_path = '...\exp_results\exp_data_lorenz_ST.mat';
load(exp_data_lorenz_ST_path);
predict_length = N_testT;
predict_output_unnorm = predict_output * data_range + data_min;
test_GT_unnorm = test_GT * data_range + data_min;
error_test_unnorm = predict_output_unnorm - test_GT_unnorm;
x_show_test = (1:predict_length)*dt;
error = calculateNRMSE(predict_output_unnorm(:,1:200), test_GT_unnorm(:,1:200), 'element-wise');

figure('color','w');
subplot(3,1,1); plot(x_show_test,predict_output_unnorm(1,:),'color',cTest);
hold on; plot(x_show_test,test_GT_unnorm(1,:),'color',cTrain);xticklabels({});
% legend('x of output','x of GT');

subplot(3,1,2); plot(x_show_test,predict_output_unnorm(2,:),'color',cTest);
hold on; plot(x_show_test,test_GT_unnorm(2,:),'color',cTrain);xticklabels({});
% legend('y of output','y of GT');

subplot(3,1,3); plot(x_show_test,predict_output_unnorm(3,:),'color',cTest);
hold on; plot(x_show_test,test_GT_unnorm(3,:),'color',cTrain);
% legend('z of output','z of GT');ylim([5,45]); 
xticklabels({'100','','102','','104','','106','','108','','110'});

%% Long-term prediction
exp_data_lorenz_LT_path = '...\exp_results\exp_data_lorenz_LT.mat';
load(exp_data_lorenz_LT_path);
sigma = 10;
rho = 28;
beta = 8/3;
x0 = [-0.8032; -0.1800; 19.4488];
dt = 0.025;
total_T = 410;
[t, data_original] = ode45(@(t,x) lorenz_system(x, sigma, rho, beta), 0:dt:total_T, x0);
N_transitT = 10/dt;
N_trainT = 8000;
N_testT = 8000;
data_original = data_original(N_transitT+1:end,:);
data_max = max(data_original,[],1);
data_min = min(data_original,[],1);
data_range = data_max - data_min;

predict_output_x = predict_output(1,:);
predict_output_z = predict_output(3,:);
test_GT_x = test_GT(1,:);
test_GT_z = test_GT(3,:);

predict_output_x_unnorm = predict_output_x .* data_range(1) + data_min(1);
predict_output_z_unnorm = predict_output_z .* data_range(3) + data_min(3);
test_GT_x_unnorm = test_GT_x .* data_range(1) + data_min(1);
test_GT_z_unnorm = test_GT_z .* data_range(3) + data_min(3);

figure('color','w');
plot(predict_output_x_unnorm,predict_output_z_unnorm, 'color', cTest);

[maxima_GT, ~] = findpeaks(test_GT_z_unnorm);
[maxima_prediction,~] = findpeaks(predict_output_z_unnorm);
figure('color','w');
plot(maxima_GT(1:end-1), maxima_GT(2:end), '.', 'MarkerSize',15, 'color',cTrain);
hold on; plot(maxima_prediction(1:end-1), maxima_prediction(2:end), '.', 'MarkerSize',14, 'color',cTest);
xlabel('z(t)');
ylabel('z(t - dt)');
title('Lorenz Attractor Return Map of Successive Maxima');
