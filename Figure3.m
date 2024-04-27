%% FIGURE 3
clear all;close all;
%% Short-term prediction
m = matfile('L22_Ninput64'); % generate this file using "generateKS.m"
data = m.uu(1:21000,:);
[~, num_inputs] = size(data);
transit_T = 620;
data = data(transit_T+1:end,:);

data_max = max(max(data));
data_min = min(min(data));
data_range = data_max-data_min;
data = (data-data_min)/data_range; 
dt = 0.75;
lambda_max = 0.043;
LL = m.d;
clear m
predict_length = 200;
exp_data_ks_ST_path1 = '...\exp_results\exp_data_KS_ST.mat';
load(exp_data_ks_ST_path1);
t = (1:1:predict_length)*dt*lambda_max;
s = (1:1:num_inputs).*LL/num_inputs;
error_test = predict_output - test_GT;

predict_output_unnorm = predict_output * data_range + data_min;
test_GT_unnorm = test_GT * data_range + data_min;
error_test_unnorm = predict_output_unnorm - test_GT_unnorm;

figure('color','w');
tttt = tiledlayout(3,1,'TileSpacing','Compact');
h(1) = nexttile(tttt);
imagesc(t,s,test_GT_unnorm);
xticklabels({});
h(2) = nexttile(tttt);
imagesc(t,s,predict_output_unnorm);
xticklabels({});
h(3) = nexttile(tttt);
imagesc(t,s,error_test_unnorm);
colormap('jet');
error_EW = calculateNRMSE(predict_output_unnorm, test_GT_unnorm, 'element-wise');
%% Long-term prediction
exp_data_ks_LT_path2 = '...\exp_results\exp_data_KS_LT.mat';
load(exp_data_ks_LT_path2);
tStart = 2761;
tEnd = 5660;
t_idx = (tStart:1:tEnd);
t_real = t_idx*0.0107;
s = (1:1:64).*22/64;
error_test = predict_output - test_GT;
figure('color','w');
tttt = tiledlayout(3,1,'TileSpacing','Compact');
h(1) = nexttile(tttt);
imagesc(t_real,s,test_GT);
xticklabels({});
h(2) = nexttile(tttt);
imagesc(t_real,s,predict_output);
xticklabels({});
h(3) = nexttile(tttt);
imagesc(t_real,s,error_test);
colormap('jet');
