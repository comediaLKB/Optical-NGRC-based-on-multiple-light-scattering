% optical NGRC observer
clear; clc; close all; 
%%
m = matfile('L22_Ninput64'); % generate this file using "generateKS.m"
data = m.uu(1:21000,:);
[~, num_inputs] = size(data);
transit_T = 200;
data = data(transit_T+1:end,:);
dt = 0.25;
lambda_max = 0.043; 
LL = m.d;
clear m
% data normalization
data_max = max(max(data));
data_min = min(min(data));
data_range = data_max-data_min;
data = (data-data_min)/data_range;

nInputVariable = 8;
nInterval = round(num_inputs/nInputVariable);
idxAllVariable = 1:num_inputs;
idxInputVariable = 1:nInterval:num_inputs;
% idxInputVariable(8) = []; % only for 7
idxAllVariable(idxInputVariable) = [];
idxOutputVariable = idxAllVariable;
idxAllVariable = 1:num_inputs;
nOutputVariable = num_inputs - nInputVariable;

inVariable = data(:,idxInputVariable);
data(:,idxInputVariable) = [];
outVariable = data;
if size(inVariable,2) ~= nInputVariable
    disp('Terminating the program.');
    return
end

s = 5; % delay stride size
k = 4; % delay number
inData = delayTimeSeries(inVariable, s, k);
outData = outVariable((1+(k-1)*s):end,:);
train_T = 10000;
test_T = 2000; 
train_GT = outData(1:train_T,:);
test_GT = outData(train_T+1:train_T+test_T,:);

% 
N = size(inData, 2) + 1;
M = 2500;
rng(10)
W = randn(M,N) /sqrt(2*N) + 1i * rand(M,N) /sqrt(2*N);

cam_bit_depth = 8;
slm_bit_depth = 8;
max_slm = 2^slm_bit_depth - 1;
max_grayvalue = 2^cam_bit_depth - 1;
input_bias = 1.5;
exp_th = 10;

%%
x_all = zeros(M, train_T+test_T);
for kkk =1: train_T+test_T
    data_now0 = inData(kkk,:);
    reinput = [data_now0, input_bias]';
    data_combined_phase = mod(pi*reinput, 2*pi);
    data_combined_phase = data_combined_phase / (2*pi);
    data_combined_phase = floor(data_combined_phase * max_slm);
    data_combined_phase = (data_combined_phase / max_slm) * (2*pi);
    x = abs(W*exp(1i*data_combined_phase)).^2;
    logicalIndex = x > exp_th;
    x(logicalIndex) = exp_th;
    x = x / exp_th;
    x_temp = floor(x * max_grayvalue) / max_grayvalue;
    x_all(:,kkk) = x_temp;
end

%% training 
x_train = x_all(:,1:train_T);
x_test = x_all(:,train_T+1:end);
beta_lib = logspace(-3,-1,10);
mse_lib = zeros(1, length(beta_lib));
[U, S, V] = svd(x_train', 'econ');
% % S = diag(S);
S_squared_diag = S.^2;
w = U' * train_GT;
for idx_beta = 1:length(beta_lib)
    beta = beta_lib(idx_beta);
    w_out = (V * (S./(S_squared_diag + beta)) * w)';
    output_test = w_out * x_test;
    error_test = output_test - test_GT';
    mse = mean2((error_test).^2);
    mse_lib(idx_beta) = mse; 
end
figure; loglog(beta_lib, mse_lib);
[~, idx_best_beta] = min(mse_lib);

%%
best_beta = beta_lib(idx_best_beta);
w_out = (V * (S./(S_squared_diag + best_beta)) * w)';
output_test = w_out * x_test;
error_test = output_test - test_GT';
mse = mean2((error_test).^2);

% Show the prediction results
output_test_full =  zeros(size(output_test, 2), num_inputs);
test_GT_full =  zeros(size(output_test, 2), num_inputs);
output_test_full(:,idxInputVariable) = inVariable(((1+(k-1)*s)+train_T):((1+(k-1)*s)+train_T+test_T-1),:);
output_test_full(:,idxOutputVariable) = output_test';
test_GT_full(:,idxInputVariable) = inVariable(((1+(k-1)*s)+train_T):((1+(k-1)*s)+train_T+test_T-1),:);
test_GT_full(:,idxOutputVariable) = test_GT;

t = (1:1:test_T)*dt*lambda_max;
ss = (1:1:num_inputs).*LL/num_inputs;
error_test_full = output_test_full - test_GT_full;

figure(201),
% sgtitle(strcat('Regularization =',num2str(beta)))
subplot(3,1,1)
imagesc(t,ss,test_GT_full');title('Actual');colorbar;
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')

subplot(3,1,2)
imagesc(t,ss,output_test_full');title('Prediction');colorbar;
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
%xlim([0, 20])
% caxis(1*[0,1])

subplot(3,1,3)
imagesc(t,ss,error_test_full');title('Error');colorbar;
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
caxis(1*[-0.5,0.5])
colormap('jet');
