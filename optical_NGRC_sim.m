% Optical NGRC simulation
clear; clc; close all; 
%%
m = matfile('L22_Ninput64'); % generate this file using "generateKS.m"
data_original = m.uu(1:21000,:);
cdelta = 0.0;   
LL = m.d;
num_inputs = size(data_original,2);
clear m;
transit_T = 200;
train_T = 8000;
test_T = 300;
dt = 0.25;
lambda_max = 0.043;
%%
data_original = data_original(transit_T+1:end,:);
data_max = max(max(data_original));
data_min = min(min(data_original));
data_range = data_max-data_min;
data = (data_original-data_min)/data_range;

data_previous = data(1:train_T,:);
data_now = data(2:train_T+1,:);
data_for_training = data(3:train_T+2,:);

% HPs (Note they should be optimized through BO)
sigma = 1.05; % bias term 1
yita = 0.93; % relative weights
beta = 0.59; % regularization parameter

data_combined = [data_now yita*data_previous sigma*ones(length(data_now),1)]';
concat_input = 1; % 1 is on; 0 is off (it's not necessary to set it as 1)
exp_th = 9;% This is to simulate overexposure of cameras in real experimental systems
noise = 0; % 1 is on; 0 is off
slm_quan = 8; % SLM quantization bit depth
cam_quan = 8; % Camera quantization bit depth
max_slm = 2^slm_quan - 1;
max_cam =2^cam_quan - 1;

N=129; % input mode
M=2500; % output mode
rng(10)
W = randn(M,N) /sqrt(2*N) + 1i * rand(M,N) /sqrt(2*N);

% mod and quantization
data_combined_phase = mod(pi*data_combined, 2*pi);
data_combined_phase = data_combined_phase / (2*pi);
data_combined_phase = floor(data_combined_phase * max_slm);
data_combined_phase = (data_combined_phase / max_slm) * (2*pi);
xtrain = abs(W*exp(1i*data_combined_phase)).^2;
if noise
    noise_std_per_unit = 0.017; % higher value indicates stronger noise
    for idx_noise = 1:size(xtrain, 2)
        speckle = xtrain(:,idx_noise);
        noisy_speckle = addNoise(speckle, noise_std_per_unit);
        xtrain(:,idx_noise) = noisy_speckle;
    end
end
% overexposure effect
logicalIndex = xtrain > exp_th;
xtrain(logicalIndex) = exp_th;
% norm and quantization
xtrain = xtrain / exp_th;
xtrain = floor(xtrain * max_cam) / max_cam;

if concat_input
    xtrain = cat(1,xtrain,data_now');
    M = M + num_inputs;
end
%% w_out calculation
idenmat = beta*speye(M);
w_out = transpose(data_for_training)*transpose(xtrain)*pinv(xtrain*transpose(xtrain)+idenmat);
output_train = w_out*xtrain;
error_train = output_train - data_for_training';
x_show_train = dt*(1:size(error_train,2));
figure;
t_train = (1:1:train_T)*dt*lambda_max;
s_train = (1:1:num_inputs).*LL/num_inputs;
subplot(3,1,1);imagesc(t_train,s_train, data_for_training');colorbar;title('GT')
subplot(3,1,2);imagesc(t_train,s_train, output_train);colorbar;title('Training');
subplot(3,1,3);imagesc(t_train,s_train,error_train);colorbar;title('error');
%% prediction
predict_length = 500;
predict_output = zeros(num_inputs, predict_length);
predict_output(:,1) = data_now(end-1,:)';
predict_output(:,2) = data_now(end,:)';
test_GT = data(train_T:train_T+predict_length-1,:)';

for kk = 3:predict_length
    reinput = [predict_output(:,kk-1);yita*predict_output(:,kk-2);sigma];
    data_combined_phase = mod(pi*reinput, 2*pi);
    data_combined_phase = data_combined_phase / (2*pi);
    data_combined_phase = floor(data_combined_phase * max_slm);
    data_combined_phase = (data_combined_phase / max_slm) * (2*pi);
    x = abs(W*exp(1i*data_combined_phase)).^2;
    if noise
        x = addNoise(x, noise_std_per_unit);   
    end
    logicalIndex = x > exp_th;
    x(logicalIndex) = exp_th;
    x = x / exp_th;
    x = floor(x * max_cam) / max_cam;
    if concat_input
        x = cat(1,x, predict_output(:,kk-1));
    end
    out = w_out*x;
    predict_output(:,kk) = out;
end

t = (1:1:predict_length)*dt*lambda_max;
s = (1:1:num_inputs).*LL/num_inputs;

% predict_output = predict_output * data_range + data_min; % normalize back
% to original data range
% test_GT = test_GT * data_range + data_min;
error_test = predict_output - test_GT;

figure,
sgtitle(strcat('Regularization =',num2str(beta))) 
subplot(3,1,1)
imagesc(t,s,test_GT);title('Actual');colorbar;
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')

subplot(3,1,2)
imagesc(t,s,predict_output);title('Prediction');colorbar;
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
%xlim([0, 20])
% caxis(1*[0,1])

subplot(3,1,3)
imagesc(t,s,error_test);title('Error');colorbar;
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
% caxis(1*[-0.5,0.5])
% caxis(1*[-2.5,2.5])
colormap('jet');