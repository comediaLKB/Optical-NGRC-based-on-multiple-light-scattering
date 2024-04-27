% Optical conventional RC simulation
clear; clc; close all; 
%%
m = matfile('L22_Ninput64'); % generate this file using "generateKS.m"
data_original = m.uu;
cdelta = 0.0;   
LL = m.d;
num_inputs = size(data_original,2);
clear m;
transit_T = 200;
train_T = 10000;
test_T = 100;
transit_train_length = transit_T + train_T;
dt = 0.25;
lambda_max = 0.043;

data_max = max(max(data_original));
data_min = min(min(data_original));
data_range = data_max-data_min;
data = (data_original-data_min)/data_range; % normalized to [0,1]
data_for_training = data(transit_T+2:transit_train_length+1,:); % 

% HPs (Note they should be optimized through BO)
leak_rate = 0.99869; % 0 - 1
input_scaling =  0.10318 * 1000  ; %  input scaling
alpha = 1; % encoding range
beta = 0.20035; % regularization coefficient
bias = 1; % bias term

M=3000; % reservoir feature size
N=M+num_inputs+1; % input size = reservoir + input + bias
rng(0)
W = randn(M,N) /sqrt(N) + 1i * randn(M,N) /sqrt(N);
W(:,M+1:end-1) = input_scaling * W(:,M+1:end-1);
x0 = rand(M,1);
x_transit_train = zeros(M,transit_train_length);
x_transit_train(:,1) = x0; % reservoir initialization
concat_input = 1; % concat_input = 0 not concat; concat_input = 1 concat
RC_size_readout = M + concat_input*num_inputs; % the RC size used for readout
%% Training
for k = 1:transit_train_length-1
    input = data(k,:); 
    reservoir = x_transit_train(:,k);
    running_input = cat(1,reservoir,input',bias); 
    cam_image = abs(W * exp(1i*alpha*pi*running_input)).^2; 
    cam_image = cam_image ./ max(cam_image);
    next_reservoir = (1 - leak_rate) * reservoir + leak_rate * cam_image;
    x_transit_train(:,k+1) = next_reservoir;
end
% abandon the transit state
xtrain = x_transit_train(:,transit_T+1:end);
if concat_input==1
    xtrain = cat(1,xtrain,data(transit_T+1:transit_train_length,:)');
end
%% w_out calculation
idenmat = beta*speye(RC_size_readout);
w_out = transpose(data_for_training)*transpose(xtrain)*pinv(xtrain*transpose(xtrain)+idenmat);
output_train = w_out*xtrain;
error_train = output_train - data_for_training';
figure;
t_train = (1:1:train_T)*dt*lambda_max;
s_train = (1:1:num_inputs).*LL/num_inputs;
subplot(3,1,1);imagesc(t_train,s_train, data_for_training');colorbar;title('GT')
subplot(3,1,2);imagesc(t_train,s_train, output_train);colorbar;title('Training');
subplot(3,1,3);imagesc(t_train,s_train,error_train);colorbar;title('error');
%% prediction
predict_length = 600;
predict_output = zeros(num_inputs, predict_length);
test_GT = data(transit_train_length+1:transit_train_length + predict_length,:)';
x_concat = xtrain(:,end);
next_reservoir = x_concat(1:M,:);
for j=1:predict_length
    output = w_out*x_concat; 
    predict_output(:,j) = output;
    running_input = cat(1, next_reservoir, output, bias);
    cam_image = abs(W * exp(1i*alpha*pi*running_input)).^2;
    cam_image = cam_image ./ max(cam_image);
    next_reservoir = (1 - leak_rate) * next_reservoir  + leak_rate * cam_image;
    if concat_input==1
        x_concat = cat(1, next_reservoir, output);
    else
        x_concat = next_reservoir;
    end
end
t = (1:1:predict_length)*dt*lambda_max;
s = (1:1:num_inputs).*LL/num_inputs;
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
caxis(1*[-0.5,0.5])
colormap('jet');