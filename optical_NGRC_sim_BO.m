% optical NGRC simulation - Bayesian optimization
clear; clc; close all; 
%%
m = matfile('L22_Ninput64'); % generate this file using "generateKS.m"
data_original = m.uu(1:21000,:);
cdelta = 0.0;   
LL = m.d;
num_inputs = size(data_original,2);
clear m;
transit_T =200;
data_original = data_original(transit_T+1:end,:);
data_max = max(max(data_original));
data_min = min(min(data_original));
data_range = data_max-data_min;
data = (data_original-data_min)/data_range;
%%
sigma = optimizableVariable('sigma', [0.1 1.5], 'Type', 'real');
yita = optimizableVariable('yita', [0.4 1.2], 'Type', 'real');
beta = optimizableVariable('beta', [0.1 5], 'Type', 'real');
parameter = [sigma, yita, beta];
global bayes_predict_output

objFun = @(parameter) getObjValue(parameter, data);
iter = 250;
rng(0);
results = bayesopt(objFun,parameter,'Verbose',1,'UseParallel',false,...
    'IsObjectiveDeterministic', false,'AcquisitionFunctionName','expected-improvement-plus',...
    'MaxObjectiveEvaluations', iter); % -per-second-plus
%% peek the intermediate results
idx_iter = 78;
num_inputs = 64;
train_T = 8000;
predict_length = 600;
idx_predict_output = ((idx_iter-1) * num_inputs + 1):idx_iter * num_inputs;
predict_output = bayes_predict_output(idx_predict_output,1:predict_length);
test_GT = data(train_T:train_T+predict_length-1,:)';
dt = 0.25;
lambda_max = 0.043;
LL = 22;
t = (1:1:predict_length)*dt*lambda_max;
s = (1:1:num_inputs).*LL/num_inputs;
error_test = predict_output - test_GT;

figure(208),
% sgtitle(strcat('Regularization =',num2str(beta))) 
subplot(3,1,1)
imagesc(t,s,test_GT);title('Actual');colorbar;
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
subplot(3,1,2)
imagesc(t,s,predict_output);title('Prediction');colorbar;
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
% xlim([0, 20])
caxis(1*[0,1])
subplot(3,1,3)
imagesc(t,s,error_test);title('Error');colorbar;
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
caxis(1*[-0.5,0.5])
colormap('jet');
%%
function objvalue = getObjValue(parameter, data)
    train_T = 8000;
    data_previous = data(1:train_T,:);
    data_now = data(2:train_T+1,:);
    data_for_training = data(3:train_T+2,:);
    num_inputs = size(data,2);
    % hyper-parameters
    sigma = parameter.sigma;
    yita = parameter.yita;
    beta= parameter.beta;
    exp_th = 7;
    noise = 1;
    data_combined = [data_now yita*data_previous sigma*ones(length(data_now),1)]';
    slm_quan = 8;
    cam_quan = 8;
    max_slm = 2^slm_quan - 1;
    max_cam =2^cam_quan - 1;
    data_combined_phase = mod(pi*data_combined, 2*pi);
    data_combined_phase = data_combined_phase / (2*pi);
    data_combined_phase = floor(data_combined_phase * max_slm);
    data_combined_phase = (data_combined_phase / max_slm) * (2*pi);
    
    concat_input = 1;
    N=129; % 2*64+1
    M=2500; % 3000
    rng(0)
    W = randn(M,N) /sqrt(N) + 1i * randn(M,N) /sqrt(N);
    xtrain = abs(W*exp(1i*data_combined_phase)).^2;
    if noise
        noise_std_per_unit = 0.017;
        for idx_noise = 1:size(xtrain, 2)
            speckle = xtrain(:,idx_noise);
            noisy_speckle = addNoise(speckle, noise_std_per_unit);
            xtrain(:,idx_noise) = noisy_speckle;
        end
    end
    logicalIndex = xtrain > exp_th;
    xtrain(logicalIndex) = exp_th;
    xtrain = xtrain / exp_th;
    xtrain = floor(xtrain * max_cam) / max_cam;
    if concat_input
        xtrain = cat(1,xtrain,data_now');
        M = M + num_inputs;
    end
    idenmat = beta*speye(M);
    w_out = transpose(data_for_training)*transpose(xtrain)*pinv(xtrain*transpose(xtrain)+idenmat);
    output_train = w_out*xtrain;
	% prediction
    global bayes_predict_output
    predict_length = 600;
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
    nrmse = sum(calculateNRMSE(predict_output,test_GT,'total',300));
    bayes_predict_output = [bayes_predict_output; predict_output];
    objvalue = nrmse;
end