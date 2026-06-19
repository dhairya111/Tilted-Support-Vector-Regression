clear all;
close all;

data = readmatrix("/MATLAB Drive/Datasets/californiaHousing.csv");

% Split into train test
x = data(:, 1:end-1);
y = data(:, end);

x = normalize(x);

% Set split ratios
trainRatio = 0.8;   % 80% for training
valRatio = 0.1;     % 10% for validation
testRatio = 0.1;    % 10% for testing

% Shuffle the data
rng(1);             % Set seed for reproducibility
n = size(data, 1);  % Total number of samples

% Dividing into train, val, test set
[trainIdx, valIdx, testIdx] = dividerand(n, trainRatio, valRatio, testRatio);

x_train = x(trainIdx, :);
y_train = y(trainIdx);

xval = x(valIdx, :);
yval = y(valIdx);

test = x(testIdx, :);
testy = y(testIdx);

x = x_train;
y = y_train;

% determine amount of outliers
percentOutliers = 0.2;

n = size(x, 1);  % Total number of data points
numPoints = round(n * percentOutliers/2);  % 10% of data points

% Randomly select indices
% rng(1);  % Set seed for reproducibility
randomIndices1 = randperm(n, numPoints);
randomIndices2 = randperm(n, numPoints);

% Multiply half of selected points by 5 and other half by -5
% y(randomIndices1) = y(randomIndices1) + unifrnd(-15*std(y),15*std(y),[length(randomIndices1),1]);
% % y(randomIndices2) = y(randomIndices2) + unifrnd(-15*std(y),15*std(y),[length(randomIndices2),1]);
y(randomIndices1) = abs(y(randomIndices1) *5);
y(randomIndices2) = -1*abs(y(randomIndices2) *5);
% y(randomIndices1) = y(randomIndices1) + abs(normrnd(0,7,[length(randomIndices1),1]));
% y(randomIndices2) = y(randomIndices2) + abs(normrnd(0,7,[length(randomIndices2),1]));
% y(randomIndices1) = normrnd(5,5,size(y(randomIndices1)));
% y(randomIndices2) = normrnd(5,5,size(y(randomIndices1)));

%scatter(x,y);

beta_max = max(abs(y));
beta_min = median(abs(y));


result = {'loss type','mse', 'mae', 'sse/sst', 'ssr/sst', 'alpha', 'c', '% of weights<0.01', 'train time'};
    
iters = 10000;
degree = 1;

% 
K_X = GaussianKernel(x_train, degree);
K_Xval = GaussianKernelTest(x,xval,degree);
K_Xtest =  GaussianKernelTest(x,test, degree);
% K_X = x_train;
% K_Xval = xval;
% K_Xtest = test;
lambda = 0;
%% 

%kernel regression
lrate = 0.003;
% rng(k);
% tic;
% [u, b, ssevals] = KernelRegressionSGD(K_X, y, degree,iters,lrate);
% train_time = toc;
% ytest_pred = K_Xtest * u+b;
% mse = mean((ytest_pred-testy).^2);
% mae =  mean(abs(ytest_pred-testy));
% sst = sum((testy - mean(testy)).^2);
% sse  = sum((ytest_pred - testy).^2);
% ssr = sum((ytest_pred - mean(testy)).^2);
% m1 = sse/sst;
% m2 = ssr/sst;
% sparsity = sum(abs(u) < 0.01)*100/numel(u);
% result = [result; {'Least Squares Loss',mse,mae,m1,m2,0,1,sparsity,train_time}];
%% 

%truncated loss
%rng(k);
% lrate = 0.0001;
% iters = 1000;
% %Truncated LS function
% trunc_tuned_mse = Inf;
% trunc_beta = 0;
% final_u = 0;final_b = 0;
% sse_vals_finals = 0;
% final_deg = 0;
% final_lambda =  0;
% final_lr = 0;
% 
% %tuning beta
% tunetimer = tic;
% for beta = linspace(beta_min,beta_max,10)
%     traintimer = tic;
%     % K_X = GaussianKernel(x_train, degree);
%     % K_Xval = GaussianKernelTest(x,xval,degree);
%     % K_Xtest =  GaussianKernelTest(x,test, degree);
% 
%      [u,b, sse_val, gradval] = TruncatedLSLossFunctionSGD(K_X, y, degree, iters, lrate, beta,lambda);
%    % [u,b, sse_val, gradval] = TruncatedLossWithoutKernels(K_X, y, degree, iters, lrate, beta,lambda);
%     train_time = toc(traintimer);
%         yval_pred =  K_Xval*u + b;
%         mse = mean((yval_pred-yval).^2);
%         if trunc_tuned_mse>mse
%             trunc_tuned_mse = mse;
%             trunc_beta=beta;
%             final_u = u;
%             final_b = b;
%             sse_vals_finals = sse_val;
%             final_deg = degree;
%             final_lambda =  lambda;
%             final_lr = lrate;
%         end
% end  
% tune_time = toc(tunetimer);
% 
% ytest_pred = K_Xtest*final_u + final_b;
% mse = mean((ytest_pred-testy).^2);
% mae =  mean(abs(ytest_pred-testy));
% sst = sum((testy - mean(testy)).^2);
% sse  = sum((ytest_pred - testy).^2);
% ssr = sum((ytest_pred - mean(testy)).^2);
% m1 = sse/sst;
% m2 = ssr/sst;
% sparsity = sum(abs(final_u) < 0.01)*100/numel(final_u);
% fprintf("truncated loss mse: %f\n",mse);
% fprintf("truncated loss mae: %f\n",mae);
% fprintf("truncated loss sse/st: %f\n",m1);
% fprintf("truncated loss ssr/st: %f\n",m2);
% fprintf("truncated loss training time: %f\n",train_time);
% fprintf("truncated loss tuning time: %f\n",tune_time);
% figure('Name','truncated loss sse');
% plot(sse_vals_finals);
% result = [result; {'Truncated Loss',mse,mae,m1,m2,0,trunc_beta,sparsity,train_time}];
%% 

%barron function
% hyperparameter tuning
lrate = 50^6;
%eta_list = [10^-4,10^-3,10^-2,10^-1,1,10^1,10^2,10^3,10^4];
% c_list = linspace(beta_min,beta_max,15);
c_list = [1070.5,1078,2313.4];
alpha_list = [-2^5,-2^4,-2^3,-2^2,-2,-1,-2^-1,-2^-2,0,2^-2,2^-1,2^0,2^1];
degree_list = [1,2,2^2,2^3,2^4,2^5,2^6];

ufinal=0;bfinal=0;final_beta=0;finaleta=0;
final_mse =Inf;
iters = 10000;
sse_vals_finals = 0;
final_degree=0;


    for alpha_idx = 1:length(alpha_list)
        for idx = 1:length(c_list)
            c = c_list(idx);
            alpha = alpha_list(alpha_idx);
            tic;
            % rng(k);
            [u,b, sse_vals] = BarronLossFunctionSGD(K_X, y, degree, iters, lrate, alpha,c,lambda);
            train_time = toc;
            yval_pred =  K_Xval*u + b;
            mse = mean((yval_pred-yval).^2);
            if final_mse>mse
                ufinal = u;
                bfinal = b;
                final_mse = mse;
                final_beta=alpha;
                finaleta=c;

                sse_vals_finals = sse_vals;
            end
        end
    end

% disp(final_mse);
ytest_pred = K_Xtest*ufinal + bfinal;
mse = mean((ytest_pred-testy).^2);
mae =  mean(abs(ytest_pred-testy));
sst = sum((testy - mean(testy)).^2);
sse  = sum((ytest_pred - testy).^2);
ssr = sum((ytest_pred - mean(testy)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(ufinal) < 0.01)*100/numel(ufinal);
figure('Name','barron loss scatter plot');
title('barron loss regression')
scatter(test, testy,'r*');
hold on;
scatter(test, ytest_pred,'b*');
fprintf("selected alpha: %f  selected c: %f\n",final_beta,finaleta);
fprintf("barron loss test mse: %f\n",mse);
fprintf("barron loss mae: %f\n",mae);
fprintf("barron loss sse/st: %f\n",m1);
fprintf("barron loss ssr/st: %f\n",m2);
fprintf("barron loss percentage of weights less than 0.01: %f\n",sparsity);
figure('Name','barron loss sse');
plot(sse_vals_finals);
result = [result; {'Barron Loss',mse,mae,m1,m2,final_beta,finaleta,sparsity,train_time}];
writecell(result, 'results.csv');
%% 

% cauchy function
% hyperparameter tuning

lrate = 10^22;
iters = 10000;
eta_list = [10^-4,10^-3,10^-2,10^-1,1,10^1,10^2,10^3,10^4];
alpha_list = [0];

ufinal=0;bfinal=0;final_beta=0;finaleta=0;
final_mse =Inf;

sse_vals_finals = 0;
final_degree=0;

for alpha_idx = 1:length(alpha_list)
    for idx = 1:length(c_list)
        c = c_list(idx);
        alpha = alpha_list(alpha_idx);
        tic;
        % K_X = GaussianKernel(x_train, degree);
        % K_Xval = GaussianKernelTest(x,xval,degree);
        % K_Xtest =  GaussianKernelTest(x,test, degree);
        % rng(k);
        [u,b, sse_vals] = BarronLossFunctionSGD(K_X, y, degree, iters, lrate, alpha,c,lambda);
        train_time = toc;
        yval_pred =  K_Xval*u + b;
        mse = mean((yval_pred-yval).^2);
        if final_mse>mse
            ufinal = u;
            bfinal = b;
            final_mse = mse;
            final_beta=alpha;
            finaleta=c;

            sse_vals_finals = sse_vals;
        end
    end
end

% disp(final_mse);
ytest_pred = K_Xtest*ufinal + bfinal;
mse = mean((ytest_pred-testy).^2);
mae =  mean(abs(ytest_pred-testy));
sst = sum((testy - mean(testy)).^2);
sse  = sum((ytest_pred - testy).^2);
ssr = sum((ytest_pred - mean(testy)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(ufinal) < 0.01)*100/numel(ufinal);
% figure('Name','cauchy loss scatter plot');
% title('cauchy loss regression')
% scatter(test, testy,'r*');
% hold on;
% scatter(test, ytest_pred,'b*');
% fprintf("selected alpha: %f  selected c: %f\n",final_beta,finaleta);
% fprintf("cauchy loss test mse: %f\n",mse);
% fprintf("cauchy loss mae: %f\n",mae);
% fprintf("cauchy loss sse/st: %f\n",m1);
% fprintf("cauchy loss ssr/st: %f\n",m2);
% fprintf("cauchy loss percentage of weights less than 0.01: %f\n",sparsity);
figure('Name','cauchy loss sse');
plot(sse_vals_finals);
result = [result; {'Cauchy Loss',mse,mae,m1,m2,final_beta,finaleta,sparsity,train_time}];
%% 

%Geman Mclure function
%hyperparameter tuning
lrate = 10^22;
%eta_list = [10^-4,10^-3,10^-2,10^-1,1,10^1,10^2,10^3,10^4];

alpha_list = [-2];

ufinal=0;bfinal=0;final_beta=0;finaleta=0;
final_mse =Inf;

sse_vals_finals = 0;
final_degree=0;

    for alpha_idx = 1:length(alpha_list)
        for idx = 1:length(c_list)
            c = c_list(idx);
            alpha = alpha_list(alpha_idx);
            tic;
            % K_X = GaussianKernel(x_train, degree);
            % K_Xval = GaussianKernelTest(x,xval,degree);
            % K_Xtest =  GaussianKernelTest(x,test, degree);
            % rng(k);
            [u,b, sse_vals] = BarronLossFunctionSGD(K_X, y, degree, iters, lrate, alpha,c,lambda);
            train_time = toc;
            yval_pred =  K_Xval*u + b;
            mse = mean((yval_pred-yval).^2);
            if final_mse>mse
                ufinal = u;
                bfinal = b;
                final_mse = mse;
                final_beta=alpha;
                finaleta=c;

                sse_vals_finals = sse_vals;
            end
        end
    end

% disp(final_mse);
ytest_pred = K_Xtest*ufinal + bfinal;
mse = mean((ytest_pred-testy).^2);
mae =  mean(abs(ytest_pred-testy));
sst = sum((testy - mean(testy)).^2);
sse  = sum((ytest_pred - testy).^2);
ssr = sum((ytest_pred - mean(testy)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(ufinal) < 0.01)*100/numel(ufinal);
% figure('Name','Geman loss scatter plot');
% title('Geman loss regression')
% scatter(test, testy,'r*');
% hold on;
% scatter(test, ytest_pred,'b*');
% fprintf("selected alpha: %f  selected c: %f\n",final_beta,finaleta);
% fprintf("Geman loss test mse: %f\n",mse);
% fprintf("Geman loss mae: %f\n",mae);
% fprintf("Geman loss sse/st: %f\n",m1);
% fprintf("Geman loss ssr/st: %f\n",m2);
% fprintf("Geman loss percentage of weights less than 0.01: %f\n",sparsity);
% figure('Name','Geman loss sse');
% plot(sse_vals_finals);
result = [result; {'Geman Mclure Loss',mse,mae,m1,m2,final_beta,finaleta,sparsity,train_time}];
%% 

%Welsch Leclerc function
%hyperparameter tuning
lrate = 10^22;
%eta_list = [10^-4,10^-3,10^-2,10^-1,1,10^1,10^2,10^3,10^4];

alpha_list = [-10000];

ufinal=0;bfinal=0;final_beta=0;finaleta=0;
final_mse =Inf;

sse_vals_finals = 0;
final_degree=0;

for alpha_idx = 1:length(alpha_list)
    for idx = 1:length(c_list)
        c = c_list(idx);
        alpha = alpha_list(alpha_idx);
        tic;
        % K_X = GaussianKernel(x_train, degree);
        % K_Xval = GaussianKernelTest(x,xval,degree);
        % K_Xtest =  GaussianKernelTest(x,test, degree);
        % rng(k);
        [u,b, sse_vals] = BarronLossFunctionSGD(K_X, y, degree, iters, lrate, alpha,c,lambda);
        train_time = toc;
        yval_pred =  K_Xval*u + b;
        mse = mean((yval_pred-yval).^2);
        if final_mse>mse
            ufinal = u;
            bfinal = b;
            final_mse = mse;                    
            final_beta=alpha;
            finaleta=c;

            sse_vals_finals = sse_vals;
        end
    end
end

% disp(final_mse);
ytest_pred = K_Xtest*ufinal + bfinal;
mse = mean((ytest_pred-testy).^2);
mae =  mean(abs(ytest_pred-testy));
sst = sum((testy - mean(testy)).^2);
sse  = sum((ytest_pred - testy).^2);
ssr = sum((ytest_pred - mean(testy)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(ufinal) < 0.01)*100/numel(ufinal);
% figure('Name','Welsch loss scatter plot');
% title('Welsch loss regression')
% scatter(test, testy,'r*');
% hold on;
% scatter(test, ytest_pred,'b*');
% fprintf("selected alpha: %f  selected c: %f\n",final_beta,finaleta);
% fprintf("Welsch loss test mse: %f\n",mse);
% fprintf("Welsch loss mae: %f\n",mae);
% fprintf("Welsch loss sse/st: %f\n",m1);
% fprintf("Welsch loss ssr/st: %f\n",m2);
% fprintf("Welsch loss percentage of weights less than 0.01: %f\n",sparsity);
figure('Name','Welsch loss sse');
plot(sse_vals_finals);
result = [result; {'Welsch Loss',mse,mae,m1,m2,final_beta,finaleta,sparsity,train_time}];
%% 

%Charbonnier function
%hyperparameter tuning
lrate = 10^22;
%eta_list = [10^-4,10^-3,10^-2,10^-1,1,10^1,10^2,10^3,10^4];
% c_list = [2^-4,2^-3,2^-2,2^-1,1,2^1,2^2,2^3,2^4,2^5,2^6,2^7,2^8,2^9,2^10];
alpha_list = [1];
iters = 1000;
ufinal=0;bfinal=0;final_beta=0;finaleta=0;
final_mse =Inf;

sse_vals_finals = 0;
final_degree=0;

    for alpha_idx = 1:length(alpha_list)
        for idx = 1:length(c_list)
            c = c_list(idx);
            alpha = alpha_list(alpha_idx);
            tic;
            % K_X = GaussianKernel(x_train, degree);
            % K_Xval = GaussianKernelTest(x,xval,degree);
            % K_Xtest =  GaussianKernelTest(x,test, degree);
            % rng(k);
            [u,b, sse_vals] = BarronLossFunctionSGD(K_X, y, degree, iters, lrate, alpha,c,lambda);
            train_time = toc;
            yval_pred =  K_Xval*u + b;
            mse = mean((yval_pred-yval).^2);
            if final_mse>mse
                ufinal = u;
                bfinal = b;
                final_mse = mse;
                final_beta=alpha;
                finaleta=c;

                sse_vals_finals = sse_vals;
            end
        end
    end

% disp(final_mse);
ytest_pred = K_Xtest*ufinal + bfinal;
mse = mean((ytest_pred-testy).^2);
mae =  mean(abs(ytest_pred-testy));
sst = sum((testy - mean(testy)).^2);
sse  = sum((ytest_pred - testy).^2);
ssr = sum((ytest_pred - mean(testy)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(ufinal) < 0.01)*100/numel(ufinal);
% figure('Name','Charbonnier loss scatter plot');
% title('Charbonnier loss regression')
% scatter(test, testy,'r*');
% hold on;
% scatter(test, ytest_pred,'b*');
% fprintf("selected alpha: %f  selected c: %f\n",final_beta,finaleta);
% fprintf("Charbonnier loss test mse: %f\n",mse);
% fprintf("Charbonnier loss mae: %f\n",mae);
% fprintf("Charbonnier loss sse/st: %f\n",m1);
% fprintf("Charbonnier loss ssr/st: %f\n",m2);
% fprintf("Charbonnier loss percentage of weights less than 0.01: %f\n",sparsity);
figure('Name','Charbonnier loss sse');
plot(sse_vals_finals);
result = [result; {'Charbonnier Loss',mse,mae,m1,m2,final_beta,finaleta,sparsity,train_time}];
writecell(result, 'results.csv');
%% 

% rng(k);
% 
ufinal=0;bfinal=0;final_beta=0;finaleta=0;
final_mse =100000;
tic;
[theta,b, sse_vals, gradval] = TiltedLossFunction(K_X, y, degree);
train_time = toc;
ytest_pred = K_Xtest * theta+ b*ones(size(K_Xtest,1),1);
mse = mean((ytest_pred-testy).^2);
mae =  mean(abs(ytest_pred-testy));
sst = sum((testy - mean(testy)).^2);
sse  = sum((ytest_pred - testy).^2);
ssr = sum((ytest_pred - mean(testy)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(theta) < 0.01)*100/numel(theta);
% figure('Name','tilted loss scatter plot');
% title('tilted loss regression')
% scatter(test, testy,'r*');
% hold on;
% scatter(test, ytest_pred,'b*');
fprintf("tilted loss test mse: %f\n",mse);
fprintf("tilted loss mae: %f\n",mae);
fprintf("tilted loss sse/st: %f\n",m1);
fprintf("tilted loss ssr/st: %f\n",m2);
fprintf("tilted loss percentage of weights less than 0.01: %f\n",sparsity);
figure('Name','tilted loss sse');
plot(sse_vals);
result = [result; {'Tilted Loss',mse,mae,m1,m2,final_beta,finaleta,sparsity,train_time}];

writecell(result, 'results.csv');



%Tilted LS loss
function [theta, b, sse_vals, gradval] =  TiltedLossFunction(x_train, y_train, degree)
    
    nrows = size(x_train,1);
    train_X = x_train;
    train_y = y_train;
    t = -1*10^(-11);
    lr = 0.0001;
    theta = zeros(size(train_X, 2), 1)/10;
    b = median(train_y);
    sse_vals = [];
    gradval = [];
    
    batch_len = 100;
    
    
    for j = 1:80000

        batch_idx = randperm(nrows,batch_len);
        batch_X = x_train(batch_idx,:);
        batch_Y = y_train(batch_idx,:);
        
        [grads_theta, grads_b] = compute_gradients_tilting(theta,b, batch_X, batch_Y, t);
        
        % if norm(grads_theta, 2) < 1e-10
        %     break;
        % end
        
        theta = theta - lr * grads_theta;
        b = b - lr*grads_b;
        
        
        loss = mean((batch_X * theta + b*ones(size(batch_X,1),1) - batch_Y).^2);
        % test_loss = mean((test * theta + b*ones(size(test,1),1) - testy).^2);
        % test_loss_vals = [test_loss_vals;test_loss];
        sse_vals = [sse_vals;loss];
        gradval = [gradval;norm(grads_theta,2)];
        
        % lr = lr/1.0000001;
        
    end
end  

function [grad,b] = compute_gradients_tilting(theta,b, X, y, t)
    
    flag = 0;
    loss = (X * theta + b*ones(size(X,1),1) - y).^2; %+ 0.0001*norm(theta,2);
    
    % Adjust loss if t > 0
    if t > 0
        max_l = max(loss);
        loss = loss - max_l;
    end
    
    % Compute gradient
    exp_loss_t = exp(loss * t) ;  %added 1e-8 to avoid division by zero
    if sum(exp_loss_t) == 0
        exp_loss_t = exp_loss_t + 1e-300*ones(size(exp_loss_t));
        flag = 1;
    end
    grad = X' * (exp_loss_t .* (X * theta+b*ones(size(X,1),1) - y));%/ numel(y);
    b = sum(exp_loss_t .* (X * theta+b*ones(size(X,1),1) - y));%/ numel(y);
    
    % Compute normalization factor ZZ
    ZZ = sum(exp_loss_t);
    
    % Normalize the gradient
    grad = grad / ZZ;%+ 0.0001*norm(theta,2);
    b = b/ZZ;
    %grad = grad';
end
%% 