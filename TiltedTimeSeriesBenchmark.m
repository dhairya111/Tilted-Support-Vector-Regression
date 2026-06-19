close all;
clear all;

% Set random seed
rng(42);

% ---------------------------
% 1. Load Time Series
% ---------------------------
data = readmatrix("/MATLAB Drive/Datasets/kanyakumari100m.csv");

% split into train test
x = data(:, 1:end-1);
y = data(:, end);

y = normalize(y);

% Inject outliers
T = length(y);
split = floor(0.7 * T);

percentOutliers = 0.1;
trainset_len = T - floor(0.3 * T);
num_outliers = floor(percentOutliers*trainset_len);
outlier_indices = randperm(trainset_len-40, num_outliers) + 20; % from 21 to T-20
outlier_indices = outlier_indices';
% randomly +4 or -4
% signs = datasample([4, -4], num_outliers);


% figure('Name',"Data before outliers");
% plot(x,y);

% gaussian outlier 
% signs = abs(normrnd(0,10,size(outlier_indices)));
signs = normrnd(0,7,size(outlier_indices));
y(outlier_indices) = y(outlier_indices) + signs;

figure('Name',"Data");
plot(x,y);


% grid on;

% ---------------------------
% 2. Prepare Features
% ---------------------------
window_size = 20;
X = zeros(T-window_size,window_size);
Y = zeros(T-window_size,1);

for i = window_size+1:T
    X(i-window_size,:) = y(i-window_size:i-1)';
    Y(i-window_size,:) = y(i)';
end

% X = normalize(X);

% Train/Test split
X_train = X(1:split, :);
X_test  = X(split+1:end, :);
Y_train = Y(1:split);
Y_test  = Y(split+1:end);



% ---------------------------
% 3. RBF Kernel
% ---------------------------
degree = 2;
% lambda = 0.0000001;
lambda = 0;

% Compute RBF Kernel

K_X = GaussianKernel(X_train, degree);
K_Xtest =  GaussianKernelTest(X_train, X_test, degree);


beta_max = max(abs(Y_train));
beta_min = abs(median(Y_train));

result = {'loss type','mse', 'mae', 'sse/sst', 'ssr/sst', 'alpha', 'c', '% of weights<0.01', 'train time'};

%% 

%kernel regression
lrate = 0.001;
iters = 100000;
% rng(k);
tic;
[u,b] = KernelRegressionSGD(K_X, Y_train, degree,iters,lrate);
train_time = toc;
ytest_pred = K_Xtest * u+b;
mse = mean((ytest_pred-Y_test).^2);
mae =  mean(abs(ytest_pred-Y_test));
sst = sum((Y_test - mean(Y_test)).^2);
sse  = sum((ytest_pred - Y_test).^2);
ssr = sum((ytest_pred - mean(Y_test)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(u) < 0.01)*100/numel(u);
result = [result; {'least squares loss',mse,mae,m1,m2,'-','-',sparsity,train_time}];

figure('Name','test vals');
plot(Y_test);
hold on;
plot(ytest_pred);
hold off;

saveas(gcf, 'lssq_pred.png');

writematrix(u,'u_leastsq.csv');
writematrix(b,'b_leastsq.csv');

%% 

%truncated loss
%rng(k);
lrate = 0.1;
iters = 1000;
%Truncated LS function
trunc_tuned_mse = Inf;
trunc_beta = 0;
final_u = 0;final_b=0;
sse_vals_finals = 0;
final_deg = 0;
final_lambda =  0;
final_lr = 0;
%tuning beta

for beta = linspace(beta_min,beta_max,15)
    tic;
    % K_X = GaussianKernel(x_train, degree);
    % K_Xval = GaussianKernelTest(x,xval,degree);
    % K_Xtest =  GaussianKernelTest(x,test, degree);

     [u,b, sse_val, gradval] = TruncatedLSLossFunctionSGD(K_X, Y_train, degree, iters, lrate, beta,lambda);
   % [u,b, sse_val, gradval] = TruncatedLossWithoutKernels(K_X, y, degree, iters, lrate, beta,lambda);
    train_time = toc;
        ytest_pred =  K_Xtest*u + b;
        mse = mean((ytest_pred-Y_test).^2);
        if trunc_tuned_mse>mse
            trunc_tuned_mse = mse;
            trunc_beta=beta;
            final_u = u;
            final_b = b;
            sse_vals_finals = sse_val;
            final_deg = degree;
            final_lambda =  lambda;
            final_lr = lrate;
        end
end  

ytrain_pred = K_X*final_u + final_b;
ytest_pred = K_Xtest*final_u + final_b;
mse = mean((ytest_pred-Y_test).^2);
mae =  mean(abs(ytest_pred-Y_test));
sst = sum((Y_test - mean(Y_test)).^2);
sse  = sum((ytest_pred - Y_test).^2);
ssr = sum((ytest_pred - mean(Y_test)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(final_u) < 0.01)*100/numel(final_u);
fprintf("truncated loss mse: %f\n",mse);
fprintf("truncated loss mae: %f\n",mae);
fprintf("truncated loss sse/st: %f\n",m1);
fprintf("truncated loss ssr/st: %f\n",m2);
figure('Name','truncated loss sse');
plot(sse_vals_finals);

% figure('Name','train predictins');
% plot(x(1:300,:),ytrain_pred(1:300,:));
% % hold on;
% plot(x(1:300,:),y(1:300,:));
% hold off;
result = [result; {'truncated loss',mse,mae,m1,m2,0,trunc_beta,sparsity,train_time}];
figure('Name','test vals');
plot(Y_test);
hold on;
plot(ytest_pred);
hold off;

saveas(gcf, 'trunc_pred.png');


writematrix(final_u,'u_trunc.csv');
writematrix(final_b,'b_trunc.csv');


%% 

% barron function
% hyperparameter tuning
lrate = 1;
%eta_list = [10^-4,10^-3,10^-2,10^-1,1,10^1,10^2,10^3,10^4];
c_list = linspace(beta_min,beta_max,15);
alpha_list = [-2^5,-2^4,-2^3,-2^2,-2,-1,-2^-1,-2^-2,0,2^-2,2^-1,2^0,2^1];
degree_list = [1,2,2^2,2^3,2^4,2^5,2^6];

ufinal=0;bfinal=0;final_beta=0;finaleta=0;
final_mse =Inf;
iters = 50000;
sse_vals_finals = 0;
final_degree=0;

% 
%     for alpha_idx = 1:length(alpha_list)
%         for idx = 1:length(c_list)
%             c = c_list(idx);
%             alpha = alpha_list(alpha_idx);
%             tic;
%             K_X = GaussianKernel(x_train, degree);
%             K_Xval = GaussianKernelTest(x,xval,degree);
%             K_Xtest =  GaussianKernelTest(x,test, degree);
%             % rng(k);
%             [u,b, sse_vals] = BarronLossFunctionSGD(K_X, y, degree, iters, lrate, alpha,c,lambda);
%             train_time = toc;
%             y_train_pred =  K_Xval*u + b;
%             mse = mean((y_train_pred-Y_train).^2);
%             if final_mse>mse
%                 ufinal = u;
%                 bfinal = b;
%                 final_mse = mse;
%                 final_beta=alpha;
%                 finaleta=c;
% 
%                 sse_vals_finals = sse_vals;
%             end
%         end
%     end
% 
% % disp(final_mse);
% ytest_pred = K_Xtest*ufinal + bfinal;
% mse = mean((ytest_pred-Y_test).^2);
% mae =  mean(abs(ytest_pred-Y_test));
% sst = sum((Y_test - mean(Y_test)).^2);
% sse  = sum((ytest_pred - Y_test).^2);
% ssr = sum((ytest_pred - mean(Y_test)).^2);
% m1 = sse/sst;
% m2 = ssr/sst;
% sparsity = sum(abs(ufinal) < 0.01)*100/numel(ufinal);
% % figure('Name','barron loss scatter plot');
% % title('barron loss regression')
% % scatter(test, Y_test,'r*');
% % hold on;
% % scatter(test, ytest_pred,'b*');
% fprintf("selected alpha: %f  selected c: %f\n",final_beta,finaleta);
% fprintf("barron loss test mse: %f\n",mse);
% fprintf("barron loss mae: %f\n",mae);
% fprintf("barron loss sse/st: %f\n",m1);
% fprintf("barron loss ssr/st: %f\n",m2);
% fprintf("barron loss percentage of weights less than 0.01: %f\n",sparsity);
% figure('Name','barron loss sse');
% plot(sse_vals_finals);
% result = [result; mse,mae,m1,m2,final_beta,finaleta,sparsity,train_time];

%% 

% cauchy function
% hyperparameter tuning

lrate = 1;
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
        [u,b, sse_vals] = BarronLossFunctionSGD(K_X, Y_train, degree, iters, lrate, alpha,c,lambda);
        train_time = toc;
        y_test_pred =  K_Xtest*u + b;
        mse = mean((y_test_pred-Y_test).^2);
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
mse = mean((ytest_pred-Y_test).^2);
mae =  mean(abs(ytest_pred-Y_test));
sst = sum((Y_test - mean(Y_test)).^2);
sse  = sum((ytest_pred - Y_test).^2);
ssr = sum((ytest_pred - mean(Y_test)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(ufinal) < 0.01)*100/numel(ufinal);
% figure('Name','cauchy loss scatter plot');
% title('cauchy loss regression')
% scatter(test, Y_test,'r*');
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

figure('Name','cauchy test vals');
plot(Y_test);
hold on;
plot(ytest_pred);
hold off;

saveas(gcf, 'cauchy_pred.png');

writematrix(ufinal,'u_cauchy.csv');
writematrix(bfinal,'b_cauchy.csv');

%% 

%Geman Mclure function
%hyperparameter tuning
lrate = 1;
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
            % K_X = GaussianKernelTest(x,xval,degree);
            % K_Xtest =  GaussianKernelTest(x,test, degree);
            % rng(k);
            [u,b, sse_vals] = BarronLossFunctionSGD(K_X, Y_train, degree, iters, lrate, alpha,c,lambda);
            train_time = toc;
            y_test_pred =  K_Xtest*u + b;
            mse = mean((y_test_pred-Y_test).^2);
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
mse = mean((ytest_pred-Y_test).^2);
mae =  mean(abs(ytest_pred-Y_test));
sst = sum((Y_test - mean(Y_test)).^2);
sse  = sum((ytest_pred - Y_test).^2);
ssr = sum((ytest_pred - mean(Y_test)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(ufinal) < 0.01)*100/numel(ufinal);
% figure('Name','Geman loss scatter plot');
% title('Geman loss regression')
% scatter(test, Y_test,'r*');
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
result = [result; {'Geman Loss',mse,mae,m1,m2,final_beta,finaleta,sparsity,train_time}];


figure('Name','geman test vals');
plot(Y_test);
hold on;
plot(ytest_pred);
hold off;

saveas(gcf, 'geman_pred.png');

writematrix(ufinal,'u_gemanmclure.csv');
writematrix(bfinal,'b_gemanmclure.csv');

%% 

%Welsch Leclerc function
%hyperparameter tuning
lrate = 1;
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
        % K_X = GaussianKernelTest(x,xval,degree);
        % K_Xtest =  GaussianKernelTest(x,test, degree);
        % rng(k);
        [u,b, sse_vals] = BarronLossFunctionSGD(K_X, Y_train, degree, iters, lrate, alpha,c,lambda);
        train_time = toc;
        y_test_pred =  K_Xtest*u + b;
        mse = mean((y_test_pred-Y_test).^2);
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
ytrain_pred = K_X*ufinal + bfinal;
mse = mean((ytest_pred-Y_test).^2);
mae =  mean(abs(ytest_pred-Y_test));
sst = sum((Y_test - mean(Y_test)).^2);
sse  = sum((ytest_pred - Y_test).^2);
ssr = sum((ytest_pred - mean(Y_test)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(ufinal) < 0.01)*100/numel(ufinal);
% figure('Name','Welsch loss scatter plot');
% title('Welsch loss regression')
% scatter(test, Y_test,'r*');
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


figure('Name','welsch test vals');
plot(Y_test);
hold on;
plot(ytest_pred);
hold off;

saveas(gcf, 'welsch_pred.png');

writematrix(ufinal,'u_welsch.csv');
writematrix(bfinal,'b_welsch.csv');
%% 

%Charbonnier function
%hyperparameter tuning
lrate = 0.1;
%eta_list = [10^-4,10^-3,10^-2,10^-1,1,10^1,10^2,10^3,10^4];
% c_list = [2^-4,2^-3,2^-2,2^-1,1,2^1,2^2,2^3,2^4,2^5,2^6,2^7,2^8,2^9,2^10];
alpha_list = [1];

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
            [u,b, sse_vals] = BarronLossFunctionSGD(K_X, Y_train, degree, iters, lrate, alpha,c,lambda);
            train_time = toc;
            y_test_pred =  K_Xtest*u + b;
            mse = mean((y_test_pred-Y_test).^2);
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
mse = mean((ytest_pred-Y_test).^2);
mae =  mean(abs(ytest_pred-Y_test));
sst = sum((Y_test - mean(Y_test)).^2);
sse  = sum((ytest_pred - Y_test).^2);
ssr = sum((ytest_pred - mean(Y_test)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(ufinal) < 0.01)*100/numel(ufinal);


figure('Name','charbo test vals');
plot(Y_test);
hold on;
plot(ytest_pred);
hold off;

saveas(gcf, 'charbo_pred.png');

writematrix(ufinal,'u_charbonnier.csv');
writematrix(bfinal,'b_charbonnier.csv');

% figure('Name','Charbonnier loss scatter plot');
% title('Charbonnier loss regression')
% scatter(test, Y_test,'r*');
% hold on;
% scatter(test, ytest_pred,'b*');
% fprintf("selected alpha: %f  selected c: %f\n",final_beta,finaleta);
% fprintf("Charbonnier loss test mse: %f\n",mse);
% fprintf("Charbonnier loss mae: %f\n",mae);
% fprintf("Charbonnier loss sse/st: %f\n",m1);
% fprintf("Charbonnier loss ssr/st: %f\n",m2);
% fprintf("Charbonnier loss percentage of weights less than 0.01: %f\n",sparsity);
% figure('Name','Charbonnier loss sse');
% plot(sse_vals_finals);
result = [result; {'Charbonnier Loss',mse,mae,m1,m2,final_beta,finaleta,sparsity,train_time}];



%% 

% rng(k);
% 
ufinal=0;bfinal=0;final_beta=0;finaleta=0;
final_mse =100000;
tic;
[theta, b, sse_vals, gradval] = TiltedLossFunction(K_X, Y_train, degree);
train_time = toc;
ytest_pred = K_Xtest * theta+ b*ones(size(K_Xtest,1),1);
mse = mean((ytest_pred-Y_test).^2);
mae =  mean(abs(ytest_pred-Y_test));
sst = sum((Y_test - mean(Y_test)).^2);
sse  = sum((ytest_pred - Y_test).^2);
ssr = sum((ytest_pred - mean(Y_test)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(theta) < 0.01)*100/numel(theta);
% figure('Name','tilted loss scatter plot');
% title('tilted loss regression')
% scatter(test, Y_test,'r*');
% hold on;
% scatter(test, ytest_pred,'b*');
fprintf("tilted loss test mse: %f\n",mse);
fprintf("tilted loss mae: %f\n",mae);
fprintf("tilted loss sse/st: %f\n",m1);
fprintf("tilted loss ssr/st: %f\n",m2);
fprintf("tilted loss percentage of weights less than 0.01: %f\n",sparsity);
figure('Name','tilted loss sse');
plot(sse_vals);
result = [result;{'Tilted Loss', mse,mae,m1,m2,final_beta,finaleta,sparsity,train_time}];


figure('Name','tilted test vals');
plot(Y_test);
hold on;
plot(ytest_pred);
hold off;

saveas(gcf, 'tilted_pred.png');

writecell(result, 'results.csv');

writematrix(theta,'u_tilt.csv');
writematrix(b,'b_tilt.csv');

%Tilted LS loss
function [theta, b, sse_vals, gradval] =  TiltedLossFunction(x_train, y_train, degree)
    
    nrows = size(x_train,1);
    train_X = x_train;
    train_y = y_train;
    t = -0.1;
    lr = 0.0001;
    theta = zeros(size(train_X, 2), 1)/10;
    b = median(train_y);
    
    sse_vals = [];
    gradval = [];
    
    batch_len = 50;
    
    
    for j = 1:100000
        % 
        % batch_idx = randperm(nrows,batch_len);
        % batch_X = x_train(batch_idx,:);
        % batch_Y = y_train(batch_idx,:);
        
        % for time series
        % if(j==1)
        batch_idx_start = randi(nrows - batch_len);
        % end
        batch_idx_end = batch_idx_start + batch_len; 
        batch_X = x_train(batch_idx_start:batch_idx_end,:);
        batch_Y = y_train(batch_idx_start:batch_idx_end,:);    


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
        
         % lr = lr/1.00001;
        
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