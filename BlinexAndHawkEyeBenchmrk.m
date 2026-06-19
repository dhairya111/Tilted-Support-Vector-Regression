clear all;
close all;


data = readmatrix("/MATLAB Drive/Datasets/servo.csv");


% split into train test
x = data(:, 1:end-1);
y = data(:, end);

% x = normalize(x); 


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


% determine amount of outliers
percentOutliers = 0.2;

n = size(x_train, 1);  % Total number of data points
numPoints = round(n * percentOutliers/2);  % 10% of data points

% Randomly select indices
% rng(1);  % Set seed for reproducibility
randomIndices1 = randperm(n, numPoints);
randomIndices2 = randperm(n, numPoints);

% Multiply half of selected points by 5 and other half by -5
% y(randomIndices1) = y(randomIndices1) + unifrnd(-15*std(y),15*std(y),[length(randomIndices1),1]);
% % y(randomIndices2) = y(randomIndices2) + unifrnd(-15*std(y),15*std(y),[length(randomIndices2),1]);
y_train(randomIndices1) = abs(y_train(randomIndices1) *5);
y_train(randomIndices2) = -1*abs(y_train(randomIndices2) *5);
% y(randomIndices1) = y(randomIndices1) + abs(normrnd(0,7,[length(randomIndices1),1]));
% y(randomIndices2) = y(randomIndices2) + abs(normrnd(0,7,[length(randomIndices2),1]));
% y(randomIndices1) = normrnd(5,5,size(y(randomIndices1)));
% y(randomIndices2) = normrnd(5,5,size(y(randomIndices1)));

%scatter(x,y);

beta_max = max(abs(y));
beta_min = median(abs(y));


result = {'loss type','mse', 'mae', 'sse/sst', 'ssr/sst', '% of weights<0.01', 'train time'};
    

degree = 0.5;

% 
K_X = GaussianKernel(x_train, degree);
K_Xval = GaussianKernelTest(x_train,xval,degree);
K_Xtest =  GaussianKernelTest(x_train,test, degree);
% K_X = x_train;
% K_Xval = xval;
% K_Xtest = test;
iters = 1000;
lrate = 0.000001;
lambda = 0.0001;

a=0.05;
b=100;     
k=20;

 
tic;
[u, bias, sse_vals] = BlinexRegression(K_X, K_Xtest, y_train,testy, degree, iters, lrate, a, b, k, lambda);
BlinexTrainTime = toc;
ytest_pred =  K_Xtest*u + bias;
mse = mean((ytest_pred-testy).^2);
mae =  mean(abs(ytest_pred-testy));
sst = sum((testy - mean(testy)).^2);
sse = sum((ytest_pred - testy).^2);
ssr = sum((ytest_pred - mean(testy)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(u) < 0.01)*100/numel(u);
disp('blinex loss mse');
display(mse);

figure('Name','blinex loss sse');
plot(sse_vals);

result = [result; {'blinex Loss',mse,mae,m1,m2,sparsity,BlinexTrainTime}];
hyperparams_blinex = {'degree',degree;'lambda',lambda;'lrate',lrate;'iters',iters;'a',a;'b',b;'k ',k};

a1 = 1;
a2 = 5;
e = 0.08;
lrate = 1e-04;
iters = 1000;
tic;
[u, bias, sse_vals] = HawkEyeRegression(K_X, K_Xtest, y_train, testy, iters, lrate, a1, a2, e, lambda);
HawkeyeTrainTime = toc;
ytest_pred =  K_Xtest*u + bias;
mse = mean((ytest_pred-testy).^2);
mae =  mean(abs(ytest_pred-testy));
sst = sum((testy - mean(testy)).^2);
sse  = sum((ytest_pred - testy).^2);
ssr = sum((ytest_pred - mean(testy)).^2);
m1 = sse/sst;
m2 = ssr/sst;
sparsity = sum(abs(u) < 0.01)*100/numel(u);
disp('hawkeye loss mse');
display(mse);

figure('Name','hawkeye loss sse');
plot(sse_vals);

result = [result; {'hawkeye Loss',mse,mae,m1,m2,sparsity,HawkeyeTrainTime}];
hyperparams_hawkeye = {'degree',degree;'lambda',lambda;'lrate',lrate;'iters',iters;'a1',a1;'a2',a2;'e',e};

function [u, bias, sse] =  BlinexRegression(K_X, K_Xtest, y_train,testy, degree, iters, lrate, a, b, k, lambda)
    data_len = size(K_X,2);
    ncols = size(K_X,2);
    nrows = size(K_X,1);

    u = zeros(data_len,1);
    bias=median(y_train);
    
    sse = zeros(iters,1);
    sse_test = zeros(iters,1);
    gradval =  zeros(iters,1);
    
    batch_len = floor(nrows-1) + 1;

    batch_X = zeros(batch_len,ncols);
    batch_Y = zeros(batch_len,1);
    grad_dw = zeros(batch_len,ncols);
    grad_db = zeros(batch_len,1); 
    
    
    for j = 1:iters

        

        batch_idx_start = randi([0,nrows - batch_len]);
        if batch_idx_start <= 0
            batch_idx_start  = 1;
        end
        batch_idx_end = batch_idx_start + batch_len - 1; 
        batch_X = K_X(batch_idx_start:batch_idx_end,:);
        %batch_Xtest = K_Xtest(batch_idx_start:batch_idx_end,:);
        batch_Y = y_train(batch_idx_start:batch_idx_end,:);

        y_pred = batch_X*u + bias;
        r = y_pred-batch_Y;

        % Derivate of loss
        
        dw = k*batch_X'*a*b*((exp(a*r) - 1)./(1+b*(exp(a*r) - a*r -1)).^2);
        db = k*a*b*(exp(a*r) - 1)./(1+b*(exp(a*r) - a*r -1)).^2;
        
        u = u - lrate*(dw+ lambda.*u);
        bias = bias - lrate*(sum(db));

        y_pred_test = K_Xtest*u + bias;

        sse(j) = sum((y_pred-batch_Y).^2);
        sse_test(j) = sum((y_pred_test-testy).^2);
        gradval(j)= norm(dw);

        
    end

    
    figure('Name','blinex sse test set');
    plot(sse_test);
    % figure('Name','standard loss scatter plot');
    % title('standard regression')
    % scatter(x_train, y_train,'r*');
    % hold on;
    % scatter(x_train, y_pred,'b*');
    % mse = mean((y_pred-y_train).^2);
    % fprintf("training mse standard: %f",mse);
     
end



function [u, bias, sse] =  HawkEyeRegression(K_X, K_Xtest, y_train, testy, iters, lrate, a1, a2, e, lambda)
    data_len = size(K_X,2);
    ncols = size(K_X,2);
    nrows = size(K_X,1);

    u = zeros(data_len,1)/100;
    bias = median(y_train);
    
    sse = zeros(iters,1);
    gradval =  zeros(iters,1);
    sse_test = zeros(iters,1);

    batch_len = floor(nrows-1)+1;

    batch_X = zeros(batch_len,ncols);
    batch_Y = zeros(batch_len,1);
    grad_dw = zeros(batch_len,ncols);
    grad_db = zeros(batch_len,1); 
    
    
    
    for j = 1:iters
        batch_idx_start = randi([0,nrows - batch_len]);
        if batch_idx_start <= 0
            batch_idx_start  = 1;
        end
        batch_idx_end = batch_idx_start + batch_len - 1; 
        batch_X = K_X(batch_idx_start:batch_idx_end,:);
        %batch_Xtest = K_Xtest(batch_idx_start:batch_idx_end,:);
        batch_Y = y_train(batch_idx_start:batch_idx_end,:);

        y_pred = batch_X*u + bias;
        r = y_pred - batch_Y;

        % Derivate of loss
        dw=zeros(batch_len,data_len);
        db=zeros(batch_len,1);
        for i=1:batch_len
            if r(i)>= e                    % Here e is epsilon
                dw(i,:)= -a2*a1^2*(r(i)-e)*exp(-a1*(r(i)-e))*batch_X(i,:)';
                db(i) = -a2*a1^2*(r(i)-e)*exp(-a1*(r(i)-e));
            elseif r(i) > -e && r(i) < e
                dw(i,:)= zeros(1,data_len);
                db(i) = zeros;
            elseif r(i) <= -e
                dw(i,:)= -a2*a1^2*(r(i)+e)*exp(a1*(r(i)+e))*batch_X(i,:)';
                db(i)= -a2*a1^2*(r(i)+e)*exp(a1*(r(i)+e));
            end
        end

        
        u = u + lrate*(sum(dw,1)' + lambda.*u);
        bias = bias + lrate*(sum(db));
        sse(j) = sum((y_pred-batch_Y).^2);
        y_pred_test = K_Xtest*u + bias;
        sse_test(j) = sum((y_pred_test-testy).^2);
        gradval(j)= norm(dw);
        lrate= lrate/(1+0.0001);
    end

    
    % figure('Name','standard loss gradval');
    % plot(gradval);
    % figure('Name','standard loss scatter plot');
    % title('standard regression')
    % scatter(x_train, y_train,'r*');
    % hold on;
    % scatter(x_train, y_pred,'b*');
    %mse = mean((y_pred-y_train).^2);
    %fprintf("training mse standard: %f",mse);
    figure('Name','hawkeye sse test set');
    plot(sse_test);
     
end

