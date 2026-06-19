function [u, b, sse] =  KernelRegressionSGD(K_X, y_train, degree, iters, lrate)
    data_len = size(K_X,2);
    ncols = size(K_X,2);
    nrows = size(K_X,1);
    u = rand(ncols,1);
    b = rand();
    % K_X = GaussianKernel(x_train, degree);
    sse = zeros(iters,1);
    gradval =  zeros(iters,1);
    lambda = 0;


    batch_len = 150;

    batch_X = zeros(batch_len,ncols);
    batch_Y = zeros(batch_len,1);
    grad_dw = zeros(batch_len,ncols);
    grad_db = zeros(batch_len,1);


    median_val  = abs(median(y_train));
    
    for i = 1:iters

        % for time series
        batch_idx_start = randi(nrows - batch_len);
        batch_idx_end = batch_idx_start + batch_len; 
        batch_X = K_X(batch_idx_start:batch_idx_end,:);
        batch_Y = y_train(batch_idx_start:batch_idx_end,:); 

        % batch_X = K_X(batch_idx,:);
        % batch_Y = y_train(batch_idx,:);

        y_pred = batch_X*u + b;
        
        dw = (1/data_len)*(batch_X'*(y_pred-batch_Y) + lambda.*u);
        db = (1/data_len)*sum((y_pred-batch_Y));
        
        u = u - lrate*dw;
        b = b - lrate*db;
        sse(i) = sum((y_pred-batch_Y).^2);
        gradval(i)= norm(dw);
        
        % if(i>2)
        % if(abs(sse(i) - sse(i-1)) < median_val*0.001)
        %     break;
        % end
        % end
        lrate= lrate/(1+0.0001);
    end
    
    % figure('Name','standard loss sse');
    % plot(sse);
    % figure('Name','standard loss gradval');
    % plot(gradval);
    % figure('Name','standard loss scatter plot');
    % title('standard regression')
    % scatter(x_train, y_train,'r*');
    % hold on;
    % scatter(x_train, y_pred,'b*');
    mse = mean((y_pred-batch_Y).^2);
    %fprintf("training mse standard: %f",mse);
    % fprintf("last iter: %f",i);
     
end 