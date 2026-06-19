function [u,b, sse, gradval] =  TruncatedLSLossFunctionSGD(K_X, y_train, degree, iters, lrate, beta,lambda)
    nrows = size(K_X,1);
    ncols = size(K_X,2);
    % u = rand(ncols,1)/10;
    u = zeros(ncols,1);
    b = zeros;
    %K_X = GaussianKernel(x_train, degree);
    sse = zeros(iters,1);
    %lambda = 0;
    batch_len = 208;

    batch_X = zeros(batch_len,ncols);
    batch_Y = zeros(batch_len,1);
    grad_dw = zeros(batch_len,ncols);
    grad_db = zeros(batch_len,1); 
    
    median_val  = abs(median(y_train));
    
    for i = 1:iters
    
        % batch_idx = randperm(nrows, batch_len); 
        
        % for time series
        batch_idx_start = randi(nrows - batch_len);
        batch_idx_end = batch_idx_start + batch_len; 
        batch_X = K_X(batch_idx_start:batch_idx_end,:);
        batch_Y = y_train(batch_idx_start:batch_idx_end,:);    

        grad_dw(:) = 0;
        grad_db(:) = 0;
        
        y_pred = batch_X*u + b;
        v = batch_Y - y_pred;
        
        idx = ( v > -beta & v < beta);
        
        grad_dw(idx, :) = -v(idx).*batch_X(idx,:);
        grad_db(idx) = -v(idx);
        
        u = u - lrate*(mean(grad_dw,1)' + lambda.*u);
        b = b - lrate*mean(grad_db,1);
        sse(i) = sum((y_pred-batch_Y).^2);
        gradval(i)= norm([mean(grad_dw,1)';mean(grad_db,1)]);
        
        % if(i>2)
        % if(sqrt(sse(i)) - sqrt(sse(i-1)) < median_val*0.001)
        %     break;
        % end
        % end
        % 
        % if(mod(i,1000)==0)
        % lrate = lrate / (1 + 1);
        % end
        %lrate= lrate/(1+0.001);
    end
    
%     figure('Name','truncated loss sse');
%     plot(sse);
%     figure('Name','truncated loss gradval');
%     plot(gradval)
%     figure('Name','truncated loss scatter');
%     scatter(x_train, y_train,'r*');
%     hold on;
%     scatter(x_train, y_pred,'b*');
%     mse = sum((y_pred-y_train).^2)/data_len;
%     disp(mse);
     % fprintf("last iter: %f",i);
end