%Barron LS loss
function [u,b, sse_vals, gradval] =  BarronLossFunctionSGD(x_train, y_train, degree, iters, lrate, alpha,c,lambda)
    
    ncols = size(x_train,2);
    nrows = size(x_train,1);
    % u = rand(size(x_train,2),1)/10;
    u = zeros(size(x_train,2),1);
    b = zeros;
    %K_X = GaussianKernel(x_train, degree);
    K_X = x_train;
    sse_vals = zeros(iters,1);
    
    
    batch_len = 208;

    batch_X = zeros(batch_len,ncols);
    batch_Y = zeros(batch_len,1);
    grad_dw = zeros(batch_len,ncols);
    grad_db = zeros(batch_len,1);


    median_val  = abs(median(y_train));

    for i = 1:iters
        % 
        % batch_idx = randperm(nrows, batch_len);  
        % batch_X = K_X(batch_idx,:);
        % batch_Y = y_train(batch_idx,:);

        % for time series
        batch_idx_start = randi(nrows - batch_len);
        batch_idx_end = batch_idx_start + batch_len; 
        batch_X = K_X(batch_idx_start:batch_idx_end,:);
        batch_Y = y_train(batch_idx_start:batch_idx_end,:);    


        y_pred = batch_X*u + b;
        v = batch_Y - y_pred;
        
        
        if alpha == 2
            dw = (1/nrows)*(batch_X'*v./(c.^2) );
            db = (1/nrows)*sum(v./(c.^2));
        elseif alpha == 0
            v = (2*v)./((v.^2)+2*c^2);
            dw = (1/nrows)*(batch_X'*v./(c.^2) );
            db = (1/nrows)*sum(v./(c.^2));
        else
            v = (v./c^2).*((((v./c).^2)./abs(alpha-2))+1).^(alpha/2 - 1);
            dw = (1/nrows)*(batch_X'*v./(c.^2) );
            db = (1/nrows)*sum(v./(c.^2));
        end

        u = u + lrate*(dw + lambda.*u);
        b = b + lrate*(db + lambda.*b);
        sse_vals(i) = sum((y_pred-batch_Y).^2);
        %gradval(i)= norm([mean(dw,2);mean(db,1)]);
        
        % if(i>2)
        % if(sqrt(sse(i)) - sqrt(sse(i-1)) < median_val*0.001)
        %     break;
        % end
        % end
        lrate= lrate/(1+0.0001);
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
     
end 