function [u,b, sse, gradval] =  TruncatedLSLossFunction(K_X, y_train, degree, iters, lrate, beta,lambda)
    data_len = size(K_X,2);
    u = rand(data_len,1)/10;
    b = zeros;
    %K_X = GaussianKernel(x_train, degree);
    sse = zeros(iters,1);
    %lambda = 0;
    
    for i = 1:iters
        grad_dw = zeros(data_len,data_len);
        grad_db = zeros(data_len,1);
        
        y_pred = K_X*u + b;
        v = y_train - y_pred;
        
        idx = ( v > -beta & v < beta);
        
        
        for ii=1:length(y_train)
            
            if (idx(ii)==0)
                grad_dw(ii,:)= zeros(length(y_train),1);
                grad_db(ii) =0;
            else
                grad_dw(ii,:) = -v(ii)*K_X(:,ii);
                grad_db(ii) = -v(ii);
            end
        end
        u = u - lrate*(mean(grad_dw,2)+ lambda.*u);
        b = b - lrate*mean(grad_db,1);
        sse(i) = sum((y_pred-y_train).^2);
        gradval(i)= norm([mean(grad_dw,2);mean(grad_db,1)]);
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
     
end