function [u,b] =  KernelRegression(K_X, y_train, degree, iters, lrate)
    data_len = size(K_X,2);
    u = rand(data_len,1);
    b = rand();
    % K_X = GaussianKernel(x_train, degree);
    sse = zeros(iters,1);
    gradval =  zeros(iters,1);
    lambda = 0;
    
    for i = 1:iters
        y_pred = K_X*u + b;
        
        dw = (1/data_len)*(K_X'*(y_pred-y_train) + lambda.*u);
        db = (1/data_len)*sum((y_pred-y_train));
        
        u = u - lrate*dw;
        b = b - lrate*db;
        sse(i) = sum((y_pred-y_train).^2);
        gradval(i)= norm(dw);
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
    mse = mean((y_pred-y_train).^2);
    %fprintf("training mse standard: %f",mse);
     
end 