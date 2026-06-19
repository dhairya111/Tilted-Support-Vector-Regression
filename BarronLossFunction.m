%Barron LS loss
function [u,b, sse_vals, gradval] =  BarronLossFunction(x_train, y_train, degree, iters, lrate, alpha,c,lambda)
    data_len = size(x_train,2);
    u = rand(size(x_train,2),1)/10;
    b = zeros;
    %K_X = GaussianKernel(x_train, degree);
    K_X = x_train;
    sse_vals = zeros(iters,1);
    
    
    for i = 1:iters
        grad_dw = zeros(data_len,data_len);
        grad_db = zeros(data_len,1);
        
        y_pred = K_X*u + b;
        v = y_train - y_pred;
        
        
        if alpha == 2
            dw = (1/data_len)*(K_X'*v./(c.^2) );
            db = (1/data_len)*sum(v./(c.^2));
        elseif alpha == 0
            v = (2*v)./((v.^2)+2*c^2);
            dw = (1/data_len)*(K_X'*v./(c.^2) );
            db = (1/data_len)*sum(v./(c.^2));
        else
            v = (v./c^2).*((((v./c).^2)./abs(alpha-2))+1).^(alpha/2 - 1);
            dw = (1/data_len)*(K_X'*v./(c.^2) );
            db = (1/data_len)*sum(v./(c.^2));
        end

        u = u + lrate*(dw+ lambda.*u);
        b = b + lrate*(db+lambda.*b);
        sse_vals(i) = sum((y_pred-y_train).^2);
        gradval(i)= norm([mean(dw,2);mean(db,1)]);
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