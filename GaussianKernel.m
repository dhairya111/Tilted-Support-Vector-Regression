
function [K_X] =  GaussianKernel(X_train, degree)
    data_len = length(X_train);
    K_X = zeros(data_len,data_len);
    for i = 1:data_len
        for j = 1:data_len
            K_X(i,j) = svkernel('rbf',X_train(i),X_train(j),degree);

        end
    end
end




