 function [K_X] =  GaussianKernelTest(X_train,test, degree)
    data_len = size(X_train,1);
    test_len = size(test,1);
    K_X = zeros(test_len,data_len);
    for i = 1:test_len
        for j = 1:data_len
            K_X(i,j) = svkernel('rbf',test(i),X_train(j),degree);

        end
    end
end
