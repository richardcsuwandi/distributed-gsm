function [pMean, pVar] = prediction(xtrain,xtest,ytrain,nTest,alpha,nv,freq,var,K)
%
Q = length(freq);
nTrain = length(xtrain);
xtest = xtest(1:nTest);

K_cross_set = kernelComponent(freq,var,xtest,xtrain);
K_test_set = kernelComponent(freq,var,xtest,xtest);

    function K_sum = sumup_kernel(K, Q, alpha)
        K_sum = 0;
        for i = 1:Q
            K_sum = K_sum + alpha(i)*K{i};
        end
    end

% sumup sub-Kernels sets
K_cross = sumup_kernel(K_cross_set, Q, alpha);
K_test = sumup_kernel(K_test_set, Q, alpha);
K_train = sumup_kernel(K, Q, alpha);
% prediction phase
K_inv = pinv(K_train + nv*eye(nTrain));
pMean = K_cross * K_inv * ytrain;
pVar = diag(K_test + nv*eye(nTest) - K_cross*K_inv*K_cross');

end