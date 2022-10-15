function [pMean, pVar] = predict(xtrain,xtest,ytrain,nTest,alpha,nv,freq,var,K)

Q = length(freq);
nTrain = length(xtrain);
xtest = xtest(1:nTest, :);

%% should not use covFunc again here, since we use constructSMP to construct kernel matrix
% 2022/08/18
% 
% K_cross_set = covFunc(freq, var, xtest, xtrain);
% K_test_set = covFunc(freq, var, xtest, xtest);
K_cross_set = constructSMP(freq, var, xtest, xtrain);
K_test_set = constructSMP(freq, var, xtest, xtest);

% sumup sub-Kernels sets
K_cross = sumup_kernel(K_cross_set, Q, alpha);
K_test = sumup_kernel(K_test_set, Q, alpha);
K_train = sumup_kernel(K, Q, alpha);

% prediction phase
K_inv = pinv(K_train + nv*eye(nTrain));
pMean = K_cross * K_inv * ytrain;
pVar = diag(K_test + nv*eye(nTest) - K_cross*K_inv*K_cross');

end