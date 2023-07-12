function [pMean_loc, pVar_loc, MSE_loc, pMean_fused, pVar_fused, MSE_fused] = loc_predict(Xtrain, xtest, Ytrain, ytest, nTest, freq, var, N, theta_t, nv, file_name)
% Plot local predictions
pMean_loc = zeros(nTest, N);
pVar_loc = zeros(nTest, N);
MSE_loc = zeros(1, N);
for j = 1:N
    K_loc = constructSMP(freq, var, Xtrain{j}, Xtrain{j}); % Generate sub-kernels
    % Prediction
    [pMean_loc(:, j), pVar_loc(:, j)] = predict(Xtrain{j}, xtest, Ytrain{j}, nTest, ...
                                        theta_t, nv, freq, var, K_loc);
    MSE_loc(j) = mean((pMean_loc(1:nTest, j)-ytest(1:nTest)).^2);
end

%  Fuse the result of local prediction
w = zeros(nTest, N);
for j = 1:N
   w(:, j) = (1 ./ pVar_loc(:, j)) ./ sum(1 ./ pVar_loc, 2);  
end

pMean_fused = sum(w .* pMean_loc, 2);
pVar_fused = 1 ./ sum(1 ./ pVar_loc, 2);

MSE_fused = mean((pMean_fused(1:nTest)-ytest(1:nTest)).^2);
end

