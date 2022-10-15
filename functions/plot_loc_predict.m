function [pMean_loc, pVar_loc, MSE_loc, pMean_fused, pVar_fused, MSE_fused] = plot_loc_predict(Xtrain, xtest, Ytrain, ytest, nTest, freq, var, N, theta_t, nv, file_name)
% Plot local predictions
pMean_loc = zeros(nTest, N);
pVar_loc = zeros(nTest, N);
MSE_loc = zeros(1, N);
for j = 1:N
    K_loc = kernelComponent(freq, var, Xtrain{j}, Xtrain{j}); % Generate sub-kernels
    % Prediction
    [pMean_loc(:, j), pVar_loc(:, j)] = prediction(Xtrain{j}, xtest, Ytrain{j}, nTest, ...
                                        theta_t, nv, freq, var, K_loc);
    figure(); hold on;
    plot(Xtrain{j},Ytrain{j},'b','LineWidth',2);
    f = [pMean_loc(1:nTest, j) + 2*sqrt(pVar_loc(1:nTest, j)); flip(pMean_loc(1:nTest, j) - 2*sqrt(pVar_loc(1:nTest, j)),1)];
    fill([xtest(1:nTest); flip(xtest(1:nTest),1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
    plot(xtest(1:nTest), ytest(1:nTest), 'g','LineWidth',2); 
    plot(xtest(1:nTest), pMean_loc(1:nTest, j), 'k','LineWidth',2);
    MSE_loc(j) = mean((pMean_loc(1:nTest, j)-ytest(1:nTest)).^2);
end

%  Fuse the result of local prediction
w = zeros(nTest, N);
for j = 1:N
   w(:, j) = (1 ./ pVar_loc(:, j)) ./ sum(1 ./ pVar_loc, 2);  
end

pMean_fused = sum(w .* pMean_loc, 2);
pVar_fused = 1 ./ sum(1 ./ pVar_loc, 2);

figure(); hold on;
xtrain = Xtrain{:}; ytrain = Ytrain{:};
plot(xtrain,ytrain,'b','LineWidth',2);
f = [pMean_fused(1:nTest) + 2*sqrt(pVar_fused(1:nTest)); flip(pMean_fused(1:nTest) - 2*sqrt(pVar_fused(1:nTest)),1)];
fill([xtest(1:nTest); flip(xtest(1:nTest),1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
plot(xtest(1:nTest), ytest(1:nTest), 'g','LineWidth',2); 
plot(xtest(1:nTest), pMean_fused(1:nTest), 'k','LineWidth',2);
title(file_name);

MSE_fused = mean((pMean_fused(1:nTest)-ytest(1:nTest)).^2);
end

