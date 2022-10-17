clc; clear;
addpath('./data_multi')
addpath('./functions')

% MOSEK setup
javaaddpath('C:\Program Files\Mosek\9.3\tools\platform\win64x86\bin\mosek.jar')
addpath('C:\Program Files\Mosek\9.3\toolbox\R2015a')

% Set seed for reproducibility
seed = 42;
rng(seed);

% Read in data & some general setup
file_name = 'ale80';
% file_name = 'airfoil600';
% file_name = 'cccp1000';
% file_name = 'toxicity436'
% file_name = 'concrete824'
% file_name = 'wine1279';

disp(['Simulation on ',file_name]);
[Xtrain, ytrain, Xtest, ytest] = load_data(file_name);
nTrain = length(ytrain);
nTest = length(ytest);
nv = evar(ytrain); % Returns an estimated variance of the Gaussian noise

% Generate multi GSM kernels
Q = 100;
P = size(Xtrain, 2);
A = P*Q;

options = struct('freq_lb', 0, ...
                 'freq_ub', 0.5, ...
                 'nDim', P, ...
                 'nFreqCand', A, ...
                 'nVarCand', 1, ...
                 'fix_var', 0.001, ...
                 'sampling', 0); % 0 for uniform, 1 for random
             
% Sample the frequencies and variances
[freq, var] = generateMultiGSM(options);

% Construct the kernels
K = constructSMP(freq, var, Xtrain, Xtrain);
L = cell(1, A);
for kk =1:A
    L{kk} = (cholcov(K{kk})).'; % Perform Cholesky decomposition
end

% Use different values of S
S_vals = [1; 2; 4; 10; 50; 100];
MSE_vals = zeros(length(S_vals), 1);
time_vals = zeros(length(S_vals), 1);
n_iters = zeros(length(S_vals), 1);
for ii = 1:length(S_vals)
    % Initialize alpha 
    % First argument 0: fix, 1: compute, 2: random
    theta = ini_Alpha(0, 0, A, ytrain, K);

    S = S_vals(ii); % Number of local machines to use
    b = A/S; % Block size

    % Partition the optimization variable into s blocks 
    Theta = mat2cell(theta, diff([0:b:A-1,A]));

    % Algorithm setup
    max_iter = 100;
    obj_val = zeros(max_iter, 1);
    tol = 1e-3;

    % Hyper-parameter optimization
    C_k = C_matrix(theta, K, nv, eye(nTrain));
    tic;
    for k = 1:max_iter
        % Main loop for parallel computation  % can also use 'parfor' for acceleration
        for i = 1:S
            disp(['Updating block ', int2str(i)]);

             m = (i - 1)*b;
            L_i = L((m + 1):(m + b));
            K_i = K((m + 1):(m + b));
            theta_i = Theta{i};

            % Compute g (excluding the i-th block)
            C_i = sparse(nTrain, nTrain);
            for j = 1:numel(theta_i)
                C_i = C_i + theta_i(j) * K_i{j};
            end
            C_minus = C_k - C_i + nv*speye(nTrain);

            % SCA minimization for each block (at k-th iteration)
            [theta_hat, nv] = dsca_optimize(ytrain, L_i, C_k, C_minus, nv);
            Theta{i} = theta_hat;
        end
        theta = vertcat(Theta{:});

        % Compute the objective value
        C_k = C_matrix(theta, K, nv, eye(nTrain));
        obj_val(k) = compute_obj_val(ytrain, C_k);
        disp(['Iteration: ', num2str(k), ' | Obj. Value: ', num2str(obj_val(k))])

        % Stopping criterion
        if k > 1
            if abs(obj_val(k) - obj_val(k - 1)) / max(1., abs(obj_val(k))) < tol
                n_iters(ii) = k;
                break
            end
        end
    end
    time_record = toc;
    time_vals(ii) = time_record;

    % Prediction
    [pMean, pVar] = predict(Xtrain, Xtest, ytrain, nTest, theta, nv, freq, var, K);

    % Record MSE
    MSE = mean((pMean-ytest(1:nTest)).^2);
    MSE_vals(ii) = MSE;
    disp(['MSE: ', num2str(MSE)])

    % Save info
    save(['./fig/dsca_multi/',file_name,'_Q',int2str(Q),'_S',int2str(S), '.mat'], ...
        'MSE', 'time_record');
    disp('-----------------------------------')
end
