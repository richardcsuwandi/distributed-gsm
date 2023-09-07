clc; clear all; close all;

addpath('./data_multi')
addpath('./functions')

% MOSEK setup
javaaddpath('/Users/richardcsuwandi/Downloads/mosek/10.0/tools/platform/osx64x86/bin/mosek.jar')
addpath('/Users/richardcsuwandi/Downloads/mosek/10.0/toolbox/r2017a')

% Read in data & some general setup
% file_name = 'ale80';
% file_name = 'airfoil600';
% file_name = 'cccp1000';
% file_name = 'toxicity436'
% file_name = 'concrete824'
% file_name = 'wine1279';
file_name = 'cccp9500';

disp(['Simulation on ',file_name]);
[xtrain, ytrain, xtest, ytest] = load_data(file_name);
nTrain = length(xtrain);
nTest = length(xtest);
nv = evar(ytrain); % Returns an estimated variance of the Gaussian noise

% Generate multi GSM kernels
Q = 100;
P = size(xtrain, 2);
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
K = constructSMP(freq, var, xtrain, xtrain);

% Initialize zeta
% First argument 0: fix, 1: compute, 2: random
zeta_init = ini_Alpha(0, 0, A, ytrain, K);

N = 10; % Number of local machines to use

% Partition the data
Xtrain = mat2cell(xtrain, diff([0:floor(nTrain/N):nTrain-1,nTrain]));
Ytrain = mat2cell(ytrain, diff([0:floor(nTrain/N):nTrain-1,nTrain]));

% ADMM setup
rho_init = 1e-10;
lambda_init = zeros(A, N); % Dual variable
options = struct('N', N, 'rho', rho_init, 'lambda_init', lambda_init, ...
                 'max_iter', 10, 'local_max_iter', 10, ...
                 'zeta_init', zeta_init, 'nv', nv, ...
                 'local_tol', 1e-3, 'tol_abs', 1e-3, 'tol_rel', 1e-3, ...
                 'mu', 10, 'nu', 2, 'apply_rb', 1);

disp('Algorithm Setup:')
disp(options)

% Hyperparameter optimization using the ADMM + DSCA framework
tic;

N = options.N;
theta_t = options.zeta_init;
lambda_t = options.lambda_init;
zeta_t = repmat(options.zeta_init, 1, N);
rho = options.rho;
nv = options.nv;

% Revised by Richard (2022-09-04):
% Pre-compute K_j and L_j for each local agent
K_loc = cell(1, N);
L_loc = cell(1, N);
for j = 1:N
     % Generate sub-kernels
    tmp1 = constructSMP(freq, var,  Xtrain{j},  Xtrain{j}); 
    tmp2 = cell(1, A);
    % Perform Cholesky decomposition
    for jj =1:A
        tmp2{jj} = (cholcov(tmp1{jj})).'; 
    end
    L_loc{j} = tmp2;
    K_loc{j} = tmp1;
end

% Main loop
for t = 1:options.max_iter
    disp('---------------------------------------------')
    disp(['Iteration ', num2str(t), ' of ADMM'])

    % Obtain the local hyperparameters
    for j = 1:N
        disp(['Optimizing local hyperparameters for agent ', num2str(j)])
        
        % Local datasets
        x_train_j = Xtrain{j};  y_train_j = Ytrain{j}; 

        % Length of local data
        nTrain_j = length(x_train_j); 
        
        % Extract the pre-computed K_j and L_j
        K_j = K_loc{j}; L_j = L_loc{j};
        
        % Initialization of DSCA for the local agent
        obj_val_j = zeros(options.local_max_iter, 1);
        S = 4; % Number of parallel computing units
        b = A/S; % Block size

        % Partition the optimization variable into S blocks
        Zeta_j = mat2cell(theta_t, diff([0:b:A-1,A]));
        Theta_t_j = mat2cell(theta_t, diff([0:b:A-1,A]));
        lambda_t_j = lambda_t(:, j);
        zeta_t_j_old = zeta_t(:, j); % Store the old value
        C_k_j = C_matrix(zeta_t(:, j), K_j, options.nv, eye(nTrain_j));
        for k = 1:options.local_max_iter
            % Main loop for parallel computation
            for i = 1:S
                disp(['Updating block ', int2str(i)]);

                m = (i - 1)*b;
                L_j_i = L_j((m + 1):(m + b));
                K_j_i = K_j((m + 1):(m + b));
                lambda_t_j_i = lambda_t_j((m + 1):(m + b));
                zeta_j_i = Zeta_j{i};
                theta_t_j_i = Theta_t_j{i};

                % Compute g (excluding the i-th block)
                C_i = sparse(nTrain_j, nTrain_j);
                for ii = 1:numel(zeta_j_i)
                    C_i = C_i + zeta_j_i(ii) * K_j_i{ii};
                end
                C_minus = C_k_j - C_i + options.nv * speye(nTrain_j);

                % SCA minimization for each block (at k-th iteration)
                [zeta_j_i_hat, nv] = admm_dsca_mosek(y_train_j, L_j_i, C_k_j, C_minus, ...
                    theta_t_j_i, lambda_t_j_i, rho, options.nv);
                % nv
                assert(nv == options.nv, 'Do you need to optimize the noise variable?')

                Zeta_j{i} = zeta_j_i_hat;
            end
            zeta_t(:, j) = vertcat(Zeta_j{:});

            % Compute the objective value
            C_k_j = C_matrix(zeta_t(:, j), K_j, options.nv, eye(nTrain_j));
            obj_val_j(k) = compute_obj_val(y_train_j, C_k_j);
            disp(['Iteration: ', num2str(k), ' | Obj. Value: ', num2str(obj_val_j(k))])
            
            % Stopping criterion for DSCA
            if k > 1
                if abs(obj_val_j(k) - obj_val_j(k - 1)) / max(1., abs(obj_val_j(k))) < options.local_tol
                    break
                end
            end
        end
    end

    % Obtain the global hyperparameters
    theta_t_old = theta_t; % Store the old value of theta_t
    theta_t = (sum(zeta_t, 2) + 1 / rho * sum(lambda_t, 2)) / N;

    % Obtain the dual variable
    for j = 1:N
        lambda_t(:, j) = lambda_t(:,j) + rho*(zeta_t(:, j) - theta_t);
    end

    % Compute the primal and dual residuals
    history.norm_pri_res(t) = norm(zeta_t - repmat(theta_t, 1, N), 'fro');
    history.norm_dual_res(t) = norm(rho*(theta_t - theta_t_old), 'fro');

    disp(['norm_pri_res: ', num2str(history.norm_pri_res(t))])
    disp(['norm_dual_res: ', num2str(history.norm_dual_res(t))])

    % Primal and dual tolerance
    [A, N] = size(lambda_t);
    history.tol_pri(t) = sqrt(A*N)*options.tol_abs + ...
                        options.tol_rel*max(norm(zeta_t, 'fro'), norm(theta_t, 'fro'));
    history.tol_dual(t) = sqrt(A*N)*options.tol_abs + ...
                        options.tol_rel*norm(rho*lambda_t, 'fro');

    disp(['tol_pri: ', num2str(history.tol_pri(t))])
    disp(['tol_dual: ', num2str(history.tol_dual(t))])
    
    % Stopping criteria for ADMM
    if t > 1
        if and( history.norm_pri_res(t) < history.tol_pri(t),...
                history.norm_dual_res(t) < history.tol_dual(t))
            break
        end
    end

    % Apply residual balancing
    if options.apply_rb
        if history.norm_pri_res(t) > options.mu*history.norm_dual_res(t)
            rho = rho * options.nu;
            disp(['rho: ', num2str(rho)])
        elseif history.norm_dual_res(t) > history.norm_pri_res(t)
            rho = rho / options.nu;
            disp(['rho: ', num2str(rho)])
        end
    end
end
time_record = toc;

% Prediction
[pMean, pVar] = predict(xtrain, xtest, ytrain, nTest, theta_t, nv, freq, var, K);

% Record MSE
MSE = mean((pMean-ytest(1:nTest)).^2);

% Plot local predictions
[pMean_loc, pVar_loc, MSE_loc, pMean_fused, pVar_fused, MSE_fused] = loc_predict(Xtrain, xtest, Ytrain, ytest, nTest, freq, var, N, theta_t, nv, file_name);

% Save info
save(['./fig/d2sca_multi/',file_name,'_rho',num2str(rho_init), '_maxiter',int2str(options.max_iter), '.mat'], ...
    'MSE', 'MSE_loc', 'MSE_fused', 'time_record', 'history');