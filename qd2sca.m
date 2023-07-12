clc; clear all; close all;

addpath('./data')
addpath('./functions')

% MOSEK setup
javaaddpath('/Users/richardcsuwandi/Downloads/mosek/10.0/tools/platform/osx64x86/bin/mosek.jar')
addpath('/Users/richardcsuwandi/Downloads/mosek/10.0/toolbox/r2017a')

% Read in data & some general setup
% file_name = 'electricitydata';
% file_name = 'passengerdata';
% file_name = 'hoteldata';
% file_name = 'employmentdata';
% file_name = 'unemployment';
% file_name = 'clay';
file_name = 'CO2';
% file_name = 'ECG_signal';  

disp(['Simulation on ',file_name]);
[xtrain, ytrain, xtest, ytest] = load_data(file_name);
nTrain = length(xtrain);
nTest = 20;

if file_name == "CO2"
    xtest = [xtrain(481); xtest];
    ytest = [ytrain(481); ytest];
    xtrain = xtrain(1:480, :);
    ytrain = ytrain(1:480, :);
    nTest = 21;
    nTrain = length(xtrain);
end

nv = evar(ytrain); % Returns an estimated variance of the Gaussian noise

% Sampling method: 0 represents fixed grids, 1 represents random.
Q = 500;
options_gen = struct('freq_lb', 0, 'freq_ub', 0.5, ...
                     'var_lb', 0, 'var_ub', 0.15, ... 
                     'Q', Q, ...
                     'nFreqCand', Q, 'nVarCand', 1, ...
                     'fix_var', 0.001,...
                     'sampling', 0 );
             
% Generate GSM kernels
[freq, var, Q] = generateGSM(options_gen); % the length of freq or var is Q we need
K = kernelComponent(freq, var, xtrain, xtrain); % Generate sub-kernels

% Initialize zeta
% First argument 0: fix, 1: compute, 2: random
zeta_init = ini_Alpha(0, 0, Q, ytrain, K);

N = 2; % Number of local machines to use

% Partition the data
Xtrain = mat2cell(xtrain, diff([0:floor(nTrain/N):nTrain-1,nTrain]));
Ytrain = mat2cell(ytrain, diff([0:floor(nTrain/N):nTrain-1,nTrain]));

% ADMM setup
rho_init = 1e-10;
lambda_init = zeros(Q, N); % Dual variable
Delta = 0.5; % Quantization resolution
options = struct('N', N, 'rho', rho_init, 'lambda_init', lambda_init, ...
                 'max_iter', 100, 'local_max_iter', 100, ...
                 'zeta_init', zeta_init, 'nv', nv, ...
                 'local_tol', 1e-3, 'tol_abs', 1e-3, 'tol_rel', 1e-3, ...
                 'mu', 10, 'nu', 2, 'apply_rb', 1, 'Delta', Delta);

disp('Algorithm Setup:')
disp(options)

% Hyperparameter optimization using the ADMM with quantization + DSCA framework
tic;

N = options.N;
theta_t = options.zeta_init;
lambda_t = options.lambda_init;
zeta_t = repmat(options.zeta_init, 1, N);
rho = options.rho;
nv = options.nv;

% Pre-compute K_j and L_j for each local agent
K_loc = cell(1, N);
L_loc = cell(1, N);
for j = 1:N
     % Generate sub-kernels
    tmp1 = kernelComponent(freq, var,  Xtrain{j},  Xtrain{j}); 
    tmp2 = cell(1, Q);
    % Perform Cholesky decomposition
    for jj =1:Q
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
        b = Q/S; % Block size

        % Partition the optimization variable into S blocks
        Zeta_j = mat2cell(theta_t, diff([0:b:Q-1,Q]));
        Theta_t_j = mat2cell(theta_t, diff([0:b:Q-1,Q]));
        lambda_t_j = lambda_t(:, j);
        zeta_t_j_old = zeta_t(:, j); % Store the old value
        C_k_j = C_matrix(zeta_t(:, j), K_j, options.nv, eye(nTrain_j));
        for k = 1:options.local_max_iter
            % Main loop for parallel computation
            for i = 1:S
                disp(['Updating block ', int2str(i)]);

                m = (i - 1)*Q/S;
                L_j_i = L_j((m + 1):(m + Q/S));
                K_j_i = K_j((m + 1):(m + Q/S));
                lambda_t_j_i = lambda_t_j((m + 1):(m + Q/S));
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

            % Break if no improvement in the variable being optimized
            if and(norm(zeta_t(:, j) - zeta_t_j_old) / max(1., abs(zeta_t(:, j))) < options.local_tol, k >= 1)
                break
            end
            
            % Quantize the local hyperparameters
            zeta_t(:, j) = quantize(zeta_t(:, j), options.Delta);
        end
    end

    % Obtain the global hyperparameters
    theta_t_old = theta_t; % Store the old value of theta_t
    theta_t = (sum(zeta_t, 2) + 1 / rho * sum(lambda_t, 2)) / N;

    % Quantize the global hyperparameters
    theta_t = quantize(theta_t, options.Delta);

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
    [Q, N] = size(lambda_t);
    history.tol_pri(t) = sqrt(Q*N)*options.tol_abs + ...
                        options.tol_rel*max(norm(zeta_t, 'fro'), norm(theta_t, 'fro'));
    history.tol_dual(t) = sqrt(Q*N)*options.tol_abs + ...
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
[pMean, pVar] = prediction(xtrain, xtest, ytrain, nTest, ...
                           theta_t, nv, freq, var, K);

% Plot and save the results
figName = ['./fig/qd2sca/',file_name,'_rho',num2str(rho_init), '_maxiter',int2str(options.max_iter), '_N', int2str(options.N), '_Delta', num2str(options.Delta), '.fig'];
plot_save(xtrain,ytrain,xtest,ytest,nTest,pMean,pVar,figName,file_name);

% Record MSE
MSE = mean((pMean-ytest(1:nTest)).^2);

% Plot local predictions
[pMean_loc, pVar_loc, MSE_loc, pMean_fused, pVar_fused, MSE_fused] = plot_loc_predict(Xtrain, xtest, Ytrain, ytest, nTest, freq, var, N, theta_t, nv, file_name);

% Save info
save(['./fig/qd2sca/',file_name,'_rho',num2str(rho_init), '_maxiter',int2str(options.max_iter), '_N', int2str(options.N), '_Delta', num2str(options.Delta), '.mat'], ...
    'MSE', 'MSE_loc', 'MSE_fused', 'time_record', 'history');