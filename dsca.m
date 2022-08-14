clc; clear all; close all;

addpath('./data')
addpath('./functions')

% MOSEK setup
javaaddpath('C:\Program Files\Mosek\9.3\tools\platform\win64x86\bin\mosek.jar')
addpath('C:\Program Files\Mosek\9.3\toolbox\R2015a')

% Read in data
file_name = 'electricitydata';
% file_name = 'passengerdata';
% file_name = 'hoteldata';
% file_name = 'employmentdata';
% file_name = 'unemployment';
% file_name = 'clay';
% file_name = 'CO2';
% file_name = 'ECG_signal';

disp(['Simulation on ',file_name]);
[xtrain, ytrain, xtest, ytest] = load_data(file_name);
nTrain = length(xtrain);
nTest = 20; % Set the number of test data

if file_name == "CO2"
    xtest = [xtrain(481); xtest];
    ytest = [ytrain(481); ytest];
    xtrain = xtrain(1:480, :);
    ytrain = ytrain(1:480, :);
    nTest = 21;
    nTrain = length(xtrain);
end

nv = evar(ytrain); % Returns an estimated variance of the Gaussian noise

% Sampling method: 0 for fixed grids, 1 for random grids
Q = 500; % Number of mixture components
options_gen = struct('freq_lb', 0, 'freq_ub', 0.5, ...
                     'var_lb', 0, 'var_ub', 0.15, ... 
                     'Q', Q, ...
                     'nFreqCand', Q, 'nVarCand', 1, ...
                     'fix_var', 0.001,...
                     'sampling', 0 );
             
% Generate the GSM kernels
[freq, var, Q] = generateGSM(options_gen);


Nystrom_activate = 0; % 0 for deactivate nystrom, 1 for activate nystrom
if Nystrom_activate == 0
    K = kernelComponent(freq, var, xtrain, xtrain); % Generate sub-kernels
    L = cell(1,Q);
    for i =1:Q
        L{i} = (cholcov(K{i})).'; % Perform Cholesky decomposition
    end
else
    % Kernel matrix low rank approximation
    nys_sample = ceil(length(xtrain)/20);
    [L,K] = Nystrom(xtrain,freq,var,nys_sample);
end

% Initialize theta. First argument 0:fix, 1: compute, 2: random.
theta = ini_Alpha(0, 0, Q, ytrain, K);

% Algorithm setup
S = 10; 
b = Q/S; 
max_iter = 10; 
obj_val = zeros(max_iter, 1);
tol = 1e-3;

% Partition the optimization variable into s blocks 
Theta = mat2cell(theta, diff([0:b:Q-1,Q]));

% Hyper-parameter optimization
C_k = C_matrix(theta, K, nv, eye(nTrain));
tic;
for k = 1:max_iter
    % Main loop for parallel update
    for i = 1:S
        disp(['Updating block ', int2str(i)]);
        
        m = (i - 1)*Q/S;
        L_i = L((m + 1):(m + Q/S));
        K_i = K((m + 1):(m + Q/S));
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
            break
        end
    end
end
time_record = toc;

% Prediction
[pMean, pVar] = prediction(xtrain, xtest, ytrain, nTest, ...
    theta, nv, freq, var, K);

figName = ['./fig/',file_name,'_Q',int2str(Q),'_S',int2str(S), '.fig'];
plot_save(xtrain, ytrain, xtest, ytest, nTest, pMean, pVar, figName, file_name);

% Record MSE
MSE = mean((pMean-ytest(1:nTest)).^2);

% Save info
save(['./fig/',file_name,'_Q',int2str(Q),'_S',int2str(S), '.mat'], ...
    'MSE', 'time_record');