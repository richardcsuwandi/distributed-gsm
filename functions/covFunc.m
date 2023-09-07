function subKernels = covFunc(means, variances, x, y)
%% To compute all the sub kernel matrices and store in a cell. (multivariate SM kernel)
%  Arguments:
%     means:                    Q x P matrix
%     variance (std actually):  1 x Q matrix 
%     x:                        input data matrix: n x P matrix
%     y (x'):                   input data matrix: m x P matrix

% return:
%     (Q) a series of (n x m) subKernel matries

%%  allocate memories
P = size(means, 2);         % P is the number of dimensions of input
Q = size(means, 1);         % Q is the number of mixture components
n = size(x, 1);             % n is the number of data points
m = size(y, 1);             % m is the number of data points
subKernels = cell(Q,1);     % Initialize a Q-by-1 empty matrix

assert(size(variances,2) == Q, 'variances with shape: 1 x Q');
assert(size(variances,1) == 1, 'variances with shape: 1 x Q');

%% compute tau, the distance between any two set of vectors,
%   which means there are [nxm] number of distance (x - x') \in R^P
Tau = zeros(n*m, P);     % Initialize a (nxm)-by-P empty matrix for Tau

% % precompute and save Tau
% X = repelem(x, m, 1);   % shape: (n x m) x P
% Y = repmat(y, n, 1);    % shape: (m x n) x P
% Tau1 = X - Y;            % shape: (m x n) x P

% 2022/08/19: revise code to precompute Tau
for p = 1:P
    % Compute tau_p and reshape
   tau_p = x(:, p) - y(:, p)'; 
   % Store the value in the p-th column of Tau
   Tau(:, p) = reshape(tau_p, n*m, 1);
end

% Construct the kernel cells
for i=1:Q
    var = variances(i);
    subKernels{i} = 1;
    for j=1:P
        % Compute the difference matrix based on x
        tau_p = Tau(:, j);

        % shape: (nxn) x 1
        tmp1 = exp(-2 * pi^2 * (tau_p.^2).*(var^2));   % Formula for SM kernel;

        tmp2 = subKernels{i} .* tmp1;
        subKernels{i} = tmp2;
    end
    subKernels{i} = cos( 2 * pi * Tau * means(i, :)' ) .* subKernels{i};
    subKernels{i} = reshape(subKernels{i}, n ,m);      % shape: n x m
end

% %% old one
% 
% % Given the lists of hyperparameters and the difference matrix, compute all
% % the sub kernel matrices and store in a cell.
% 
% P = size(x, 2);             % P is the number of dimensions of input
% Q = size(means, 1);         % Q is the number of mixture components
% subKernels = cell(Q,1);     % Initialize a Q-by-1 empty matrix
% 
% % Construct the kernel cells
% for i=1:Q
%     var = variances(i);
%     subKernels{i} = 1;
%     
%     % Initialize the final tau
%     tau = sparse(size(x, 1));
%     for j=1:P
%         % Compute the difference matrix based on x
%         tau_p = diff_mat(x(:, j), y(:, j));        % diffMat is tau_p
%         tmp1 = exp(-2*pi^2*(tau_p.^2).*(var^2));   % Formula for SM kernel;
% %         diffMat = sq_dist(x(:, j), y(:, j));        % diffMat is tau
% %         tmp1 = exp(-2*pi^2* (diffMat) .*(var^2));     % Formula for SM kernel;
% 
%         tmp2 = subKernels{i} .* tmp1;
%         subKernels{i} = tmp2;
%         
%         % pre-compute the final tau
%         tau = tau + tau_p*means(i, j);
%     end
% %     mu = means(i);   % also something wrong here, mu_i should be an vector px1
%     % new added code: 2022/08/17
% %     diffMat1 = sq_dist(x', y');           
% %     subKernels{i} = cos(2*pi*mu*tau) .* subKernels{i};
% 
%       subKernels{i} = cos(2*pi*tau) .* subKernels{i};
% end
