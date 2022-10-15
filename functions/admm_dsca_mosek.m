% 
% Last updated: 2022-09-03
% zlin: for the purpose of debugging
% 

function [zeta, nv] = admm_dsca_mosek(y, L, C_k, C_minus, theta_t, lambda_t, rho, nv)

rho_c = rho/2;
theta = [0; theta_t];

% Extract the dimensions
N = size(y, 1);
p = length(L);
assert(p == size(lambda_t, 1))
assert(p == size(theta_t, 1))

clear prob;
[~, res] = mosekopt('symbcon echo(0)');

% Problem formulation in MOSEK:
%
% minimize     c'*x 
% subject to   blc <= A*x <= buc
%              blx <= x <= bux
%              x in K
%
% The variable x is partitioned as,
%
%   x = (v, z_0, zeta_0, w_0, z_1, zeta_1, w_1, ..., z_p, zeta_p, w_p)
%
% where zeta_0 represents the noise variance (nv)

% Note that the order of x, where now v is in the first position

Kp = zeros(1, p);
for i = 1:p, Kp(i) = size(L{i}, 2); end

% The objective function can be expressed as,
%
% c'*x = rho_c * v + sum_i 2*z_i + sum_i grad_h_i * zeta_i (rho_c = rho/2)

C = cell(p+2, 1);  % initilization

C{1} = rho_c;      % for v
C{2} = sparse([1,2], [1,1], [2,1], N+2, 1);  % for (z_0, zeta_0, w_0)
for i = 1:p, C{i+2} = sparse([1,2], [1,1], [2, 1], Kp(i)+2, 1); end

% The constraint "blc <= A*x <= buc" corresponds to the equality
% constraints,
%
%   A*x = [ sum_i sqrtm(C_minus)*w_0 + L_i*w_i ] = [ y  ]
%         [ zeta_0                             ] = [ nv ]

A = cell(1, p+2);
A{1} = sparse(N, 1); % for v
A{2} = [sparse(N, 2), sqrtm(C_minus)/sqrt(nv)]; % for (z_0, zeta_0, w_0)
for i = 1:p, A{i+2} = [sparse(N, 2), L{i}]; end

% Specify the non-conic part
prob.c   = cell2mat(C);
prob.a   = cell2mat(A);
prob.blc = y;
prob.buc = y;

% Add equality constraint since noise variance is fixed
prob.a   = [prob.a; sparse(1, 3, 1, 1, size(prob.a, 2))];
prob.blc = [prob.blc; nv];
prob.buc = [prob.buc; nv];
 
prob.blx = -inf*ones(size(prob.a, 2), 1);
prob.bux = inf*ones(size(prob.a, 2), 1);

%% Specify the p+2 cones
% % wrong... (MSK_RES_ERR_CONE_OVERLAP_APPEND (The cone to be appended has one variable which is already member of another cone.))
% zeta_index = cumsum([2 N+2 Kp(1:end-1)+2]);
% prob.cones.type   = [ones(p+1,1) * res.symbcon.MSK_CT_RQUAD; res.symbcon.MSK_CT_QUAD];
% prob.cones.sub    = [1:size(prob.a, 2), zeta_index];
% prob.cones.subptr = [cumsum([1 N+2 Kp(1:end-1)+2]), size(prob.a, 2)];

%% Specify the p+2 cones
F_all  = [];
G_all  = [];
CP_all = [];
idx = cumsum([2 N+2 Kp(1:end-1)+2]);  % z_i index
n = size(prob.a, 2);  

%% part 1 --- The (p+2)-th rotated quadratic cone: 2v*0.5 >=|zeta-theta|^2
% x = [v, z_0 zeta_0, w_0, z_1, zeta_1, w_1, ..., z_p, zeta_p, w_p]
F = zeros(p+3, n);

% specify coefficient for v
F(1, :) = sparse([1, zeros(1, n-1)]);

% specify the second line, which is all zero
F(2, :) = zeros(1, n);

% specify the coefficient for zeta
F(3:end, idx+1) = speye(p+1);

% summarize all
F_all = [F_all; F];
G_all = [G_all; 0; 0.5; -theta];
P = [res.symbcon.MSK_CT_RQUAD, size(F, 1)];
CP_all = [CP_all  P];


%% Part 2 ---- The p+1 rotated quadratic cones 
% (z_i, zeta_i,  w_i) \in Q_r^{ Kp(i) + 2 }
for i = 1:p+1
    if i==1
        % number of variables: N+2
        % coefficient matrix: (0, z_0, zeta_0, w_0, 0, 0, ..., 0)
        F = [sparse(N+2, 1), speye(N+2), sparse(N+2, n-N-3)];   
    else    
        % number of variables: Kp(i-1) + 2, i.e., Q_r^{Kp(i-1) + 2} 

        % index of the i-th variable block: (z_i, zeta_i, w_i)
        iidx = idx(i);
        
        % initialization of F
        F = zeros(Kp(i-1)+2, n);

        % specify the coefficient matrix for corresponding variable block 
        F(:, iidx: iidx + Kp(i-1) + 1) = speye(Kp(i-1)+2);
    end

    % aggregate the optimization structure
    F_all = [F_all; F];
    G_all = [G_all; zeros(size(F, 1), 1)];
    P = [res.symbcon.MSK_CT_RQUAD, size(F, 1)];
    CP_all = [CP_all  P];
end

% All cones
prob.f = F_all;
prob.g = G_all;
prob.cones = CP_all;

%%
% Update problem data
R = chol(C_k); 
grad_h = zeros(p+1,1);
for i = 2:p+1
    grad_h(i) = norm(R'\L{i-1}, 'fro')^2;
end
prob.c(idx+1) = [0; lambda_t] + grad_h;

% Solve the problem
% [~, res] = mosekopt('minimize info', prob);
[~, res] = mosekopt('minimize info echo(0)', prob);

% Extract solution
solution = res.sol.itr.xx(idx+1);
nv       = solution(1);
zeta     = solution(2:end);

end
