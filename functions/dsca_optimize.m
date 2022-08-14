% Last updated: 2022-08-11

function [theta, nv] = dsca_optimize(y, L, C_k, C_minus, nv)

% Extract the dimensions
N = size(y, 1);
p = length(L);

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
%   x = (z_0, theta_0, w_0, z_1, theta_1, w_1, ..., z_p, theta_p, w_p)
%
% where theta_0 represents the noise variance (nv)

Kp = zeros(1, p);
for i = 1:p, Kp(i) = size(L{i}, 2); end

% The objective function can be expressed as,
%
%   c'*x = sum_i 2*z_i + sum_i grad_h_i*theta_i
C = cell(p+1, 1);
C{1} = sparse([1,2], [1,1], [2,1], N+2, 1);
for i = 1:p, C{i+1} = sparse([1,2], [1,1], [2, 1], Kp(i)+2, 1); end

% The constraint "blc <= A*x <= buc" corresponds to the equality
% constraints,
%
%   A*x = [ sum_i sqrtm(C_minus)*w_0 + L_i*w_i ] = [ y  ]
%         [ theta_0                            ] = [ nv ]

A = cell(1, p+1);
A{1} = [sparse(N, 2), sqrtm(C_minus)/sqrt(nv)];
for i = 1:p, A{i+1} = [sparse(N, 2), L{i}]; end

% Specify the non-conic part
prob.c   = cell2mat(C);
prob.a   = cell2mat(A);
prob.blc = y;
prob.buc = y;

% Add equality constraint since noise variance is fixed
prob.a   = [prob.a; sparse(1, 2, 1, 1, size(prob.a, 2))];
prob.blc = [prob.blc;nv];
prob.buc = [prob.buc;nv];
 
prob.blx = -inf*ones(size(prob.a, 2), 1);
prob.bux = inf*ones(size(prob.a, 2), 1);

% Specify the cones
prob.cones.type   = ones(p+1,1)*res.symbcon.MSK_CT_RQUAD;
prob.cones.sub    = 1:size(prob.a,2);
prob.cones.subptr = cumsum([1 N+2 Kp(1:end-1)+2]);

% Update problem data
R = chol(C_k); 
grad_h = zeros(p+1,1);
for i = 2:p+1
    grad_h(i) = norm(R'\L{i-1}, 'fro')^2;
end
prob.c(prob.cones.subptr+1) = grad_h;

% Solve the problem
[~, res] = mosekopt('minimize info echo(0)', prob);

% Extract solution
solution = res.sol.itr.xx(prob.cones.subptr+1);
nv       = solution(1);
theta    = solution(2:end);

end