function C = C_matrix(alpha, K, nv, eyeM)
%C_MATRIX CONTRUCTOR
%   Class support:
%       alpha: double;
%       K: cell, Kernel collection;
%       nv: double, noise;
%       eyeM: double, indentity matrix of size n by n;
    C = 0;
    Q = numel(K);
    for i = 1:Q
        C = C + alpha(i)*K{i};
    end
    C = C + nv*eyeM;
end