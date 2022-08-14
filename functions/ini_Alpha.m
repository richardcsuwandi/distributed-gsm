function alpha = ini_Alpha(method,fixed_value,Q,ytrain,K)
%This function initializes alpha. It can be fixed by one value(usually zero
%or one), or calculated specifically.

%Method:
%0: fix, 1: compute, 2: random.
%If the value is fixed, ini_Alpha('fix',fixd_value,Q,[],[])
%If the value need to be computed, ini_Alpha('compute',[],Q,ytrain,K)
if method==0
    iniAlpha = repmat(fixed_value,Q,1);
elseif method==1
    iniAlpha = zeros(Q,1);
    sampleCovMatrix = ytrain*ytrain';
    vec1 = sampleCovMatrix(:); % Reshape sampleCovMatrix into a column vector
    fNorm1 = norm(sampleCovMatrix, 'fro'); % Compute the Frobenius norm of sampleCovMatrix
    for k=1:Q
        subKernel = K{k};
        vec2 = subKernel(:); % Reshape subKernel into a column vector
        fProduct = vec1.'*vec2;
        fNorm2 = norm(subKernel,'fro'); % Compute the Frobenius norm of subKernel
        iniAlpha(k) = fProduct/(fNorm1*fNorm2);
    end
    s = sum(iniAlpha);
    iniAlpha = iniAlpha/s;
elseif method==2
    alpha_variance = 10;
    iniAlpha = max((sqrt(alpha_variance)*randn(Q,1)),0); % Generate values from max(N(0, alpha_variance), 0)
end
alpha = iniAlpha;
end