function K_sum = sumup_kernel(K, Q, alpha)
    K_sum = 0;
    for i = 1:Q
        K_sum = K_sum + alpha(i)*K{i};
    end
end