function [obj_grad] = compute_obj_grad(ytrain, C, K_i)
    obj_grad = zeros(size(K_i, 1), 1); % Initialize the gradient as zeros
    for j = 1:numel(obj_grad)
        tmp1 = - ytrain' * (C\K_i{j}) * (C\ytrain);
        tmp2 = trace(C\K_i{j});
        obj_grad(j) = tmp1 + tmp2;
    end
end

