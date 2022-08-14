function [obj_val] = compute_obj_val(ytrain, C)
    L = chol(C)';  
    
    % Compute objective value
    tmp = (L\ytrain);
    obj_val = norm(tmp)^2 + 2*sum(log(diag(L)));
end