function subKernels = constructSMP(means, variances, x, y)

P = size(x, 2);             % P is the input dimension
A = size(means, 1);         % A is the number of mixture components
subKernels = cell(A, 1);    % Initialize a A-by-1 empty matrix

% Construct the kernel cells
for i=1:A
    subKernels{i} = 1;
    for j=1:P
        % Extract the mean and variance
        var = variances(i, j);
        mean = means(i, j);

        % Compute the difference matrix based on x
        diffMat = diff_mat(x(:, j), y(:, j));       % diffMat is tau
        tmp1 = exp(-2*pi^2*(diffMat.^2).*(var^2));  % Formula for SM kernel;
        tmp2 = cos(2*pi*mean*diffMat);
        
        % 2022/08/18: found a bug here, the multiplication did not include
        %             the subKernels{i} from the other dimensions
%         subKernels{i} = tmp1 .* tmp2;
        subKernels{i} = subKernels{i} .* tmp1 .* tmp2; 
    end
end