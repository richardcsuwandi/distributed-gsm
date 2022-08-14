function K = kernelComponent(freq_grid, var_grid, x, y)
%Given the lists of hyperparameters and the difference matrix, compute all
%the sub kernel matrices and store in a cell.

Q = length(freq_grid);
subKernels = cell(Q,1); % Initialize a Q-by-1 empty matrix

% compute the difference matrix based on x_train
diffMat = diff_mat(x, y); % diffMat is tau

% constructing the Kernel cells
for k=1:Q
    freqPara = freq_grid(k);
    varPara = var_grid(k);
    subKernels{k} = exp(-2*pi^2*(diffMat.^2).*(varPara^2)).*cos(2*pi*freqPara*diffMat); % Formula for GSM kernel
end
K = subKernels;

end