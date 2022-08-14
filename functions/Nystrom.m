%Nystrom_approximation method

function [U_NY,K_NY] = Nystrom(xtrain,freq_grid,var_grid,num_samp) 
%   Nystrom
%   Nystrom_1005 provides an estimate of the original kernel matrix
% 
%   Input
%   freq_grid: the vector of kernel frequency
%   var_grid: the vactor of kernel variance
%   num_samp: the number of training inputs that need to sample
%
%   Output
%   K_NY: the nystrom approximate kernel
%   U_NY: the chlosky decompostion of nystrom approximate kernel

nTrain = length(xtrain);

%Step-1: sub-sampling training input

xtrainSub = randsample(xtrain, num_samp); % Sub-sample xtrain
xtrainSub = sort(xtrainSub);
nTrainSub = length(xtrainSub);

%Step-2 compute the K_sub_sub & K_full_sub

% diffMatSub = diff_mat(xtrainSub,xtrainSub);
% 
% diffMatFullVsSub = diff_mat(xtrain,xtrainSub);

U_NY = cell(1,length(var_grid));

K_NY = cell(1,length(var_grid));

SS = kernelComponent(freq_grid, var_grid, xtrainSub, xtrainSub);
FS = kernelComponent(freq_grid, var_grid, xtrain, xtrainSub);

for ii = 1:length(freq_grid)
    

    gridSM_SubVsSub = SS{ii};

    gridSM_FullVsSub = FS{ii};

    %step-3 perform eigendecomposition of k_sub_sub

    [eigenvecs_Ny, eigenvalues_Ny] = eig(gridSM_SubVsSub);

    %step-4 apply Nystrom approximation to eigenvalue and eigenvector

    get_eigen = diag(eigenvalues_Ny);

    label = 1:length(get_eigen);

    label = label(get_eigen > 0);

    sel_eigenvecs_Ny = eigenvecs_Ny(:,label);

    sel_eigenvalues_Ny = eigenvalues_Ny(label,label);

    eigenvalues_FullApprox = diag(nTrain/nTrainSub).*sel_eigenvalues_Ny;

    eigenvalues_repmat = repmat(diag(sel_eigenvalues_Ny).', nTrainSub,1);

    eigenvecs_FullApprox = sqrt(nTrainSub/nTrain)*gridSM_FullVsSub*rdivide(sel_eigenvecs_Ny,eigenvalues_repmat);

    %step-5 obtain a low-rank approx of original kernal matrix and cholesky
    %decomposition of approximate matrix

    U_NY{ii} = eigenvecs_FullApprox*sqrt(eigenvalues_FullApprox);

    K_NY{ii} = eigenvecs_FullApprox*eigenvalues_FullApprox*eigenvecs_FullApprox.';


end