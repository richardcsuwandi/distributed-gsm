% Function to generate the frequency and variance grids for GSM kernel
% Adapted from generateGSM.m
function [freq, var] = generateMultiGSM(options)
freq_grid = cell(options.nDim, 1);
var_grid = cell(options.nDim, 1);
for i = 1:options.nDim
    if options.sampling == 0 % Uniform 
        freq_grid{i} = linspace(options.freq_lb, options.freq_ub, options.nFreqCand)'; % Generate linearly spaced vector from (freq_lb, freq_ub)
        var_can = options.fix_var;
        var_grid{i} = repmat(var_can, 1, options.nFreqCand)'; % Create a vector whose elements contain the repeated value of var_can
    elseif options.sampling == 1 % Random
        freq_grid{i} = options.freq_lb + (options.freq_ub - options.freq_lb)*rand(options.nFreqCand,1);
        var_grid{i} = repelem(options.fix_var, options.nFreqCand);
    end
end

freq = horzcat(freq_grid{:});
var = horzcat(var_grid{:});
end