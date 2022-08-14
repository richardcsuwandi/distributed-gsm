function diff_mat = diff_mat(x, y)
%DIFF_MAT difference matrix
%   diff_mat return the difference matrix based on the x_train
%
%   Class support for the input x, y:
%       float: double; column vector

diff_mat = x - y';

end