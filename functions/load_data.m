function [x_train, y_train, x_test, y_test] = load_data(file_name)
%LOAD_DATA read in data
%   load_data(file_name) read in data from a .mat file in the ./data
%   directory
%   
%   Class support for input file_name:
%       char; 
%       name form: 'XXXX.mat'

load([file_name,'.mat']);

x_train = xtrain;
y_train = ytrain;
x_test = xtest;
y_test = ytest;

end