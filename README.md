# Distributed Learning for Grid Spectral Mixture (GSM) kernel

This repository contains the code and data used in the paper ["Gaussian Process Regression with Grid Spectral Mixture Kernel: Distributed Learning for Multidimensional Data"](https://ieeexplore.ieee.org/document/9841347) published at the 25th International Conference on Information Fusion (FUSION) 2022.
- R. C. Suwandi, Z. Lin, Y. Sun, Z. Wang, L. Cheng and F. Yin, "Gaussian Process Regression with Grid Spectral Mixture Kernel: Distributed Learning for Multidimensional Data," 2022 25th International Conference on Information Fusion (FUSION), 2022, pp. 1-8, doi: 10.23919/FUSION49751.2022.9841347.

## Citation
```
@INPROCEEDINGS{9841347,  
  author={Suwandi, Richard Cornelius and Lin, Zhidi and Sun, Yiyong and Wang, Zhiguo and Cheng, Lei and Yin, Feng},  
  booktitle={2022 25th International Conference on Information Fusion (FUSION)},   
  title={Gaussian Process Regression with Grid Spectral Mixture Kernel: Distributed Learning for Multidimensional Data},   
  year={2022},  
  volume={},  
  number={},  
  pages={1-8},  
  doi={10.23919/FUSION49751.2022.9841347}
}
```

## Example
To see an example, please run the `dsca.m` file for the Distributed SCA (DSCA) algorithm or the `d2sca.m` file for the Doubly Distributed SCA (D$^2$SCA) algorithm.
To change the data set, simply uncomment one of the listed data sets and comment the others, e.g., for the `Electricity` data set,
```
% Read in data & some general setup
file_name = 'electricitydata';
% file_name = 'passengerdata';
% file_name = 'hoteldata';
% file_name = 'employmentdata';
% file_name = 'unemployment';
% file_name = 'clay';
% file_name = 'CO2';
% file_name = 'ECG_signal';  

disp(['Simulation on ',file_name]);
[xtrain, ytrain, xtest, ytest] = load_data(file_name);
nTrain = length(xtrain);
nTest = 20;
```
Other algorithm setups can also be changed in the corresponding files.

## Dependencies
The current version of the code uses MATLAB R2021a (https://www.mathworks.com/products/matlab.html) and MOSEK version 9.3 (https://docs.mosek.com/9.3/install/installation.html). Please refer to the corresponding websites for the installation instructions.
