This repository contains the source codes and data supporting the findings presented in the paper
"Optical Next Generation Reservoir Computing" by H.W, J.H., Y.B., K.T., M.J., Q.L., and S.G..

Below is an overview of the contents:

1. Folder "exp_results": it contains the experimental results in the paper.
2. Figure2.m + Figure3.m + Figure4.m: these files use the data in the folder 
"exp_results" to generate the figures 2, 3, and 4 of the paper.
3. optical_NGRC_sim.m: this file contains the code to simulate the optical NGRC.
4. optical_ConvRC_sim.m: this file contains the code to simulate the optical conventional RC.
5. optical_NGRC_sim_BO.m: this file contains the code to optimize the hyperparameters of the optical NGRC using Bayesian Optimization approach.
6. optical_NGRC_observer.m: this file contains the code to simulate the optical NGRC observer.
7. KS_Spline_Baseline.ipynb: this file contains the code to implement cubic spline interpolation of KS system.
8. generateKS.m: this file contains the code to generate data of the KS system, for instance 'L22_Ninput64.mat'.
9. addNoise.m: this is a function to add noise to the data.
10. calculateNRMSE.m: this is a function to calculate the NRMSE.
11. delayTimeSeries.m: this is a function to delay the time series.

If you find this repository useful, please cite the paper. 
```Optical Next Generation Reservoir Computing, arXiv:2404.07857.```