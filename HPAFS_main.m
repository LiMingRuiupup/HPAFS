% HPAFS: Harmonic Point Attention-based Feature Selection

clear all; close all; clc;

%% Data loading

% Experimental data are not distributed due to confidentiality restrictions
% related to the measurement platform. Users should provide their own data.
%
% Required variables in data.mat:
%   X_tr: N_tr-by-n training sample matrix
%   y_tr: N_tr-by-1 training concentration labels
%   X_te: N_te-by-n test sample matrix
%   y_te: N_te-by-1 test concentration labels

if ~isfile('data.mat')
    error('HPAFS:MissingData', ...
        'Please provide your dataset as data.mat in the current folder.');
end

load('data.mat');

%% Optimization settings

n = size(X_tr,2); % Number of harmonic sampling points
N_set = 50; % Solution set size
N_iter =1000; % Maximum number of iterations
N_obj = 2; % Number of objectives

%% HPA-MOSTA parameters

Parameters.N_T   = 3;       % Candidate selection size
Parameters.N_SE  = 2;       % Search intensity
Parameters.rho_e = 0.015;   % Initial enhancement ratio
Parameters.rho_t = 0.0025;  % State transition ratio


%% 1. HPA calculation

h = CalculateHPA(X_tr,y_tr); 


%% 2. Multi-objective construction
% Objective columns [rho, E]: feature selection rate and modeling error, respectively.

%% 3. Multi-objective optimization

tic  
[Z_star,ObjValues,history_obj_mean,Z_init,hv] = HPA_MOSTA_main(@objective_function,n,N_obj,N_set,N_iter,X_tr,y_tr,h,Parameters);
optimization_time=toc;

%% 4. Best solution selection

[I_star,best_idx] = SelectBest(Z_star,ObjValues,X_tr,y_tr);

%% 5. Feature subset modeling

% Extract the training feature subset.
X_tr_subset=X_tr(:,I_star);
% Train the OLS concentration measurement model.
M_star = fitlm(X_tr_subset, y_tr);  

% Compute the training performance metrics.
trainMetrics = regression_metrics(M_star.Fitted, y_tr);

%% 6. Online measurement


% Extract the test feature subset.

X_te_subset=X_te(:,I_star);
% Predict the oxygen concentrations.
y_hat_te= predict(M_star, X_te_subset);  


% Compute the measurement performance metrics.
testMetrics  = regression_metrics(y_hat_te, y_te);


%% Result reporting

fprintf('\n==================================================\n');
fprintf('                  HPAFS Results\n');
fprintf('==================================================\n');

% Feature selection results
fprintf('Number of harmonic sampling points: %d\n', n);
fprintf('Number of selected feature points: %d\n', numel(I_star));
fprintf('Feature selection rate: %.4f%%\n', 100 * numel(I_star) / n);
fprintf('Selected solution index: %d\n', best_idx);
fprintf('Modeling error: %.4f\n', ObjValues(best_idx, 2));

fprintf('Selected harmonic feature point indices: ');
fprintf('%d ', I_star);
fprintf('\n');

% Optimization performance
fprintf('\nOptimization runtime: %.4f s\n', optimization_time);
fprintf('HV: %.4f\n', hv);

% Training and measurement performance metrics
fprintf('\n--------------------------------------------------\n');
fprintf('%-8s %12s %14s\n', 'Metric', 'Training', 'Measurement');
fprintf('--------------------------------------------------\n');

fprintf('%-8s %11.4f %12.4f\n', 'MAE', trainMetrics.MAE, testMetrics.MAE);
fprintf('%-8s %11.4f %12.4f\n', 'MAPE', trainMetrics.MAPE, testMetrics.MAPE);
fprintf('%-8s %11.4f %12.4f\n', 'RMSE', trainMetrics.RMSE, testMetrics.RMSE);
fprintf('%-8s %11.4f %12.4f\n', 'R2', trainMetrics.R2, testMetrics.R2);

fprintf('==================================================\n\n');



