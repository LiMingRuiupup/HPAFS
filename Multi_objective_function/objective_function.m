function obj = objective_function(Data)
% Objective evaluation of a binary solution.
% Input: Data - cell array with Data{1} = X_tr, Data{2} = y_tr, and Data{3} = z.
% Output: obj = [rho, E] - feature selection rate and modeling error (residual standard error).

z=Data{1,3};
X_tr=Data{1,1};
y_tr=Data{1,2}; 

% Extract the feature subset.
del_X = z==0;  
X_subset = X_tr;   
X_subset(:,del_X) = [];

% Compute the feature selection rate.
n_selected=size(X_subset,2);
rho=n_selected/size(z,2);

% Fit an OLS model and compute the residual standard error.
 mdl = fitlm(X_subset, y_tr);  
E=mdl.RMSE;    

obj(1) = rho;
obj(2) = E;

end
