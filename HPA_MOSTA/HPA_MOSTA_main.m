function [Z_star,ObjValues,history_obj_mean,Z_init,hv] = HPA_MOSTA_main(objective_fun,n,N_obj,N_set,N_iter,X_tr,y_tr,h,Parameters)
% HPA-MOSTA multi-objective optimization.
% Inputs: objective_fun - objective function handle; n - number of harmonic sampling points; N_obj - number of objectives;
%         N_set - solution set size; N_iter - maximum iterations; X_tr, y_tr - training samples and concentration labels;
%         h - HPA vector; Parameters - HPA-MOSTA parameters.
% Outputs: Z_star - optimized solution set; ObjValues - objective values [rho, E];
%          history_obj_mean - mean objective values [rho, E] at each iteration; Z_init - initial solution set;
%          hv - estimated normalized HV.

N_T = Parameters.N_T;
N_SE = Parameters.N_SE;
rho_e = Parameters.rho_e;
rho_t = Parameters.rho_t;

[Z]=ADSI(N_set,n,h);    % Initialize the solution set using ADSI


Z_init=Z;   % Store the initial solution set

ObjValues = evaluate_objectives(objective_fun,Z,X_tr,y_tr); % Evaluate the initial solution set


history_obj_mean = zeros(N_iter + 1, N_obj);% Initialize the history of mean objective values
history_obj_mean(1,:)=mean(ObjValues,1);% Record the initial mean objective values

% Store the base search parameters.
N_T_base=N_T;
N_SE_base=N_SE;

% Set the parameters for the initial enhanced search phase.
N_T_enhanced = ceil(N_T * 10);
N_SE_enhanced = ceil(N_SE * 10);

N_T = N_T_enhanced;
N_SE = N_SE_enhanced;


N_enhanced=ceil((N_iter*rho_e)); % Number of initial enhanced search iterations


% Iterative optimization.
for t = 1:N_iter    
    
    % Restore the base parameters after the initial enhanced search phase.
    if t>N_enhanced
       N_T=N_T_base;
       N_SE=N_SE_base;
    end

    FrontNo = ENS_SS(ObjValues);% Assign Pareto front numbers to the current solutions
    
    % Generate transitioned solutions.
    [Z_trans,Obj_trans]  = ASST_generate_offspring(objective_fun,Z,h,FrontNo,N_T,N_SE,n,X_tr,y_tr,rho_t);
        
    
    % Merge the source and transitioned solutions.
    merged_pool = [Z, ObjValues; ...
                   Z_trans, Obj_trans];

    % Remove duplicate objective vectors, retaining the first occurrence.
    [~, unique_indices] = unique(merged_pool(:, n+1:n+N_obj), 'rows', 'stable');
    unique_pool = merged_pool(unique_indices, :);

    % Perform non-dominated sorting and compute crowding distances.
    ranked_pool = non_dominated_sort_and_crowding(unique_pool, N_obj, n);

    % Perform elite selection.
    elite_pool = elite_select(ranked_pool, N_obj, n, N_set);

    % Extract the solutions and objective values.
    Z = elite_pool(:, 1:n);
    ObjValues = elite_pool(:, n+1:n+N_obj);

    % Record the mean objective values.
    history_obj_mean(t+1, :) = mean(ObjValues, 1);
end

% Set the fixed HV reference point.
hv_reference(1)=1; % Reference value for the feature selection rate
hv_reference(2)=5; % Reference value for the modeling error
N_mc=1000000;

ref_point=[hv_reference(1),hv_reference(2)]; 
hv=hypervolume(ObjValues,ref_point,N_mc);


Z_star = Z;

end
    


 
