function [Z_trans, Obj_trans] = ASST_generate_offspring(objective_fun, Z_src, h, FrontNo, N_T, N_SE, n, X_tr, y_tr, rho_t)
% Candidate solution generation and attention-driven sparse state transition (ASST).
% Inputs: objective_fun - objective function handle; Z_src - source solution set; h - HPA vector;
%         FrontNo - Pareto front numbers; N_T - current candidate selection size;
%         N_SE - current search intensity; n - number of harmonic sampling points; X_tr, y_tr - training samples and concentration labels;
%         rho_t - state transition ratio.
% Outputs: Z_trans - unique transitioned solutions; Obj_trans - objective values [rho, E].

index = find(FrontNo == 1 | FrontNo == 2); 

% Select priority candidates from the first two Pareto fronts.
if length(index) > N_T
    selected_indices = index(1:N_T);   
else
    selected_indices = index;
end

Z_cand = Z_src(selected_indices, :);

% Randomly sample the remaining solutions for diversity injection.
all_indices = 1:size(Z_src, 1);
remaining_indices = setdiff(all_indices, selected_indices);

if ~isempty(remaining_indices)
    if length(remaining_indices) >= N_T
        injected_indices = remaining_indices(randperm(length(remaining_indices), N_T));
    else
        injected_indices = remaining_indices;
    end
    % Merge the candidate solutions.
    Z_cand = [Z_cand; Z_src(injected_indices, :)];
end


% Preallocate storage for transitioned solutions.
N_cand = size(Z_cand, 1);
Z_trans_all = zeros(2 * N_cand * N_SE, n);

row_start = 1;

for i = 1:N_cand

    % Sparse substitute operator
    Z_temp = sparse_substitute_operator(Z_cand(i, :), h, N_SE, rho_t);
    N_temp = size(Z_temp, 1);
    Z_trans_all(row_start:row_start+N_temp-1, :) = Z_temp;
    row_start = row_start + N_temp;

    % Swap operator
    Z_temp = swap_operator(Z_cand(i, :), h, N_SE, rho_t);
    N_temp = size(Z_temp, 1);
    Z_trans_all(row_start:row_start+N_temp-1, :) = Z_temp;
    row_start = row_start + N_temp;
end

% Retain the generated transitioned solutions.
Z_trans_all = Z_trans_all(1:row_start-1, :);

% Remove duplicate transitioned solutions.
Z_trans = unique(Z_trans_all, 'rows');

% Evaluate the transitioned solutions.
Obj_trans = evaluate_objectives(objective_fun, Z_trans, X_tr, y_tr);

end