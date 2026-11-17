function [I_star,best_idx] = SelectBest(Z_star,ObjValues,X_tr,y_tr)
% Best solution selection.
% Inputs: Z_star - optimized solution set; ObjValues - objective values [rho, E];
%         X_tr, y_tr - training samples and concentration labels.
% Outputs: I_star - selected harmonic feature point indices; best_idx - solution row index in Z_star.

N_solutions=size(Z_star,1);
% Selection criteria: E_CV, rho, and E.

rank_ECV=[];
rank_rho=[];
rank_E=[];

% Construct repeated stratified K-fold cross-validation partitions.

K = 5;
R = 10;  % Number of repetitions

y_tr = y_tr(:);
concentration_strata = categorical(y_tr);   % Stratify by oxygen concentration level

cv_partitions = cell(R, 1);

for r = 1:R
    cv_partitions{r} = cvpartition(concentration_strata, 'KFold', K);
end

% Compute the cross-validation error of each solution.
E_CV = zeros(N_solutions, 1);

for j = 1:N_solutions
    unselected_mask = Z_star(j,:) == 0;
    X_subset = X_tr;
    X_subset(:, unselected_mask) = [];


    E_val = zeros(K * R, 1);
    fold_counter = 1;
    
    for r = 1:R
        cv = cv_partitions{r};
    
        for i = 1:cv.NumTestSets
            train_idx = cv.training(i);
            val_idx = cv.test(i);
    
            model_cv = fitlm(X_subset(train_idx, :), y_tr(train_idx));
            y_hat_val = predict(model_cv, X_subset(val_idx, :));
    
            y_val = y_tr(val_idx);
    
            % Compute the root mean squared error on the validation fold.
            E_val(fold_counter) = sqrt(mean((y_hat_val - y_val).^2));
    
            fold_counter = fold_counter + 1;
        end
    end
    
    % Average over all repetitions and validation folds.
    E_CV(j) = mean(E_val);
end


% Rank solutions by cross-validation error in ascending order.
[~, order_ECV] = sort(E_CV);
rank_ECV(order_ECV) = 1:numel(E_CV);  


% Rank solutions by feature selection rate in ascending order.
[~,order_rho]=sort(ObjValues(:,1)); 
rank_rho(order_rho) = 1:numel(ObjValues(:,1));  


% Rank solutions by modeling error in ascending order.
[~,order_E]=sort(ObjValues(:,2)); 
rank_E(order_E) = 1:numel(ObjValues(:,2));  

RankMatrix=[rank_ECV;rank_rho;rank_E];

weights=[0.25,0.50,0.25];
Score=weights*RankMatrix;   % Weighted rank aggregation in the order [E_CV, rho, E]



% Select the minimum-score solution and resolve ties.


% Identify solutions within the score tolerance.
Score_min = min(Score);
epsilon = 1e-6;
candidate_idx = find(abs(Score - Score_min) < epsilon);  


if numel(candidate_idx) == 1  
    best_idx = candidate_idx(1);  
else
    % For tied scores, prefer the smaller feature selection rate.
    candidate_rho = ObjValues(candidate_idx, 1);
    min_rho = min(candidate_rho);
    candidate_idx_rho = candidate_idx(abs(candidate_rho - min_rho) < 1e-12);
    if numel(candidate_idx_rho) == 1  
        best_idx = candidate_idx_rho(1);
    else   % For tied feature selection rates, prefer the smaller modeling error.
        candidate_E = ObjValues(candidate_idx_rho, 2);
        [~, temp_idx] = min(candidate_E);
        best_idx = candidate_idx_rho(temp_idx);
    end

end

I_star_original = find(Z_star(best_idx,:) ~= 0);
% disp(['best_idx=',num2str(best_idx)]);
I_star=I_star_original;

end
