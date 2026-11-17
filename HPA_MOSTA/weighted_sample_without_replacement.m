function pos = weighted_sample_without_replacement(weights, num_select)
% Weighted sampling without replacement.
% Inputs: weights - nonnegative sampling weights; num_select - target sample count.
% Output: pos - indices into the input weight vector.

weights = weights(:)';   
    N_items = length(weights);
    eps_val = 1e-12;

    % Truncate negative weights to zero.
    weights(weights < 0) = 0;

    % Normalize the weights; use uniform weights if their sum is negligible.
    if sum(weights) < eps_val
        weights = ones(1, N_items) / N_items;
    else
        weights = weights / sum(weights);
    end

    pos = zeros(1, num_select);
    temp_prob = weights;

    for k = 1:num_select
        % Sample from the cumulative distribution.
        cdf = cumsum(temp_prob);
        rand_value = rand;
        idx = find(cdf >= rand_value, 1, 'first');  
        pos(k) = idx;

        % Set the selected weight to zero.
        temp_prob(idx) = 0;

        if sum(temp_prob) < eps_val  % Stop when the remaining weight is below the numerical threshold
            pos = unique(pos(1:k), 'stable');  
            return;   
        else
            temp_prob = temp_prob / sum(temp_prob);   % Renormalize the remaining weights
        end
    end

    pos = unique(pos, 'stable');
end
