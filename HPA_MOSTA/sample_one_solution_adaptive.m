function pos = sample_one_solution_adaptive(h_norm, u, k_i)
% Single-solution sampling in ADSI.
% Inputs: h_norm - normalized HPA weights; u - point coverage counts across previous solutions;
%         k_i - target number of selected points.
% Output: pos - selected harmonic point indices.

h_norm = h_norm(:)';
    u = u(:)';
    eps_val = 1e-12;

    n = length(h_norm);
    k_i = min(k_i, n);

    % Compute the coverage suppression factor.
    cover_penalty = 1 ./ (1 + u);

    % Initialize the neighborhood suppression factors.
    ell = ones(1, n);

    % Determine the neighborhood radius from the target number of selected points.
    r_i = max(1, round(n / (2 * k_i)));

    pos = zeros(1, k_i);

    for k = 1:k_i
        % Compute the joint sampling weights.
        p = h_norm .* cover_penalty .* ell;
        p(p < 0) = 0;

        if sum(p) < eps_val
            pos = unique(pos(1:k-1), 'stable');
            return;
        end

        p = p / sum(p);

        % Sample from the cumulative distribution.
        cdf = cumsum(p);
        rand_value = rand;
        selected_point = find(cdf >= rand_value, 1, 'first');

        pos(k) = selected_point;

        % Exclude the selected point from subsequent sampling.
        ell(selected_point) = 0;

        % Compute the neighborhood shrinkage factor.
        relative_hpa = h_norm(selected_point) / (max(h_norm) + eps_val);
        gamma_s = 1 - relative_hpa;     
        gamma_s = max(0.2, gamma_s); % Lower bound on the shrinkage factor

        % Update the neighborhood suppression factors.
        left_idx = max(1, selected_point - r_i);
        right_idx = min(n, selected_point + r_i);

        neigh = left_idx:right_idx;
        neigh(neigh == selected_point) = [];

        ell(neigh) = gamma_s * ell(neigh);
    end

    pos = unique(pos, 'stable');
end