function Z_init = ADSI(N_set, n, h)
% Adaptive dual-suppression initialization (ADSI).
% Inputs: N_set - solution set size; n - number of harmonic sampling points; h - HPA vector.
% Output: Z_init - N_set-by-n binary initial solution set.

h = h(:)';
    eps_val = 1e-12;

    if length(h) ~= n
        error('The length of h must equal n.');
    end

    % Normalize the HPA weights.
    h(h < 0) = 0;

    if sum(h) < eps_val
        h_norm = ones(1, n) / n;
    else
        h_norm = h / sum(h);
    end

    % Compute the effective support size.
    k_eff = ceil(1 / sum(h_norm.^2));
    k_eff = max(1, min(k_eff, n));

    % Set the size of the sparse stratum.
    N_sparse = floor(0.5 * N_set);
    Z_init = zeros(N_set, n);

    % Initialize cross-solution coverage counts.
    u = zeros(1, n);

    % Generate solutions in the sparse stratum.
    for i = 1:N_sparse
        k_i = randi([1, k_eff], 1, 1);   
        pos = sample_one_solution_adaptive(h_norm, u, k_i);
        Z_init(i, pos) = 1;
        u(pos) = u(pos) + 1;
    end
    % Generate solutions in the denser stratum.
    for i = N_sparse+1:N_set
        k_min = min(n, k_eff + 1);
        k_max = floor((k_eff + n) / 2);

        if k_max < 1
            k_max = 1;
        end
        if k_min > k_max
            k_min = k_max;
        end

        k_i = randi([k_min, k_max], 1, 1);
        pos = sample_one_solution_adaptive(h_norm, u, k_i);
        Z_init(i, pos) = 1;
        u(pos) = u(pos) + 1;
    end
end