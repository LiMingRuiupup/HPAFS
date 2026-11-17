function Z_swap = swap_operator(z, h, N_SE, rho_t)
% Swap operator.
% Inputs: z - 1-by-n binary candidate solution; h - HPA vector;
%         N_SE - search intensity; rho_t - state transition ratio.
% Output: Z_swap - N_SE-by-n matrix of generated solutions.

z = z(:)';
    h = h(:)';
    n = length(z);
    eps_val = 1e-12;

    % Normalize the HPA weights.
    h(h < 0) = 0;
    if sum(h) < eps_val
        h_norm = ones(1, n) / n;
    else
        h_norm = h / sum(h);
    end

    % Determine the maximum number of swaps.
    q_swap = max(1, floor(n * rho_t / 2));   

    Z_swap = zeros(N_SE, n);

    for i = 1:N_SE
        z_swap = z;

        I1_initial = find(z_swap == 1);   % Selected point indices
        I0_initial = find(z_swap == 0);   % Unselected point indices

        if isempty(I1_initial) || isempty(I0_initial)
            Z_swap(i, :) = z_swap;
            continue;
        end

        q_swap_actual = min([q_swap, length(I1_initial), length(I0_initial)]);

        % Swap selected and unselected points sequentially.
        for k = 1:q_swap_actual

            I1 = find(z_swap == 1);
            I0 = find(z_swap == 0);   

            if isempty(I1) || isempty(I0)
                break;
            end

            % Construct HPA weights for selecting a point to remove.
            if max(h_norm(I1)) < eps_val
                w1 = ones(1, length(I1));
            else
                w1 = max(h_norm(I1)) - h_norm(I1) + eps_val;
            end
            w1 = w1 / sum(w1);

            % Construct HPA weights for selecting a point to add.
            w0 = h_norm(I0) + eps_val;
            w0 = w0 / sum(w0);

            p1_local = weighted_sample_without_replacement(w1, 1);
            p0_local = weighted_sample_without_replacement(w0, 1);

            p1 = I1(p1_local);
            p0 = I0(p0_local);

            % Exchange the binary states.
            tmp_val = z_swap(p1);
            z_swap(p1) = z_swap(p0);
            z_swap(p0) = tmp_val;
        end

        Z_swap(i, :) = z_swap;
    end
end