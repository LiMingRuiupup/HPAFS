function Z_sparse = sparse_substitute_operator(z, h, N_SE, rho_t)
% Sparse substitute operator.
% Inputs: z - 1-by-n binary candidate solution; h - HPA vector;
%         N_SE - search intensity; rho_t - state transition ratio.
% Output: Z_sparse - N_SE-by-n matrix of generated solutions.

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

    % Determine the base removal count.
    q_drop_base = max(1, floor(n * rho_t));

    Z_sparse = zeros(N_SE, n);

    for i = 1:N_SE
        z_sparse = z;

        I1 = find(z_sparse == 1);   % Selected point indices
        I0 = find(z_sparse == 0);   % Unselected point indices

        if isempty(I1)
            Z_sparse(i, :) = z_sparse;
            continue;
        end

        % Compute the current feature selection rate.
        n_selected = length(I1);
        rho = n_selected / n;

        % Limit the removal count to the available selected points.
        q_drop = min(q_drop_base, length(I1));

        % Determine the addition count from the feature selection rate.
        q_add = floor(q_drop * rho);

        % Keep the addition count smaller than the removal count.
        q_add = min(q_add, q_drop - 1);

        % Limit the addition count to the available unselected points.
        q_add = min(q_add, length(I0));

        % Construct HPA weights for point removal.
        if length(I1) == 1
            drop_pos = I1;
        else
            if max(h_norm(I1)) < eps_val
                drop_w = ones(1, length(I1));
            else
                drop_w = max(h_norm(I1)) - h_norm(I1) + eps_val;    
            end
            drop_w = drop_w / sum(drop_w);

            drop_local = weighted_sample_without_replacement(drop_w, q_drop);
            drop_pos = I1(drop_local);
        end

        z_sparse(drop_pos) = 0;

        % Construct HPA weights for point addition.
        if q_add > 0
            I0_updated = find(z_sparse == 0);

            if ~isempty(I0_updated)
                add_w = h_norm(I0_updated) + eps_val;
                add_w = add_w / sum(add_w);

                q_add = min(q_add, length(I0_updated));
                add_local = weighted_sample_without_replacement(add_w, q_add);
                add_pos = I0_updated(add_local);

                z_sparse(add_pos) = 1;
            end
        end

        % Ensure a nonempty feature subset.
        min_features = 1;
        n_selected_after = sum(z_sparse);
        if n_selected_after < min_features
            remain0 = find(z_sparse == 0);
            if ~isempty(remain0)
                add_more = min(min_features - n_selected_after, length(remain0));
                extra_w = h_norm(remain0) + eps_val;
                extra_w = extra_w / sum(extra_w);
                extra_local = weighted_sample_without_replacement(extra_w, add_more);
                z_sparse(remain0(extra_local)) = 1;
            end
        end

        Z_sparse(i, :) = z_sparse;
    end
end