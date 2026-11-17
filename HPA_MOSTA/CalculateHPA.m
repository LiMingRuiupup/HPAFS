function [h] = CalculateHPA(X_tr, y_tr)
% Harmonic point attention (HPA) calculation.
% Inputs: X_tr - N_tr-by-n training sample matrix; y_tr - N_tr-by-1 concentration label vector.
% Output: h - 1-by-n HPA vector.

[N_tr, n] = size(X_tr);

    concentration_levels = unique(y_tr);
    G = length(concentration_levels);

    eps_val = 1e-12; % Numerical stabilization constant

    % Set the denominator floor for relative variation.
    tau = 0.01 * median(abs(X_tr(:)));
    if tau < 1e-8
        tau = 1e-8;
    end

    % Initialize the HPA component metrics.
    d = zeros(1, n);   % Concentration discrimination
    s = zeros(1, n);   % Measurement stability

    % Identify samples and sample counts at each concentration level.
    group_idx = cell(G, 1);
    N_g = zeros(G, 1);
    for g = 1:G
        group_idx{g} = (y_tr == concentration_levels(g));
        N_g(g) = sum(group_idx{g});
    end

    % Compute the overall mean at each sampling point.
    mu_global = mean(X_tr, 1);

    % Compute concentration discrimination.
    for j = 1:n
        SS_between = 0;
        SS_within = 0;

        for g = 1:G
            x_cj = X_tr(group_idx{g}, j);
            mu_jc = mean(x_cj);

            SS_between = SS_between ...
                + N_g(g) * (mu_jc - mu_global(j))^2;
            SS_within = SS_within ...
                + sum((x_cj - mu_jc).^2);
        end

        df_between = G - 1;
        df_within = N_tr - G;

        B_j = SS_between / max(df_between, 1);
        W_j = SS_within / max(df_within, 1);

        d(j) = B_j / (W_j + eps_val);
    end

    % Compute measurement stability.
    for j = 1:n
        cv_sum = 0;

        for g = 1:G
            x_cj = X_tr(group_idx{g}, j);

            mu_jc = mean(x_cj);
            sigma_jc = std(x_cj, 1);   % Population standard deviation

            denom = max(abs(mu_jc), tau);
            cv_jc = sigma_jc / denom;

            cv_sum = cv_sum + cv_jc;
        end

        cv_bar = cv_sum / G;
        s(j) = 1 / (cv_bar + eps_val);
    end

    % Normalize the component metrics.
    d_norm = minmax_norm(d);
    s_norm = minmax_norm(s);

    % Compute the adaptive fusion weights.
    std_d = std(d_norm, 1);
    std_s = std(s_norm, 1);

    if (std_d + std_s) < eps_val
        alpha = 0.5;
        beta = 0.5;
    else
        alpha = std_d / (std_d + std_s);
        beta = std_s / (std_d + std_s);
    end

    % Compute the HPA vector.
    h = alpha * d_norm + beta * s_norm;

end