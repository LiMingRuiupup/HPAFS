function x_norm = minmax_norm(x)
% Min-max normalization.
% Input: x - vector of metric values.
% Output: x_norm - normalized metric vector.

xmin = min(x);
    xmax = max(x);

    if abs(xmax - xmin) < 1e-12
        x_norm = zeros(size(x));
    else
        x_norm = (x - xmin) / (xmax - xmin);
    end
end

