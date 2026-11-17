function metrics = regression_metrics(y_pred, y_true)
% Regression performance evaluation.
% Inputs: y_pred - predicted values; y_true - reference values.
% Output: metrics - structure containing MAE, MAPE (ratio), MSE, RMSE, and R2.

validateattributes(y_pred, {'numeric'}, ...
    {'real', 'vector', 'nonempty', 'finite'}, mfilename, 'y_pred', 1);
validateattributes(y_true, {'numeric'}, ...
    {'real', 'vector', 'nonempty', 'finite'}, mfilename, 'y_true', 2);

if numel(y_pred) ~= numel(y_true)
    error('regression_metrics:SizeMismatch', ...
        'y_pred and y_true must contain the same number of values.');
end

y_pred = double(y_pred(:));
y_true = double(y_true(:));
residuals = y_pred - y_true;

metrics.MAE = mean(abs(residuals));
metrics.MAPE = mean(abs(residuals ./ (y_true + eps)));
metrics.MSE = mean(residuals.^2);
metrics.RMSE = sqrt(metrics.MSE);

SSE = sum(residuals.^2);
SST = sum((y_true - mean(y_true)).^2);
if SST == 0
    metrics.R2 = NaN;
else
    metrics.R2 = 1 - SSE / SST;
end
end
