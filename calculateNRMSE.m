function error = calculateNRMSE(series1, series2, mode, n)
    % series1: First time series [d, n], prediction
    % series2: Second time series [d, n], ground truth
    % mode: Calculation mode ('element-wise' or 'total')
    % n: Number of previous time steps to consider (valid only in 'total' mode)

    if size(series1) ~= size(series2)
        error('Input time series must have the same dimensions.');
    end

    if nargin < 3
        mode = 'element-wise';
    end

    if nargin < 4 && strcmpi(mode, 'total')
        error('Number of previous time steps (n) must be provided in ''total'' mode.');
    end

    d = size(series1, 1);  % Data dimension
    if strcmpi(mode, 'element-wise')
        n = size(series1, 2);  % Number of time steps
        squaredError = (series1 - series2).^2;
        meanSquaredError = sum(squaredError, 'all') / (d * n);
        rmse = sqrt(meanSquaredError);
        maxVal = max(series2, [], 'all');
        error_element_wise = rmse / maxVal;  % Normalized RMSE
        error = error_element_wise;  % Assign the error value
    elseif strcmpi(mode, 'total')
        if n > 0 && n <= size(series1, 2)
            error = zeros(1, n);  % Initialize error vector
            maxVal = max(series2,[],  'all');
            for i = 1:n
                squaredError = (series1(:, 1:i) - series2(:, 1:i)).^2;
                meanSquaredError = sum(squaredError, 'all') / (d * i);
%                 meanSquaredError = sum(squaredError, 'all')/ (d * n);
                rmse = sqrt(meanSquaredError);
                maxVal = max(series2(:, 1:i), [], 'all');
                error(i) = rmse / maxVal;  % Normalized RMSE for i previous time steps
            end
        else
            error('Number of previous time steps (n) must be a positive value less than or equal to the number of time steps.');
        end
    else
        error('Invalid mode. Please choose either ''element-wise'' or ''total''.');
    end
end

