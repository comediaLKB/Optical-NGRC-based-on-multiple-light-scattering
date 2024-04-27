function Y = delayTimeSeries(X, s, k)
    % Input:
    %   X: Time series data with dimensions [T, N]
    %   s: Stride size between two time steps
    %   k: Number of time delays
    
    % Get the dimensions of the input time series
    [T, N] = size(X);
    
    % Initialize the delayed time series matrix
%     Y = zeros(T - (k - 1) * s, N, k);
    Y = [];
    
    % Populate the delayed time series matrix
    for i = 1:k
        % Calculate the time index for the current delay
        idxStart = (i - 1) * s + 1; % 1; s+1, 2s+1, ...
        idxEnd = T - (k - i) * s; 

        % Extract the delayed time series for the current delay
%         Y(:, :, i) = X(idx:T - (k - i) * s, :);
        Y = cat(2, Y, X(idxStart:idxEnd, :));
    end
end