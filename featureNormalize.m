function [X_norm, mu, sigma] = featureNormalize(X)
% zero mean and scale.
    mu = mean(X);
    X_norm = bsxfun(@minus, X, mu);

% we do not compute 'std' of X_norm directly (sigma = std(X_norm)). cause it will freeze the
% computer. we do it patch by patch.
% Note. on my macbook pro, just use 'sigma = std(X_norm)'. it's OKay!
    [~, N] = size(X_norm);
    patch_size = 10000;
    
    if N <= patch_size
        sigma = std(X_norm);
    else
        % matrix is too big to compute at one time.
        % divide and conquer.
        sigma = [];
        ite_num = floor(N / patch_size);
        rem = mod(N, patch_size);
        
        for i = 1:ite_num
            fprintf('processing patch # %d...\n', i);
            sigma = [sigma, std(X_norm(:, (i-1) * patch_size + 1 : i * patch_size))];
        end
        if rem ~= 0
            fprintf('processing remaining...\n');
            sigma = [sigma, std(X_norm(:, ite_num * patch_size + 1 : end))];
        end
    end
    


    
    
    X_norm = bsxfun(@rdivide, X_norm, sigma);




end
