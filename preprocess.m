function pX = preprocess(X)
    % data in X is arranged as columns.
    % X is single. cause of 'out-of-memory error'.
    mu = mean(X, 2);
    
    diff = bsxfun(@minus, X, mu);
    sigma = std(X, 0, 2);
    
    pX = bsxfun(@rdivide, diff, sigma);
end