function ratio = predict(X, y_train, W)
    [~, N] = size(X);
    
    W1 = W{1};
    W2 = W{2};
    H = max(0, W1*X);
    S = W2*[ones(1,N); H];
    [~, pred] = max(S);
    
    ratio = mean(pred-1 == y_train');
end