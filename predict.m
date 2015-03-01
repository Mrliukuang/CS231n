function ratio = predict(X, y_train, W1, W2)
    [~, N] = size(X);
    H = max(0, W1*X);
    S = W2*[ones(1,N); H];
    [~, pred] = max(S);
    
    ratio = mean(pred-1 == y_train');
end