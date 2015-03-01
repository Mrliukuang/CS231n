function [loss, dW] = softmax_loss(X, y, W)
    [~, N] = size(X);
    X = double(X);
    reg = 1e-5;
    
    scores = W * X;
    softmax_X = softmax(scores);
    
    y_idx = sub2ind(size(scores), double(y'+1), 1:N);
    loss = mean(-log(softmax_X(y_idx))) + 0.5*reg*sum(sum(W.*W));
    
    coef = softmax_X;
    coef(y_idx) = coef(y_idx) - 1;
    dW = coef * X' / N + reg * W;
end