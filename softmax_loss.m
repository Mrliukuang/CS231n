% function [loss, dW] = softmax_loss(X, y, W)
%     [~, N] = size(X);
%     X = double(X);
%     reg = 1e-5;
%     
%     scores = W * X;
%     softmax_X = softmax(scores);
%     
%     y_idx = sub2ind(size(scores), double(y'+1), 1:N);
%     loss = mean(-log(softmax_X(y_idx))) + 0.5*reg*sum(sum(W.*W));
%     
%     coef = softmax_X;
%     coef(y_idx) = coef(y_idx) - 1;
%     dW = coef * X' / N + reg * W;
% end

function [loss, grad] = softmax_loss(scores, y)
    %  Just like 'svm_loss.m':
    %  scores: the output of the NN of size [K, N].
    %  y: the target output of size [N, 1] and starts from 0.
    
    [N, ~] = size(y);
    y_idx = sub2ind(size(scores), double(y'+1), 1:N);
    reg = 0;
    
    probs = softmax(scores);
    loss = -mean(log(probs(y_idx)));
            % + 0.5*reg*sum(sum(W.*W));
    
    grad = probs;
    grad(y_idx) = grad(y_idx) - 1;
    grad = grad / N;
end