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