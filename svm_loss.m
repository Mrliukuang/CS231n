function [loss, grad] = svm_loss(scores, y)
    %  scores: the output of the NN of size [K, N].
    %  y: the target output of size [N, 1] and starts from 0.
    [N, ~] = size(y);
    
    y_idx = sub2ind(size(scores), double(y' + 1), 1:N);
    scores_y = scores(y_idx);

    diff = bsxfun(@minus, scores, scores_y);
    margins = max(0, diff + 1.0);   % note delta = 1.0
    margins(y_idx) = 0;
    
    % sumof = @(w)(sum(sum(w.^2)));       % anonymous function for regression
    loss = sum(sum(margins)) / N;
           % + 0.5*model.reg*(sumof(model.W1) + sumof(model.W2));  % regression term
    
    % grad = d(loss)/d(scores)
    grad = zeros(size(scores));
    grad(margins>0) = 1;
    grad(y_idx) = grad(y_idx) - sum(double(margins>0));
    grad = grad / N;
end
