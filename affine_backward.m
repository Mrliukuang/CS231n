function [dX_affined, dW2, db2] = affine_backward(X_affined, dscores, W2)
    % da is daffine_cache
    dW2 = X_affined * dscores';
    db2 = sum(dscores, 2);
    dX_affined = W2 * dscores;
end