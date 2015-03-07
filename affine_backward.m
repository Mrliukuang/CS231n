function [da, dW2, db2] = affine_backward(affine_cache, dscores, W2)
    % da is daffine_cache
    dW2 = affine_cache * dscores';
    db2 = sum(dscores, 2);
    da = W2 * dscores;
end