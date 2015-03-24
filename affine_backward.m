function [dX_affine, dW, db] = affine_backward(dscores, layer)
    X_affine = layer.X_affine;
    W = layer.W;
    
    dW = X_affine * dscores';
    db = sum(dscores, 2);
    dX_affine = W * dscores;
end