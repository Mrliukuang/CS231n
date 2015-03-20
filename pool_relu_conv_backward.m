function [dX_out, dW, db] = pool_relu_conv_backward(dX_affine, cache, W1, pool_param)
    X_cols = cache{1};
    X_conv = cache{2};
%     X_relu = cache{3};
    X_pool = cache{4};
    max_ind = cache{5};

    if ndims(dX_affine) == 2
        % back pass reshape layer
        dX_pool = reshape(dX_affine, size(X_pool));
    else
        dX_pool = dX_affine;
    end
    
    % back pass pooling layer
    % Note.  when gradient_check dX_relu, need to choose those !=0 values as check_ind.
    dX_relu = max_pool_backward(X_pool, dX_pool, max_ind, pool_param);
    
    % back pass ReLU layer
    dX_conv = relu_backward(dX_relu, X_conv);
    
    % back pass Conv layer
    [dX_out, dW, db] = conv_backward(X_cols, dX_conv, W1);
end