function [a, cache] = conv_relu_pool_forward(X, W, b, conv_param, pool_param)
    % conv
    [a, cols] = conv_forward(X, W, b, conv_param);
    X_conved = a;
    
    % ReLU
    a = max(0, a);
    X_relued = a;
    
    % max-pooling
    a = max_pool_forward(a, pool_param);
    X_pooled = a;
     
    cache{1} = cols;
    cache{2} = X_conved;
    cache{3} = X_relued;
    cache{4} = X_pooled;
end




