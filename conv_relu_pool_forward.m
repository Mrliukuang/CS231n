function [a, cache] = conv_relu_pool_forward(X, W, b, conv_param, pool_param)
    % conv
    a = conv_forward(X, W, b, conv_param);
    conv_cache = a;
    
    % ReLU
    a = max(0, a);
    relu_cache = a;
    
    % max-pooling
    a = max_pool_forward(a, pool_param);
    pool_cache = a;
     
    cache{1} = conv_cache;
    cache{2} = relu_cache;
    cache{3} = pool_cache;
end




