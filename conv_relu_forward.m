function [a, cache] = conv_relu_forward(X, W1, b1, conv_param)
    % conv
    [a, cols] = conv_forward(X, W1, b1, conv_param);
    X_conved = a;
    
    % ReLU
    a = max(0, a);
    X_relued = a;
    
    cache{1} = cols;
    cache{2} = X_conved;
    cache{3} = X_relued;
    
end




