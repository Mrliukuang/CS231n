function layer = init_layer(name, param)
    % name: layer type, one of 'input', 'conv', 'pool', 'fc'
    layer.name = name;  
    
%     if strcmp(name, 'input')
%         % Input layer, param = [H, W, C, N] of input X.
%         layer.X_size = param;
%     
%     else
    if strcmp(name, 'conv')
        % Conv layer
        conv_sz = param.conv_size;  % [filter_h, filter_w, C, filter_n]
        
        layer.W = 1e-3 * randn(conv_sz);
        layer.b = zeros(conv_sz(4), 1);
        layer.conv_size = conv_sz;  % = size(W)
        layer.stride = param.stride;
        layer.pad = param.pad;
        
    elseif strcmp(name, 'pool')
        % Max-pool layer
        layer.pool_size = param.pool_size;  % [pool_h, pool_w];
        layer.stride = param.stride;
    
    elseif strcmp(name, 'fc')
        % FC layer, param = input tall vector length
        num_classes = 10;
        layer.W = 1e-3 * randn(param, num_classes);
        layer.b = zeros(num_classes, 1);
        layer.num_classes = num_classes;
        
    end
    

end