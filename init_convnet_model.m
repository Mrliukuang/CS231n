function model = init_convnet_model()
    % X: Training data of size [H, W, C, N].
    
%     % Input layer
%     input_layer = init_layer('input', X_size);

    % Conv layer 1
    conv_param1.conv_size = [5, 5, 3, 32];
    conv_param1.stride = 1;
    conv_param1.pad = (5-1)/2;
    conv_layer_1 = init_layer('conv', conv_param1);

    % Conv layer 2
    conv_param2.conv_size = [3, 3, 32, 16];
    conv_param2.stride = 1;
    conv_param2.pad = (3-1)/2;
    conv_layer_2 = init_layer('conv', conv_param2);

    % Pool layer
    pool_param.pool_size = [2, 2];
    pool_param.stride = 1;  % pooling doesn't support stride now
    pool_layer = init_layer('pool', pool_param);

    % FC layer
    fc_param = 16*32*32/4;
    fc_layer = init_layer('fc', fc_param);

    % Combine all layers into the model
    model.layer_num = 4;
    model.layer{1} = conv_layer_1;
    model.layer{2} = conv_layer_2;
    model.layer{3} = pool_layer;
    model.layer{4} = fc_layer;



% X_sample.  input sample of 3 dimensional.
%     weight_scale = 1e-3;
%     bias_scale = 0;
%     
%     num_classes = 10;
%     num_filters = 50;
%     filter_size = 5;    % 5*5 filter, we assume filter_width = filter_height
%     
%     [H, W, C] = size(X_sample);
%     model.W{1} = weight_scale*randn(5, 5, 3, 32);
%     model.b{1} = bias_scale*randn(32, 1);
%     model.W{2} = weight_scale*randn(3, 3, 32, 16);
%     model.b{2} = bias_scale*randn(16, 1);
%     
%     model.W{3} = weight_scale*randn(16*H*W/4, num_classes);
%     model.b{3} = bias_scale*randn(num_classes, 1);

    
%     model.W{1} = weight_scale*randn(filter_size, filter_size, C, num_filters);
%     model.b{1} = bias_scale*randn(num_filters, 1);
%     
%     model.W{2} = weight_scale*randn(num_filters*H*W/4, num_classes);
%     model.b{2} = bias_scale*randn(num_classes, 1);
end