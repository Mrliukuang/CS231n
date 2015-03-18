function model = init_convnet_model(X_sample)
    % X_sample.  input sample of 3 dimensional.
    weight_scale = 1e-3;
    bias_scale = 0;
    
    num_classes = 10;
    num_filters = 50;
    filter_size = 5;    % 5*5 filter, we assume filter_width = filter_height
    
    [H, W, C] = size(X_sample);
    model.W{1} = weight_scale*randn(5, 5, 3, 32);
    model.b{1} = bias_scale*randn(32, 1);
    model.W{2} = weight_scale*randn(3, 3, 32, 16);
    model.b{2} = bias_scale*randn(16, 1);
    
    model.W{3} = weight_scale*randn(16*H*W/4, num_classes);
    model.b{3} = bias_scale*randn(num_classes, 1);

    
%     model.W{1} = weight_scale*randn(filter_size, filter_size, C, num_filters);
%     model.b{1} = bias_scale*randn(num_filters, 1);
%     
%     model.W{2} = weight_scale*randn(num_filters*H*W/4, num_classes);
%     model.b{2} = bias_scale*randn(num_classes, 1);
end