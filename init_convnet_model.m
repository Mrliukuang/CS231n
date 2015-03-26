function model = init_convnet_model()
    % X: Training data of size [H, W, C, N].

    % Layer1. Conv layer
    conv_param1.conv_size = [5, 5, 1, 32];
    conv_param1.stride = 1;
    conv_param1.pad = (5-1)/2;
    conv_layer_1 = init_layer('conv', conv_param1);

    % Layer2. Conv layer
    conv_param2.conv_size = [5, 5, 32, 32];
    conv_param2.stride = 1;
    conv_param2.pad = (5-1)/2;
    conv_layer_2 = init_layer('conv', conv_param2);
    
    % Layer3. Pool layer
    pool_param.pool_size = [2, 2];
    pool_param.stride = 1;  % pooling doesn't support stride now
    pool_layer_1 = init_layer('pool', pool_param);
    
    % Layer4. Conv layer
    conv_param3.conv_size = [3, 3, 32, 32];
    conv_param3.stride = 1;
    conv_param3.pad = (3-1)/2;
    conv_layer_3 = init_layer('conv', conv_param3);
    
    % Layer5. Conv layer
    conv_param4.conv_size = [3, 3, 32, 32];
    conv_param4.stride = 1;
    conv_param4.pad = (3-1)/2;
    conv_layer_4 = init_layer('conv', conv_param4);
    
    % Layer6. Pool layer
    pool_layer_2 = init_layer('pool', pool_param);

    % Layer7. FC layer
    % fc_param = 16*32*32/4;
    fc_param = 32*12*12;
    fc_layer = init_layer('fc', fc_param);

    % Combine all layers into the model
    model.layer_num = 7;
    model.layer{1} = conv_layer_1;
    model.layer{2} = conv_layer_2;
    model.layer{3} = pool_layer_1;
    model.layer{4} = conv_layer_3;
    model.layer{5} = conv_layer_4;
    model.layer{6} = pool_layer_2;
    model.layer{7} = fc_layer;

end