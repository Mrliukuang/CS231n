function ratio = predict_val(X_val, y_val, W1, b1, W2, b2, W3, b3, conv_param1, conv_param2, pool_param)
    rs = zeros(50, 1);
    for i = 1:50
        batch_mask = 200*(i-1)+1 : 200*i;
        X_batch = X_val(:, :, :, batch_mask);
        y_batch = y_val(batch_mask);
        
        [X_conv1, ~] = conv_forward(X_batch, W1, b1, conv_param1);
    
        X_relu1 = max(0, X_conv1);
        
        [X_conv2, ~] = conv_forward(X_relu1, W2, b2, conv_param2);
        X_relu2 = max(0, X_conv2);
        
        [X_pool, ~] = MaxPooling(double(X_relu2), [pool_param.height, pool_param.weight]);
        
        % fully affine
        [scores, ~] = affine_forward(X_pool, W3, b3);
        
        [~, pred] = max(scores);
        rs(i) = mean(pred-1 == y_batch');
        fprintf('val itr #%d: acc = %f\n', i, rs(i));
    end
    
    ratio = mean(rs);
end