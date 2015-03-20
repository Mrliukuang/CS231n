function grad = eval_gradient_b(X, y, check_ind, W1, b1, W2, b2, conv_param)
    h = 1e-5;
    
    grad = zeros(size(b1));
    for i = 1:numel(check_ind)
        [sub_1, sub_2] = ind2sub(size(b1), check_ind(i));
        b_1 = b1;
        b_2 = b1;        
        b_1(sub_1, sub_2) = b_1(sub_1, sub_2) + h;
        b_2(sub_1, sub_2) = b_2(sub_1, sub_2) - h;
        
        X_conv1 = conv_forward(X, W1, b_1, conv_param);
        X_conv2 = conv_forward(X, W1, b_2, conv_param);
        
        loss1 = softmax_loss(affine_forward(X_conv1, W2, b2), y);
        loss2 = softmax_loss(affine_forward(X_conv2, W2, b2), y);
        grad(sub_1, sub_2) = (loss1 - loss2)/(2*h);
        fprintf('processing ind %d: grad = %f\n', check_ind(i), grad(sub_1, sub_2));
    end
end