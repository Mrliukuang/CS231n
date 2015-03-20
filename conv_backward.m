function [dX_out, dW, db] = conv_backward(X, X_cols, dX_conv, W, conv_param)
    [filter_h, filter_w, C, filter_n] = size(W);
    % 1. reshape dX_conv to dX_cols
    dX_cols = permute(dX_conv, [3, 4, 1, 2]);
    dX_cols = reshape(dX_cols, filter_n, []);

    % 2. get dW_r
    dW_r = dX_cols * X_cols';
    
    % 3. restore dW1
    dW = reshape(dW_r', [filter_h, filter_w, C, filter_n]);
    
    % 4. get db1
    db = sum(dX_cols, 2);
    
    % 5. get dX_in
    W_r = reshape(W, [], filter_n)';
    dX_out = W_r' * dX_cols;
    dX_out = col_2_im(X, dX_out, filter_h, filter_w, conv_param);
end
