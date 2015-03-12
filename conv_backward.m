function [dW1, db1] = conv_backward(X_cols, dX_conv, W1)
    [filter_h, filter_w, C, filter_n] = size(W1);
    % 1. reshape dX_conv to dX_cols
    dX_cols = permute(dX_conv, [3, 1, 2, 4]);
    dX_cols = reshape(dX_cols, filter_n, []);
    % 2. get dW_r
    dW_r = dX_cols * X_cols';
    % 3. restore dW1
    dW1 = reshape(dW_r, [filter_h, filter_w, C, filter_n]);
    % 4. get db1
    db1 = sum(dX_cols, 2);
end