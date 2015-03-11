function dX_conv = relu_backward(dX_relu, X_conv)
    % when X_conv > 0: dX_conv = dX_relu
    % otherwise:       dX_conv = 0
    dX_conv = zeros(size(dX_relu));
    pos_ind = find(X_conv > 0);
    dX_conv(pos_ind) = dX_relu(pos_ind);
end