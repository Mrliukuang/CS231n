% clear; close all; clc;

% load CIFAR10_3D_zero_meaned.mat;

conv_param.stride = 1;
conv_param.pad = 0;

for i=1:100
    fprintf('%d\n', i);
    batch_mask = randi(50000, 200, 1);
    X_batch = X_tr(:, :, :, batch_mask);
    cols = im_2_col(X_batch, 3, 3, conv_param);
    X_batch = col_2_im(X_batch, cols, 3, 3, conv_param);
end