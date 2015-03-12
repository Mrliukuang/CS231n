% clear; 
clc; close all;

% load convdata.mat;
model = init_convnet_model(X(:,:,:,1));

[H, W, C, N] = size(X); % 28*28*1*60000
% H = height
% W = weight
% C = image channels, 1 for gray image, 3 for RGB image
% N = # of examples

% sample one batch
N = 100;
X_batch = X(:,:,:,1:N);
y_batch = y_tr(1:N, :);

W1 = model.W{1};    % 5*5*1*32
b1 = model.b{1};    % 32*1
W2 = model.W{2};    % 6272*10
b2 = model.b{2};    % 10*1

% filter_h == filter_w, the filter is square
[filter_h, filter_w, ~, filter_n] = size(W1);

% convolute params.
conv_param.stride = 1;
conv_param.pad = (filter_h-1)/2;

% HH = (H + 2 * conv_param.pad - filter_h) / conv_param.stride + 1;
% WW = (W + 2 * conv_param.pad - filter_w) / conv_param.stride + 1;

% pool params.
pool_param.stride = 1;
pool_param.height = 2;
pool_param.weight = 2;

% forward pass.
[a, cache] = conv_relu_pool_forward(X_batch, W1, b1, conv_param, pool_param);
X_cols = cache{1};
X_conv = cache{2};
X_relu = cache{3};
X_pool = cache{4};

% fully affine
[scores, X_affine] = affine_forward(a, W2, b2);
 
% back prop
[loss, dscores] = softmax_loss(scores, y_batch);

% back pass affine layer
% checked!
[dX_affine, dW2, db2] = affine_backward(X_affine, dscores, W2);

% back pass reshape layer
% checked!
dX_pool = reshape(dX_affine, size(X_pool));

% back pass pooling layer
% checked!
% Note.  when gradient_check dX_relu, need to choose those !=0 values as check_ind.
dX_relu = max_pool_backward(X_relu, X_pool, dX_pool, pool_param);

% back pass ReLU layer
dX_conv = relu_backward(dX_relu, X_conv);

% back pass Conv layer
[dW1, db1] = conv_backward(X_cols, dX_conv, W1);























