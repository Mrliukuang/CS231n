clear; clc; close all;

load convdata.mat;
model = init_convnet_model(X(:,:,:,1));

[H, W, C, N] = size(X); % 28*28*1*60000
% H = height
% W = weight
% C = image channels, 1 for gray image, 3 for RGB image
% N = # of examples

W1 = model.W{1};    % 32*1*5*5
b1 = model.b{1};    % 32*1
W2 = model.W{2};    % 6272*10
b2 = model.b{2};    % 10*1

[filter_h, filter_w, ~] = size(W1);

% convolute params.
conv_param.stride = 1;
conv_param.pad = (filter_h-1)/2;

% pool params.
pool_param.stride = 1;
pool_param.height = 2;
pool_param.weight = 2;

% forward pass.
X_batch = X(:,:,:,1:100);
[a, ~] = conv_relu_pool_forward(X_batch, W1, b1, conv_param, pool_param);
















