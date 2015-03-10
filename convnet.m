clear; clc; close all;

load convdata.mat;
model = init_convnet_model(X(:,:,:,1));

[H, W, C, N] = size(X); % 28*28*1*60000
% H = height
% W = weight
% C = image channels, 1 for gray image, 3 for RGB image
% N = # of examples

% sample one batch
X_batch = X(:,:,:,1:100);
y_batch = y_tr(1:100, :);
N = 100;


W1 = model.W{1};    % 5*5*1*32
b1 = model.b{1};    % 32*1
W2 = model.W{2};    % 6272*10
b2 = model.b{2};    % 10*1

% filter_h == filter_w, the filter is square
[filter_h, filter_w, ~, filter_n] = size(W1);

% convolute params.
conv_param.stride = 1;
conv_param.pad = (filter_h-1)/2;

HH = (H + 2 * conv_param.pad - filter_h) / conv_param.stride + 1;
WW = (W + 2 * conv_param.pad - filter_w) / conv_param.stride + 1;

% pool params.
pool_param.stride = 1;
pool_param.height = 2;
pool_param.weight = 2;

% forward pass.
[a, cache] = conv_relu_pool_forward(X_batch, W1, b1, conv_param, pool_param);
conv_cache = cache{1};
relu_cache = cache{2};
pool_cache = cache{3};

cols = im_2_col(X_batch, filter_h, filter_w, conv_param);
W1_r = reshape(W1, [], filter_n)';
b = bsxfun(@plus, W1_r * cols, b1);
b = col_2_im(b, HH, WW, filter_n, N);


% 
% 
% % fully affine
% [scores, affine_cache] = affine_forward(a, W2, b2);
%  
% % back prop
% [loss, dscores] = softmax_loss(scores, y_batch);
% 
% % back pass affine layer
% % checked!
% [da, dW2, db2] = affine_backward(affine_cache, dscores, W2);
% 
% % back pass reshape layer
% % checked!
% [h, w, C, N] = size(pool_cache);
% dpool_cache = reshape(da, [h, w, C, N]);
% 
% % back pass pooling layer
% mask = ones(pool_param.height, pool_param.weight);
% drelu_cache = zeros(h*pool_param.height, w*pool_param.weight, C, N);
% 
% for i=1:C
%     for j=1:N
%         mask = relu_cache(:,:,i,j)>0;
%         dpool_rep = kron(dpool_cache(:,:,i,j), ones(pool_param.height, pool_param.weight));
%         drelu_cache(:,:,i,j) = mask .* dpool_rep;
%     end
% end
% 

























