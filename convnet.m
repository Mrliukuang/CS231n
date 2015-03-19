% clear;
clc; close all;

% load convdata.mat;
% load CIFAR10_3D_zero_meaned.mat;
% X = X_tr;
model = init_convnet_model(X(:,:,:,1));

% load parameters
W1 = model.W{1};
b1 = model.b{1};
W2 = model.W{2};
b2 = model.b{2};
W3 = model.W{3};
b3 = model.b{3};


% H = height
% W = weight
% C = image channels, 1 for gray image, 3 for RGB image
% N = # of examples
[H, W, C, N] = size(X); % 28*28*1*60000


% filter_h == filter_w, the filter is square
[filter_h1, filter_w1, ~, filter_n1] = size(W1);
[filter_h2, filter_w2, ~, filter_n2] = size(W2);

% convolute params.
conv_param1.stride = 1;
conv_param1.pad = (filter_h1-1)/2;
conv_param2.stride = 1;
conv_param2.pad = (filter_h2-1)/2;

% HH = (H + 2 * conv_param.pad - filter_h) / conv_param.stride + 1;
% WW = (W + 2 * conv_param.pad - filter_w) / conv_param.stride + 1;

% pool params.
pool_param.stride = 1;
pool_param.height = 2;
pool_param.weight = 2;

% load validation data
val_ind = randi(N, 500, 1);
X_val = X(:, :, :, val_ind);
y_val = y_tr(val_ind);

best_acc = 0;
best.loss = Inf;
% best.W1 = W1;
% best.b1 = b1;
% best.W2 = W2;
% best.b2 = b2;

epoch_num = 20;
batch_size = 200;
its_per_epoch = N / batch_size;
num_iters = epoch_num * its_per_epoch;
step_size = 1e-3;   % learning rate
step_size_decay = 0.95;

step_cache{1} = zeros(size(W1));
step_cache{2} = zeros(size(b1));
step_cache{3} = zeros(size(W2));
step_cache{4} = zeros(size(b2));

val_acc_history = zeros(num_iters, 1);
loss_history = zeros(num_iters, 1);


% X_batch = X_val;
% y_batch = y_val;

% batch_mask = randi(N, batch_size, 1);
% X_batch = X(:, :, :, batch_mask);
% y_batch = y_tr(batch_mask);

for i = 1:5000
    %% sample one batch examples.
    batch_mask = randi(N, batch_size, 1);
    X_batch = X(:, :, :, batch_mask);
    y_batch = y_tr(batch_mask);
    
    
    % forward pass.
    % [a, cache] = conv_relu_pool_forward(X_batch, W1, b1, conv_param, pool_param);
%     [a, cache1] = conv_relu_forward(X_batch, W1, b1, conv_param1);
%     [b, cache2] = conv_relu_pool_forward(a, W2, b2, conv_param2, pool_param);
    [X_conv1, X_cols] = conv_forward(X_batch, W1, b1, conv_param1);
    X_relu1 = max(0, X_conv1);
    
    [X_conv2, X_relu1_cols] = conv_forward(X_relu1, W2, b2, conv_param2);
    X_relu2 = max(0, X_conv2);
    
    [X_pool, max_ind] = MaxPooling(X_relu2, [pool_param.height, pool_param.weight]);
    
    % fully affine
    [scores, X_affine] = affine_forward(X_pool, W3, b3);
    
    % back prop
    [loss, dscores] = softmax_loss(scores, y_batch);
    loss_history(i) = loss;
    
    % test parameters on validation data
    % acc = predict(X_val, y_val, W1, b1, W2, b2, conv_param, pool_param);
    acc = predict(scores, y_batch);
    
    val_acc_history(i) = acc;
    fprintf('#%d: loss: %f | accuracy: %f | bestloss: %f | best_acc: %f\n', i, loss, acc, best.loss, best_acc)
    if loss < best.loss
        fprintf('bingo!\n')
        best.loss = loss;
        %         best.W1 = W1;
        %         best.b1 = b1;
        %         best.W2 = W2;
        %         best.b2 = b2;
    end
    
    if acc > best_acc
        fprintf(' Yes!\n')
        best_acc = acc;
    end
    
    % back pass affine layer
    [dX_affine, dW3, db3] = affine_backward(X_affine, dscores, W3);
    dX_pool = reshape(dX_affine, size(X_pool));
    
    % back pass pool layer
    dX_relu2 = max_pool_backward(X_pool, dX_pool, max_ind, pool_param);
    
    % back pass relu layer
    dX_conv2 = relu_backward(dX_relu2, X_conv2);
    
    % back pass conv layer
    
    
    
%     % backward
%     [db, dW2, db2] = pool_relu_conv_backward(dX_affine, cache2, W2, pool_param);
%     [dW1, db1] = relu_conv_backward(db, cache1, W1);
%     
%     % update params
%     [W1, b1] = update_param(W1, b1, dW1, db1, step_size);
%     [W2, b2] = update_param(W2, b2, dW2, db2, step_size);
    
    if mod(i, its_per_epoch) == 0
        step_size = step_size * step_size_decay;
    end
    
    
end































