% clear;
clc; close all;
addpath('.\util')
% load convdata.mat;
% load XY128
% load CIFAR10_3D_zero_meaned.mat;

% X_tr = double(X_tr);

% load validation data
% val_ind = randi(N, 500, 1);
% X_val = X_tr(:, :, :, 1:10000);
% y_val = y_tr(1:10000);

% load training data
% X = X_tr(:,:,:, 10001:end);
% y = y_tr(10001:end);
% 
% clear X_tr y_tr;

% init model
model = init_convnet_model();
% X = double(X);
reg = 1e-6;

% H = height
% W = weight
% C = image channels, 1 for gray image, 3 for RGB image
% N = # of examples
[H, W, C, N] = size(X);

best_acc = 0;
best.loss = Inf;

epoch_num = 20;
batch_size = 128;
its_per_epoch = uint32(N / batch_size);
num_iters = epoch_num * its_per_epoch;
lr = 0.005;   % learning rate
lr_decay = 0.95;

% step_cache{1} = zeros(size(W1));
% step_cache{2} = zeros(size(b1));
% step_cache{3} = zeros(size(W2));
% step_cache{4} = zeros(size(b2));

loss_history = zeros(num_iters, 1);
val_acc_history = zeros(epoch_num, 1);

% X_batch = X_val;
% y_batch = y_val;

batch_mask = randi(N, 20, 1);
% X_batch = X(:, :, :, batch_mask);
% y_batch = y(batch_mask);
X_batch = X(:,:,:,1:50);
y_batch = y(1:50);


for i = 1:500
    %% sample one batch examples.
%     batch_mask = randi(N, batch_size, 1);
%     X_batch = X(:, :, :, batch_mask);
%     y_batch = y(batch_mask);
  
    [loss, scores, dscores, model] = pass_forward(X_batch, y_batch, model);
    model = pass_backward(model, dscores);
    
    % Add regularization
    [reg_loss, model] = add_reg(model, reg);

    loss_history(i) = loss + reg_loss;
    
    acc = predict(scores, y_batch);
    fprintf('#%d: loss: %f, acc: %f | bestloss: %f, best_acc: %f\n', i, loss, acc, best.loss, best_acc)
    if loss < best.loss
        fprintf('bingo!\n')
        best.loss = loss;
        best_model = model;
    end
    
    if acc > best_acc
        fprintf(' Yes!\n')
        best_acc = acc;
    end
    
    % test parameters on validation data
%     if mod(i, its_per_epoch) == 0
%         fprintf('VALIDATION! #%d\n', i/its_per_epoch);
%         val_acc_history(i) = predict_val(X_val, y_val, W1, b1, W2, b2, W3, b3, conv_param1, conv_param2, pool_param);
%     end

    model = update_model(model, lr);


    
    
end






















