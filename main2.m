%% using Mini-batch instead of the whole dataset to update parameters.
clc; close all;

load('XY.mat');

N = 50000;
D = 324;
K = 10;

model = init_model(N, D, K);
reg = model.reg;

step_size = 1e-2;
step_size_decay = 0.95;

batch_size = 100;
num_epochs=30;

iterations_per_epoch = N / batch_size;

num_iters = 15000;
%num_iters = num_epochs * iterations_per_epoch;

% if you use mini-batch GD, then you can not use loss to evalutate
% converage. you need to use cross-validation accuracy.
val_ind = randi(N, 1000, 1);
X_val = X(:, val_ind);
y_val = y_train(val_ind);

bestloss = inf;
W = model.W;
best_acc = 0;

step_cache{1} = zeros(size(W{1}));
step_cache{2} = zeros(size(W{2}));

val_acc_history = zeros(num_iters, 1);
for it = 1:num_iters
    %% sample one batch examples.
    batch_mask = randi(N, batch_size, 1);
    X_batch = X(:, batch_mask);
    y_batch = y_train(batch_mask);
    
    %% load weights;
    W1 = W{1};
    W2 = W{2};
    
    %% forward pass.
    H = max(0, W1*X_batch);
    H = [ones(1, batch_size); H];
    
    S = max(0, W2*H);
    [loss, dS] = softmax_loss(S, y_batch);
    
    %% backprop.
    dW2 = dS * H';
    dH = W2' * dS;
    
    dH(H==0) = 0;
    dH = dH(2:end, :);
    dW1 = dH*X_batch';
    
    acc = predict(X_val, y_val, W);
    val_acc_history(it) = acc;
    fprintf('epoch %d accuracy: %f, best accuracy: %f\n', it, acc, best_acc)
    if acc > best_acc
        best_acc = acc;
        bestW{1} = W1;
        bestW{2} = W2;
    end
    
    %% update the weights.
    step_cache{1} = step_cache{1} * 0.95 - dW1 * step_size;
    step_cache{2} = step_cache{2} * 0.95 - dW2 * step_size;
    W{1} = W1 + step_cache{1};
    W{2} = W2 + step_cache{2};
end