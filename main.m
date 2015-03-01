clear; clc; close all;

% load('../data/data_double.mat');
load('../data/data_norm.mat')
load('../data/X_hog.mat')


% X_train = [zeros(1, N); X_train];

X = get_feature(X_train);
[D, N] = size(X);
X = [ones(1, N); X];

% W1 = randn(50, 324) * 0.001;
% W1 = [zeros(50, 1), W1];
%
% W2 = randn(10, 50) * 0.001;
% W2 = [zeros(10, 1), W2];

N = 50000;
D = 324;
K = 10;

model = init_model(N, D, K);
W1 = model.W1;
W2 = model.W2;

step_size = 1e-2;      % this need be tuned! carefully!
% get_step_size(X, y_train, W1, W2, dW1, dW2, model)

reg = model.reg;

bestloss = Inf;
for i = 1:5000
    H = max(0, W1*X);     % size = 50 * 50000
    H = [ones(1, N); H];
    
    S = max(0, W2*H);
    [loss, dS] = svm_loss(S, y_train);   % TODO. model doesn't update.
   
    dW2 = dS * H';
    dH = W2' * dS;  % 51 * 50000
    
    dH(H==0) = 0;   % I think this should be W2*H==0, but I think H==0 is OK & more efficient.
    dH = dH(2:end, :);
    dW1 = dH*X';  % 50 * 325
    
    fprintf('trial # %d; loss: %f; best loss: %f\n', i, loss, bestloss)
    if loss < bestloss
        fprintf('Bingo!\n')
        bestloss = loss;
        if mod(i, 10) == 0
            bestW1 = W1;
            bestW2 = W2;
        end
    end
    W1 = W1 - dW1 * step_size;
    W2 = W2 - dW2 * step_size;
end













