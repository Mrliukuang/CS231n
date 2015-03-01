clc; close all;

N = 50000;
D = 324;
K = 10;

model = init_model(N, D, K);
W1 = model.W1;
W2 = model.W2;

H1 = W1*X;
H2 = max(0, H1);
H2 = [ones(1, N); H2];

S1 = W2*H2;
S2 = max(0, S1);

[loss, dS2] = svm_loss(S2, y_train);
dS1 = dS2;
dS1(S1<=0) = 0;

dW2 = dS1 * H2';      
dH2 = W2' * dS1;      

dH1 = dH2(2:end, :);
dH1(H1<=0) = 0;

dW1 = dH1*X';














