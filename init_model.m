function model = init_model(N, D, K)
    % 1 hidden layer model.
    model.N = N;
    model.D = D;
    model.K = K;
    model.hidden_n = 50;
    % model.reg = 1e-3;
    model.reg = 0;
    
    W1 = randn(model.hidden_n, D) * 0.001;  % set weights are small random number
    W1 = [zeros(model.hidden_n, 1), W1];    % set biases all to zero
    model.W1 = W1;
    
    W2 = randn(K, model.hidden_n) * 0.001;
    W2 = [zeros(K, 1), W2];
    model.W2 = W2;
end