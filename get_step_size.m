function get_step_size(X, y_train, W1, W2, dW1, dW2, model)
    [~, N] = size(X);
    for i = -5:2
        step_size = 10^i;
        W1_new = W1 - dW1 * step_size;
        W2_new = W2 - dW2 * step_size;
        
        H = max(0, W1_new * X);
        H = [ones(1, N); H];
    
        S = max(0, W2_new * H);
        fprintf('for step_size: %f, the loss = %f\n',...
                step_size, svm_loss(S, y_train));
        
    end
end