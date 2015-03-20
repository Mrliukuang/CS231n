function [W, b] = update_param(W_old, b_old, dW, db, step_size)
    %% update the weights.
    %     step_cache{1} = step_cache{1} * 0.95 - dW1 * step_size;
    %     step_cache{2} = step_cache{2} * 0.95 - db1 * step_size;
    %     step_cache{3} = step_cache{3} * 0.95 - dW2 * step_size;
    %     step_cache{4} = step_cache{4} * 0.95 - db2 * step_size;
    %     W1 = W1 + step_cache{1};
    %     b1 = b1 + step_cache{2};
    %     W2 = W2 + step_cache{3};
    %     b2 = b2 + step_cache{4};
    W = W_old - step_size * dW;
    b = b_old - step_size * db;
end