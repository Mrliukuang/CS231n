function X = reshape_data(X_old)
    % reshaped data X is of 4 dimension. [H * W * C * N]
    % H = image weight
    % W = image height
    % C = # of colors   i.e. RGB: C = 3
    % N = # of examples
    
    [L, N] = size(X_old);
    H = sqrt(L);
    W = H;
    C = 1;
    
    X = zeros(H, W, C, N);
    %% reshape all date to 4D
    for i = 1:N
        % fprintf('reshaping image %d...\n', i);
        x = X_old(:, i);
        X(:, :, C, i) = reshape(x, [H, W]);
    end
end