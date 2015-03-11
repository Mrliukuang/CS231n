function X_pooled = max_pool_forward(X_conved, pool_param)
    % X_conved: conved images now wait for pooling.
    % we adopt distinct pooling strategy.
    % the 'pool_param.stride' is not used.
    pool_h = pool_param.height;
    pool_w = pool_param.weight;
    
    [H, W, C, N] = size(X_conved);
    
    % output size
    if any([mod(H, pool_h), mod(W, pool_w)])
        error('pool size error')
    end
    HH = H/pool_h;
    WW = W/pool_w;
    
    % maximize h-direction first
    X_pooled = reshape(X_conved, pool_h, [], C, N);
    X_pooled = max(X_pooled);
    X_pooled = reshape(X_pooled, HH, [], C, N);
    X_pooled = permute(X_pooled, [2, 1, 3, 4]);
    
    % maximize w-direction second
    X_pooled = reshape(X_pooled, pool_w, [], C, N);
    X_pooled = max(X_pooled);
    X_pooled = reshape(X_pooled, HH, WW, C, N);
    X_pooled = permute(X_pooled, [2, 1, 3, 4]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           A naive method using for loop.    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     X_pooled = zeros(HH, WW, C, N);
%     for i = 1:WW
%         for j = 1:HH
%             % map out index to in index
%             x = i + (i-1)*(pool_w-1);
%             y = j + (j-1)*(pool_h-1);
%             
%             cube = X_conved(y:y+pool_h-1, x:x+pool_w-1, :, :);
%             X_pooled(j, i, :, :) = max(max(cube, [], 2), [], 1);
%         end
%     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end