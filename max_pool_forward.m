function [X_pool, ind] = max_pool_forward(X_relu, pool_param)
    % X_conved: conved images now wait for pooling.
    % we adopt distinct pooling strategy.
    % the 'pool_param.stride' is not used.
    
    % function works like sub2ind, but faster! (maybe...)
    get_ind = @(siz, x1, x2, x3, x4) x1 + (x2-1)*siz(1) + (x3-1)*siz(1)*siz(2) + (x4-1)*siz(1)*siz(2)*siz(3);
    
    pool_h = pool_param.height;
    pool_w = pool_param.weight;

    [H, W, C, N] = size(X_relu);

    % output size
    if any([mod(H, pool_h), mod(W, pool_w)])
        error('pool size error')
    end
    HH = H/pool_h;
    WW = W/pool_w;
    
    X1 = reshape(X_relu, pool_h, [], C, N);
    [m1, i1] = max(X1);
    
    X1 = reshape(m1, HH, [], C, N);
    i1 = reshape(i1, HH, [], C, N);
    
    X1 = permute(X1, [2, 1, 3, 4]);
    i1 = permute(i1, [2, 1, 3, 4]);
    
    X1 = reshape(X1, pool_w, [], C, N);
    i1 = reshape(i1, pool_w, [], C, N);
    
    [X_pool, col_ind] = max(X1);
    
    % h_mask = col_ind;
    w_mask = repmat(1:HH*WW, 1, 1, C, N);
    c_mask = repmat(reshape(1:C, 1, 1, []), 1, HH*WW, N);
    c_mask = reshape(c_mask, 1, [], C, N);
    
    n_mask = ones(1, HH*WW, C, N);
    tmp = reshape(repmat(1:N, C, 1), 1, 1, C, N);
    n_mask = bsxfun(@times, n_mask, tmp);
    
    row_ind = i1(get_ind(size(i1), col_ind, w_mask, c_mask, n_mask));
    
    X_pool = reshape(X_pool, WW, HH, C, N);
    row_ind = reshape(row_ind, WW, HH, C, N);
    col_ind = reshape(col_ind, WW, HH, C, N);
    
    X_pool = permute(X_pool, [2,1,3,4]);
    row_ind = permute(row_ind, [2,1,3,4]);
    col_ind = permute(col_ind, [2,1,3,4]);
    
    % map local subscript to global subscript
    delta = pool_h * (0:HH-1)' * ones(1, WW);
    row_ind = bsxfun(@plus, row_ind, delta);

    delta = pool_w * ones(HH, 1) * (0:WW-1);
    col_ind = bsxfun(@plus, col_ind, delta);
    
    % map global subscript to global index
    c_mask = repmat(reshape(1:C, 1, 1, []), HH, WW, N);
    c_mask = reshape(c_mask, HH, WW, C, N);
    
    n_mask = ones(HH, WW, C, N);
    tmp = reshape(repmat(1:N, C, 1), 1, 1, C, N);
    n_mask = bsxfun(@times, n_mask, tmp);
    
    ind = get_ind(size(X_relu), row_ind, col_ind, c_mask, n_mask);
    
    
%     
%     % maximize h-direction first
%     X_pool = reshape(X_relu, pool_h, [], C, N);
%     X_pool = max(X_pool);
%     X_pool = reshape(X_pool, HH, [], C, N);
%     X_pool = permute(X_pool, [2, 1, 3, 4]);
%     
%     % maximize w-direction second
%     X_pool = reshape(X_pool, pool_w, [], C, N);
%     X_pool = max(X_pool);
%     X_pool = reshape(X_pool, HH, WW, C, N);
%     X_pool = permute(X_pool, [2, 1, 3, 4]);
%     
    

% Note act_row & act_col are of 3D, if you wanna use these max index,
% first reshape X into 3D (merge C & N into one dimension)
%     cells = mat2cell(X_relu, pool_h*ones(1, HH), pool_w*ones(1, WW), C, N);
%     [m1, i1] = cellfun(@max, cells, 'UniformOutput', false);
%     [m2, i2] = cellfun(@max, m1, 'UniformOutput', false);
%     X_pool = cell2mat(m2);

%     p1 = cell2mat(i1);
%     p2 = cell2mat(i2);
% 
%     % reshape to 3 dimension
%     p1 = reshape(p1, HH, W, []);
%     p2 = reshape(p2, HH, WW, []);
% 
%     delta = pool_w * ones(HH, 1) * (0:WW-1);
%     p2 = bsxfun(@plus, p2, delta);
% 
%     row_mask = repmat((1:HH)', 1, WW, C*N);
%     thick_mask = reshape(1:C*N, 1, 1, C*N);
%     thick_mask = repmat(thick_mask, HH, WW);
% 
%     act_row = p1(sub2ind(size(p1), row_mask, p2, thick_mask));
% 
% 
%     delta = pool_h * (0:HH-1)' * ones(1, WW);
%     act_row = bsxfun(@plus, act_row, delta);
%     act_col = p2;
% 
%     max_ind.row = act_row;
%     max_ind.col = act_col;
    
    





%     X_pooled = zeros(HH, WW, C, N);
%     for i = 1:WW
%         for j = 1:HH
%             % map out index to in index
%             x = i + (i-1)*(pool_w-1);
%             y = j + (j-1)*(pool_h-1);
%
%             cube = X_relued(y:y+pool_h-1, x:x+pool_w-1, :, :);
%             X_pooled(j, i, :, :) = max(max(cube, [], 2), [], 1);
%         end
%     end

end