function dX_relu = max_pool_backward(X_pool, dX_pool, max_ind, pool_param)
    [HH, WW, C, N] = size(X_pool);
    pool_h = pool_param.height;
    pool_w = pool_param.weight;
    
    H = HH * pool_h;
    W = WW * pool_w;

    dX_p = repmat(dX_pool, pool_h, pool_w);
    dX_relu = zeros(H, W, C, N);
    dX_relu(max_ind) = dX_p(max_ind);
    
    
%     dX_relu = zeros(size(X_relu));
%     
% 
%     for i = 1:WW
%         for j = 1:HH
%             % map out index to in index
%             x = i + (i-1)*(pool_w-1);
%             y = j + (j-1)*(pool_h-1);
% 
%             cube = X_relu(y:y+pool_h-1, x:x+pool_w-1, :, :);
%             max_val = X_pool(j, i, :, :);
% 
%             mask = bsxfun(@eq, cube, max_val);
%             
%             % TODO: for several same max value, how to backprop the gradient?
%             % #1: equally distribute the gradient
%             % #2: randomly choose one, like the first one (seems this is prefered!)
%             % which is reasonable?
%             % It seems which way doesn't matter, cause the ReLU layer
%             % before will set all zero-value gradients to zero! So who cares!
%             
%             
%             %scale = sum(reshape(mask, [], 1, C, N), 1);
%             %t = bsxfun(@times, mask, dX_pool(j, i, :, :));
%             %dX_relu(y:y+pool_h-1, x:x+pool_w-1, :, :) = bsxfun(@rdivide, t, scale);
%            
%             dX_relu(y:y+pool_h-1, x:x+pool_w-1, :, :) = bsxfun(@times, mask, dX_pool(j, i, :, :));
%         end
%     end

end



% function dX_relu = max_pool_backward(X_relu, X_pool, dX_pool, pool_param)
%     [H, W, C, N] = size(X_pool);
%     pool_h = pool_param.height;
%     pool_w = pool_param.weight;
% 
%     dX_relu = zeros(size(X_relu));
% 
%     for i = 1:W
%         for j = 1:H
%             % map out index to in index
%             x = i + (i-1)*(pool_w-1);
%             y = j + (j-1)*(pool_h-1);
% 
%             cube = X_relu(y:y+pool_h-1, x:x+pool_w-1, :, :);
%             max_val = X_pool(j, i, :, :);
% 
%             mask = bsxfun(@eq, cube, max_val);
%             
%             % TODO: for several same max value, how to backprop the gradient?
%             % #1: equally distribute the gradient
%             % #2: randomly choose one, like the first one (seems this is prefered!)
%             % which is reasonable?
%             % It seems which way doesn't matter, cause the ReLU layer
%             % before will set all zero-value gradients to zero! So who cares!
%             
%             
%             %scale = sum(reshape(mask, [], 1, C, N), 1);
%             %t = bsxfun(@times, mask, dX_pool(j, i, :, :));
%             %dX_relu(y:y+pool_h-1, x:x+pool_w-1, :, :) = bsxfun(@rdivide, t, scale);
%            
%             dX_relu(y:y+pool_h-1, x:x+pool_w-1, :, :) = bsxfun(@times, mask, dX_pool(j, i, :, :));
%         end
%     end
% 
% end

