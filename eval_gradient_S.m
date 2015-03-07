%% evaluate dS for S after ReLU. S2 = max(0, S).
% function grad_S = eval_gradient_S(S, y_train, check_ind)
%     h = 1e-5;
%     
%     grad_S = zeros(size(S));
%     for i = 1:numel(check_ind)
%         [sub_1, sub_2] = ind2sub(size(S), check_ind(i));
%         S1 = S;
%         S2 = S;        
%         S1(sub_1, sub_2) = S1(sub_1, sub_2) + h;
%         S2(sub_1, sub_2) = S2(sub_1, sub_2) - h;
% 
%         loss1 = softmax_loss(S1, y_train);
%         loss2 = softmax_loss(S2, y_train);
%         grad_S(sub_1, sub_2) = (loss1 - loss2)/(2*h);
%         fprintf('processing ind %d; grad = %f\n', check_ind(i), grad_S(sub_1, sub_2));
%     end
% end

%% check dpool_cache.
% function grad_S = eval_gradient_S(pool_cache, y_train, check_ind, W2, b2)
%     h = 1e-5;
%     
%     grad_S = zeros(size(pool_cache));
%     for i = 1:numel(check_ind)
%         [sub_1, sub_2] = ind2sub(size(pool_cache), check_ind(i));
%         S1 = pool_cache;
%         S2 = pool_cache;        
%         S1(sub_1, sub_2) = S1(sub_1, sub_2) + h;
%         S2(sub_1, sub_2) = S2(sub_1, sub_2) - h;
% 
%         loss1 = softmax_loss(affine_forward(S1, W2, b2), y_train);
%         loss2 = softmax_loss(affine_forward(S2, W2, b2), y_train);
%         grad_S(sub_1, sub_2) = (loss1 - loss2)/(2*h);
%         fprintf('processing ind %d; grad = %f\n', check_ind(i), grad_S(sub_1, sub_2));
%     end
% end

%% check drelu_cache
function grad_relu = eval_gradient_S(relu_cache, y_train, check_ind, pool_param, W2, b2)
    h = 1e-5;
    
    grad_relu = zeros(size(relu_cache));
    for ii = 1:numel(check_ind)
        [sub_1, sub_2] = ind2sub(size(relu_cache), check_ind(ii));
        S1 = relu_cache;
        S2 = relu_cache;        
        S1(sub_1, sub_2) = S1(sub_1, sub_2) + h;
        S2(sub_1, sub_2) = S2(sub_1, sub_2) - h;
        
        S1 = max_pool_forward(S1, pool_param);
        S2 = max_pool_forward(S2, pool_param);

        loss1 = softmax_loss(affine_forward(S1, W2, b2), y_train);
        loss2 = softmax_loss(affine_forward(S2, W2, b2), y_train);
        grad_relu(sub_1, sub_2) = (loss1 - loss2)/(2*h);
        fprintf('processing ind %d; grad = %f\n', check_ind(ii), grad_relu(sub_1, sub_2));
    end
    
    function out = max_pool_forward(in, pool_param)
        % in: conved images now wait for pooling.
        pool_h = pool_param.height;
        pool_w = pool_param.weight;
        
        [in_h, in_w, C, N] = size(in);
        out_h = in_h/pool_h;
        out_w = in_w/pool_w;
        
        out = zeros(out_h, out_w, C, N);
        for i = 1:out_w
            for j = 1:out_h
                % map out index to in index
                x = i + (i-1)*(pool_w-1);
                y = j + (j-1)*(pool_h-1);
                
                cube = in(y:y+pool_h-1, x:x+pool_w-1, :, :);
                out(j, i, :, :) = max(max(cube, [], 2), [], 1);
            end
        end
    end
end

%% evaluate dS before ReLU. S1 = W2 * H2
% function grad_S = eval_gradient_S(S, y_train, check_ind)
%     h = 1e-7;   % notice h should be tuned to find precise gradient.
%                 % 1e-7 is good for svm_loss.
%     
%     grad_S = zeros(size(S));
%     for i = 1:numel(check_ind)
%         [sub_1, sub_2] = ind2sub(size(S), check_ind(i));
%         S1 = S;
%         S2 = S;        
%         S1(sub_1, sub_2) = S1(sub_1, sub_2) + h;
%         S2(sub_1, sub_2) = S2(sub_1, sub_2) - h;
% 
%         loss1 = softmax_loss(max(0, S1), y_train);  % the only difference.
%         loss2 = softmax_loss(max(0, S2), y_train);
%         grad_S(sub_1, sub_2) = (loss1 - loss2)/(2*h);
%         fprintf('processing ind %d; grad = %f\n', check_ind(i), grad_S(sub_1, sub_2));
%     end
% end