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

% check dX_affine
% function grad_S = eval_gradient_S(X_affined, y_train, check_ind, W2, b2)
%     h = 1e-5;
%     X_affined = reshape(X_affined, [32,32,32,10]);
%     grad_S = zeros(size(X_affined));
%     for i = 1:numel(check_ind)
%         [sub_1, sub_2] = ind2sub(size(X_affined), check_ind(i));
%         S1 = X_affined;
%         S2 = X_affined;        
%         S1(sub_1, sub_2) = S1(sub_1, sub_2) + h;
%         S2(sub_1, sub_2) = S2(sub_1, sub_2) - h;
%         
%         loss1 = softmax_loss(affine_forward(S1, W2, b2), y_train);
%         loss2 = softmax_loss(affine_forward(S2, W2, b2), y_train);
%         grad_S(sub_1, sub_2) = (loss1 - loss2)/(2*h);
%         fprintf('processing ind %d; grad = %f\n', check_ind(i), grad_S(sub_1, sub_2));
%     end
% end

% function grad_S = eval_gradient_S(X_affined, y_train, check_ind, W2, b2)
%     h = 1e-5;
%     X_affined = reshape(X_affined, [32,32,32,10]);
%     grad_S = zeros(size(X_affined));
%     for i = 1:numel(check_ind)
%         [sub_1, sub_2] = ind2sub(size(X_affined), check_ind(i));
%         S1 = X_affined;
%         S2 = X_affined;        
%         S1(sub_1, sub_2) = S1(sub_1, sub_2) + h;
%         S2(sub_1, sub_2) = S2(sub_1, sub_2) - h;
%         
%         loss1 = softmax_loss(affine_forward(S1, W2, b2), y_train);
%         loss2 = softmax_loss(affine_forward(S2, W2, b2), y_train);
%         grad_S(sub_1, sub_2) = (loss1 - loss2)/(2*h);
%         fprintf('processing ind %d; grad = %f\n', check_ind(i), grad_S(sub_1, sub_2));
%     end
% end


% check dX
function grad_S = eval_gradient_S(X, y, check_ind, W1, b1, W2, b2, conv_param)
    h = 1e-3;
    
    grad_S = zeros(size(X));
    for i = 1:numel(check_ind)
        [sub_1, sub_2] = ind2sub(size(X), check_ind(i));
        X1 = X;
        X2 = X;        
        X1(sub_1, sub_2) = X1(sub_1, sub_2) + h;
        X2(sub_1, sub_2) = X2(sub_1, sub_2) - h;
        
        X_conv1 = conv_forward(X1, W1, b1, conv_param);
        X_conv2 = conv_forward(X2, W1, b1, conv_param);
        
        loss1 = softmax_loss(affine_forward(X_conv1, W2, b2), y);
        loss2 = softmax_loss(affine_forward(X_conv2, W2, b2), y);
        grad_S(sub_1, sub_2) = (loss1 - loss2)/(2*h);
        fprintf('processing ind %d: grad = %f\n', check_ind(i), grad_S(sub_1, sub_2));
    end
end

% check dX_relu
% function grad_relu = eval_gradient_S(X_relu, y_batch, check_ind, pool_param, W2, b2)
%     h = 1e-8;
%     
%     grad_relu = zeros(size(X_relu));
%     for i = 1:numel(check_ind)
%         [sub_1, sub_2] = ind2sub(size(X_relu), check_ind(i));
%         S1 = X_relu;
%         S2 = X_relu;        
%         S1(sub_1, sub_2) = S1(sub_1, sub_2) + h;
%         S2(sub_1, sub_2) = S2(sub_1, sub_2) - h;
%         
%         S1 = max_pool_forward(S1, pool_param);
%         S2 = max_pool_forward(S2, pool_param);
% 
%         loss1 = softmax_loss(affine_forward(S1, W2, b2), y_batch);
%         loss2 = softmax_loss(affine_forward(S2, W2, b2), y_batch);
%         grad_relu(sub_1, sub_2) = (loss1 - loss2)/(2*h);
%         fprintf('processing ind %d; grad = %f\n', check_ind(i), grad_relu(sub_1, sub_2));
%     end
% end

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