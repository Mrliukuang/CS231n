% function grad = eval_numerical_gradient(X_train, y_train, W)
% %               f(w+h) - f(w)
% %  grad = lim  ---------------
% %         h->0        h
% % 0. this is for check dW.
% % 1. y_train is a tall vector starts from 0.
% % 2. It's only for gradient check. 'DO NOT' compute gradent using this fucntion.
% 
%     h = 1e-5;
%     
%     grad = zeros(size(W));
%     for i = 1:100
%         [sub_1, sub_2] = ind2sub(size(W), i);
%         W1 = W;
%         W2 = W;
%         W1(sub_1, sub_2) = W1(sub_1, sub_2) + h;
%         W2(sub_1, sub_2) = W2(sub_1, sub_2) - h;
% 
%         loss1 = svm_loss(X_train, y_train, W1);
%         loss2 = svm_loss(X_train, y_train, W2);
%         grad(sub_1, sub_2) = (loss1 - loss2)/(2*h);
%         fprintf('processing ind %d; grad = %f\n', i, grad(sub_1, sub_2));
%     end
% end


% check dW1.
% function grad_W = eval_gradient_W(H, y_train, W1, W2)
%     [N, ~] = size(y_train);
%     h = 1e-5;
%     
%     ind = 20;
%     for i = 1:50
%         W_1 = W1;
%         W_2 = W1;
%         W_1(i, ind) = W_1(i, ind) + h;
%         W_2(i, ind) = W_2(i, ind) - h;
% 
%         S1 = max(0, W_1*H);
%         S2 = max(0, W_2*H);
%         S1 = [ones(1, N); S1];
%         S2 = [ones(1, N); S2];
%         
%         loss1 = svm_loss(max(0, W2*S1), y_train);
%         loss2 = svm_loss(max(0, W2*S2), y_train);
%         grad_W(i) = (loss1 - loss2)/(2*h);
%         fprintf('processing ind %d; grad = %f\n', i, grad_W(i));
%     end
% end


% check dW2.
% function grad_W = eval_gradient_W(H, y_train, check_ind, W2)
%     h = 1e-7;
%     
%     grad_W = zeros(size(W2));
%     for i = 1:numel(check_ind)
%         [sub_1, sub_2] = ind2sub(size(W2), check_ind(i));
%         W_1 = W2;
%         W_2 = W2;
%         W_1(sub_1, sub_2) = W_1(sub_1, sub_2) + h;
%         W_2(sub_1, sub_2) = W_2(sub_1, sub_2) - h;
% 
%         S1 = max(0, W_1*H);
%         S2 = max(0, W_2*H);
%         loss1 = svm_loss(S1, y_train);
%         loss2 = svm_loss(S2, y_train);
%         grad_W(sub_1, sub_2) = (loss1 - loss2)/(2*h);
%         fprintf('processing ind %d; grad = %f\n', check_ind(i), grad_W(sub_1, sub_2));
%     end
% end

% check dW2 for ConvNet
% function grad_W = eval_gradient_W(X_affine, y_train, check_ind, W2, b2)
%     h = 1e-7;
%     X_affine = reshape(X_affine, [32,32,32,10]);
%     grad_W = zeros(size(W2));
%     for i = 1:numel(check_ind)
%         [sub_1, sub_2] = ind2sub(size(W2), check_ind(i));
%         W_1 = W2;
%         W_2 = W2;
%         W_1(sub_1, sub_2) = W_1(sub_1, sub_2) + h;
%         W_2(sub_1, sub_2) = W_2(sub_1, sub_2) - h;
%         
%         loss1 = softmax_loss(affine_forward(X_affine, W_1, b2), y_train);
%         loss2 = softmax_loss(affine_forward(X_affine, W_2, b2), y_train);
%         grad_W(sub_1, sub_2) = (loss1 - loss2)/(2*h);
%         fprintf('processing ind %d; grad = %f\n', check_ind(i), grad_W(sub_1, sub_2));
%     end
% end

% check dW1
function grad = eval_gradient_W(X, y, check_ind, W1, b1, W2, b2, conv_param)
    h = 1e-7;
    
    grad = zeros(size(W1));
    for i = 1:numel(check_ind)
        [sub_1, sub_2] = ind2sub(size(W1), check_ind(i));
        W_1 = W1;
        W_2 = W1;        
        W_1(sub_1, sub_2) = W_1(sub_1, sub_2) + h;
        W_2(sub_1, sub_2) = W_2(sub_1, sub_2) - h;
        
        X_conv1 = conv_forward(X, W_1, b1, conv_param);
        X_conv2 = conv_forward(X, W_2, b1, conv_param);
        
        loss1 = softmax_loss(affine_forward(X_conv1, W2, b2), y);
        loss2 = softmax_loss(affine_forward(X_conv2, W2, b2), y);
        grad(sub_1, sub_2) = (loss1 - loss2)/(2*h);
        fprintf('processing ind %d: grad = %f\n', check_ind(i), grad(sub_1, sub_2));
    end
end












