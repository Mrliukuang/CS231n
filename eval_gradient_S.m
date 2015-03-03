%% evaluate dS for S after ReLU. S2 = max(0, S).
function grad_S = eval_gradient_S(S, y_train, check_ind)
    h = 1e-5;
    
    grad_S = zeros(size(S));
    for i = 1:numel(check_ind)
        [sub_1, sub_2] = ind2sub(size(S), check_ind(i));
        S1 = S;
        S2 = S;        
        S1(sub_1, sub_2) = S1(sub_1, sub_2) + h;
        S2(sub_1, sub_2) = S2(sub_1, sub_2) - h;

        loss1 = softmax_loss(S1, y_train);
        loss2 = softmax_loss(S2, y_train);
        grad_S(sub_1, sub_2) = (loss1 - loss2)/(2*h);
        fprintf('processing ind %d; grad = %f\n', check_ind(i), grad_S(sub_1, sub_2));
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