function grad_b = eval_gradient_b(cache, y_train, check_ind, W2, b2)
    h = 1e-7;
    
    grad_b = zeros(size(b2));
    for i = 1:numel(check_ind)
        [sub_1, sub_2] = ind2sub(size(b2), check_ind(i));
        b_1 = b2;
        b_2 = b2;
        b_1(sub_1, sub_2) = b_1(sub_1, sub_2) + h;
        b_2(sub_1, sub_2) = b_2(sub_1, sub_2) - h;
        
        S1 = bsxfun(@plus, W2' * cache, b_1);
        S2 = bsxfun(@plus, W2' * cache, b_2);
        
        loss1 = softmax_loss(S1, y_train);
        loss2 = softmax_loss(S2, y_train);
        grad_b(sub_1, sub_2) = (loss1 - loss2)/(2*h);
        fprintf('processing ind %d; grad = %f\n', check_ind(i), grad_b(sub_1, sub_2));
    end
end