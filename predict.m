% function ratio = predict(X_val, y_val, W1, b1, W2, b2, conv_param, pool_param)
%     a = conv_relu_pool_forward(X_val, W1, b1, conv_param, pool_param);
%     scores = affine_forward(a, W2, b2);
%     
%     [~, pred] = max(scores);
%     ratio = mean(pred-1 == y_val');
% end

function ratio = predict(scores, y)
    [~, pred] = max(scores);
    
    ratio = mean(pred-1 == y');
end

% function ratio = predict(X, y_train, W)
%     [~, N] = size(X);
%     
%     W1 = W{1};
%     W2 = W{2};
%     H = max(0, W1*X);
%     S = W2*[ones(1,N); H];
%     [~, pred] = max(S);
%     
%     ratio = mean(pred-1 == y_train');
% end