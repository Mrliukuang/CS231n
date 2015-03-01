%% evaluate gradient H2.
function grad_H = eval_gradient_H(H, y_train, W)
    h = 1e-6;

    ind = 10;
    for i = 1:51
        H1 = H;
        H2 = H;        
        H1(i, ind) = H1(i, ind) + h;
        H2(i, ind) = H2(i, ind) - h;

        loss1 = svm_loss(max(0, W*H1), y_train);
        loss2 = svm_loss(max(0, W*H2), y_train);
        grad_H(i) = (loss1 - loss2)/(2*h);
        fprintf('processing ind %d; grad = %f\n', i, grad_H(i));
    end
end