function [loss, scores, dscores, model] = pass_forward(X_batch, y_batch, model)
    % Pass forward every layer of the model
    X_flow = X_batch;
    for i = 1:model.layer_num
        switch model.layer{i}.name
            case 'conv'
                model.layer{i}.input_size = size(X_flow);
                
                [X_flow, X_cols] = conv_forward(X_flow, model.layer{i});
                model.layer{i}.X_conv = X_flow;  % Conv -> ReLU
                model.layer{i}.X_cols = X_cols;
                X_flow = max(0, X_flow);
                model.layer{i}.X_relu = X_flow;
            case 'pool'
                [X_flow, max_ind] = MaxPooling(X_flow, model.layer{i}.pool_size);
                model.layer{i}.X_pool = X_flow;
                model.layer{i}.max_ind = max_ind;
            case 'fc'
                model.layer{i}.input_size = size(X_flow);
                
                [scores, X_flow] = affine_forward(X_flow, model.layer{i});
                model.layer{i}.X_affine = X_flow;
                [loss, dscores] = softmax_loss(scores, y_batch);
        end
    end
end