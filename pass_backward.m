function model = pass_backward(model, dscores)
    % Pass backwards
    dflow = dscores;
    for i = model.layer_num : -1 : 1
       switch model.layer{i}.name
            case 'fc'
                [dflow, dW, db] = affine_backward(dflow, model.layer{i});
                model.layer{i}.dW = dW;
                model.layer{i}.db = db;
                
                dflow = reshape(dflow, model.layer{i}.input_size);
            case 'pool'
                % dflow = reshape(dflow, size(model.layer{i}.X_pool));
                dflow = max_pool_backward(dflow, model.layer{i});
            case 'conv'
               % pass back ReLU layer
                dflow = relu_backward(dflow, model.layer{i});
                [dflow, dW, db] = conv_backward(dflow, model.layer{i});
                model.layer{i}.dW = dW;
                model.layer{i}.db = db;
        end 
    end

end