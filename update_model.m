function model = update_model(model, lr)
    for i = 1:model.layer_num
        if ~strcmp(model.layer{i}.name, 'pool')
            model.layer{i}.W = model.layer{i}.W - lr * model.layer{i}.dW;
            model.layer{i}.b = model.layer{i}.b - lr * model.layer{i}.db;
        end
    end
end