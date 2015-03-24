function [reg_loss, model] = add_reg(model, reg)
    % Anonymous func for reg_loss
    % Make sure v is a tall vector when using it
    get_sum = @(v) sum(v .* v);
    
    reg_loss = 0;
    for i = 1:model.layer_num
        if ~strcmp(model.layer{i}.name, 'pool')
            model.layer{i}.dW = model.layer{i}.dW + reg * model.layer{i}.W;
            reg_loss = reg_loss + get_sum(model.layer{i}.W(:));
        end
    end

    reg_loss = 0.5 * reg * reg_loss;
end