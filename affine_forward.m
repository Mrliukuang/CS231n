function [scores, cache] = affine_forward(in, layer)
    W = layer.W;
    b = layer.b;
    % in of shape [HH, WW, filter_n, N]
    [~, ~, ~, N] = size(in);
    
    % reshape each input to one column
    in = reshape(in, [], N);

    cache = in;
    scores = bsxfun(@plus, W' * in, b);
end