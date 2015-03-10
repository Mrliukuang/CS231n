function out = max_pool_forward(in, pool_param)
    % in: conved images now wait for pooling.
    pool_h = pool_param.height;
    pool_w = pool_param.weight;
    
    [in_h, in_w, C, N] = size(in);
    out_h = in_h/pool_h;
    out_w = in_w/pool_w;
    
    out = zeros(out_h, out_w, C, N);
    for i = 1:out_w
        for j = 1:out_h
            % map out index to in index
            x = i + (i-1)*(pool_w-1);
            y = j + (j-1)*(pool_h-1);
            
            cube = in(y:y+pool_h-1, x:x+pool_w-1, :, :);
            out(j, i, :, :) = max(max(cube, [], 2), [], 1);
        end
    end
end