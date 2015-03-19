function m = col_2_im(X, cols, filter_h, filter_w, conv_param)
    [H, W, C, N] = size(X);
    pad = conv_param.pad;
    stride = conv_param.stride;
    
    HH = (H + 2*pad - filter_h) / stride + 1;
    WW = (W + 2*pad - filter_w) / stride + 1;
    
    m = zeros(size(X));
    i = 1;
    j = 1;
    for n = 1 : HH*WW
        col = cols(:, (n-1)*N+1 : n*N);
        sq = reshape(col, [filter_h, filter_w, C, N]);
        m(i:i+filter_h-1, j:j+filter_w-1, :, :) = m(i:i+filter_h-1, j:j+filter_w-1, :, :) + sq;
        
        i = i + stride;
        if i >= H
            i=1;
            j=j+stride;
        end
    end
    

end