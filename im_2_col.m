function cols = im_2_col(X, filter_h, filter_w, conv_param)
    [H, W, C, N] = size(X);
    padding = conv_param.pad;
    stride = conv_param.stride;

    HH = (H + 2*padding - filter_h) / stride + 1;
    WW = (W + 2*padding - filter_w) / stride + 1;

    p = padding;
    X_padded = padarray(X, [p, p]);

    cols = zeros(C * filter_h * filter_w, N * HH * WW);

    for c = 1:C
        for ii = 1:filter_h
            for jj = 1:filter_w
                row = (c-1) * filter_w * filter_h + (ii-1) * filter_h + jj;
                for yy = 1:HH
                    for xx = 1:WW
                        for i = 1:N
                            col = (yy-1) * WW * N + (xx-1) * N + i;
                            cols(row, col) = X_padded(stride*(yy-1)+ii, stride*(xx-1)+jj, c, i);
                        end
                    end
                end
            end
        end
    end
end