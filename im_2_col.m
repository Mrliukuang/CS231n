function cols = im_2_col(X, filter_h, filter_w, conv_param)
    % This file is kind like matlab build-in 'im2col.m', but with strides.
    [H, W, C, N] = size(X);
    pad = conv_param.pad;
    stride = conv_param.stride;
    
    get_ind = @(siz, x1, x2, x3, x4) x1 + (x2-1)*siz(1) + (x3-1)*siz(1)*siz(2) + (x4-1)*siz(1)*siz(2)*siz(3);

    HH = (H + 2*pad - filter_h) / stride + 1;
    WW = (W + 2*pad - filter_w) / stride + 1;
    
    X_padded = padarray(X, [pad, pad]);

    cols = zeros(C*filter_h*filter_w, N*HH*WW);
    
    for w = 1:WW
        for h = 1:HH
            for c = 1:C
                for n = 1:N
                    
                    
                end
            end
            
        end
    end
    
    
    
    
%     for w = 1:WW
%         for h = 1:HH
%             x = 1 + (w-1) * stride;
%             y = 1 + (h-1) * stride;
%             
%             cube = X_padded(y : y+filter_h-1, x : x+filter_w-1, :, :);
%             cube = reshape(cube, [], N);
%             ind = sub2ind([HH, WW], h, w);
%             cols(:, (ind-1)*N+1 : ind*N) = cube;
%         end       
%     end  

end