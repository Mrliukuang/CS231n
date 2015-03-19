function cols = im_2_col(X, filter_h, filter_w, conv_param)
    % This file is kind like matlab build-in 'im2col.m', but with strides.
    [H, W, C, N] = size(X);
    pad = conv_param.pad;
    stride = conv_param.stride;
    
    HH = (H + 2*pad - filter_h) / stride + 1;
    WW = (W + 2*pad - filter_w) / stride + 1;
    
    X_padded = padarray(X, [pad, pad]);

    cols = zeros(C*filter_h*filter_w, N*HH*WW);
 
    % this method is faster!
    for w = 1:WW
        for h = 1:HH
            x = 1 + (w-1) * stride;
            y = 1 + (h-1) * stride;
            
            cube = X_padded(y : y+filter_h-1, x : x+filter_w-1, :, :);
            cube = reshape(cube, [], N);
            ind = sub2ind([HH, WW], h, w);
            cols(:, (ind-1)*N+1 : ind*N) = cube;
        end       
    end  
    
    
    % this method is simpler, but slower
%     for n = 1:N
%         for c = 1:C
%             img = X_padded(:, :, c, n);
%             cols((c-1)*filter_h*filter_w+1 : c*filter_h*filter_w, (n-1)*HH*WW+1 : n*HH*WW) = im2col(img, [filter_h, filter_w], 'sliding');
%         end
%         
%     end
    
    
    
    
    
    
    
    


end