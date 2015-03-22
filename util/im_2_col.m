function cols = im_2_col(X, filter_h, filter_w, conv_param)
    % This file is kind like matlab build-in 'im2col.m', but with strides.
%     [H, W, C, N] = size(X);
    pad = conv_param.pad;
    stride = conv_param.stride;
    
%     HH = (H + 2*pad - filter_h) / stride + 1;
%     WW = (W + 2*pad - filter_w) / stride + 1;
    
    X_padded = padarray(X, [pad, pad]);
    
% Fastest method using c, 2X faster than Method #2    
    cols = im_2_col_c(X_padded, [filter_h, filter_w], stride);
    
    
    
% Method #2    
%     cols = zeros(C*filter_h*filter_w, N*HH*WW);
%     
%     % this method is faster!
%     for w = 1:WW
%         for h = 1:HH
%             x = 1 + (w-1) * stride;
%             y = 1 + (h-1) * stride;
%             
%             cube = X_padded(y : y+filter_h-1, x : x+filter_w-1, :, :);
%             cube = reshape(cube, [], N);
%             % ind = sub2ind([HH, WW], h, w);
%             ind = h + (w-1)*HH;
%             cols(:, (ind-1)*N+1 : ind*N) = cube;
%         end       
%     end  
    


% Method #3
%  Fancy index method, but seems not better than the one above...
%     layer_size = H*W;
% %     filter_h = 3;
% %     filter_w = 3;
% %     
% %     stride = 3;
%     
% %     HH = (H-filter_h)/stride+1;
% %     WW = (W-filter_w)/stride+1;
%     
%     % the index accross C-index
%     cind = 1 : C*layer_size : 1+(N-1)*C*layer_size;
%     hind = (0:filter_h-1)';
%     % t of size [filter_h, N]
%     t = bsxfun(@plus, cind, hind);
%     
%     % t2 of size [filter_w*filter_h, N]
%     t2 = zeros(filter_w*filter_h, N);
%     for i = 0:filter_w-1
%         t2(i*filter_h+(1:filter_h), :) = t + i*H;
%     end
%     
%     % t3 of size [C*filter_w*filter_h, N]
%     t3 = zeros(C*filter_w*filter_h, N);
%     for i = 0:C-1
%         t3(i*filter_h*filter_w+(1:filter_h*filter_w), :) = t2 + i*layer_size;
%     end
%     
%     % t4 of size [C*filter_w*filter_h, N * HH]
%     t4 = zeros(C*filter_w*filter_h, N*HH);
%     for i = 0:HH-1
%         t4(:, i*N+(1:N)) = t3 + i*stride;
%     end
%     
%     % t5 of size [C*filter_w*filter_h, N*HH*WW]
%     t5 = zeros(C*filter_w*filter_h, N*HH*WW);
%     for i = 0:WW-1
%         t5(:, i*N*HH+(1:N*HH)) = t4 + i*stride*H;
%     end
%     
%     cols = X_padded(t5);
    
    

% Method #4
% this method is simpler, but slower
%     for n = 1:N
%         for c = 1:C
%             img = X_padded(:, :, c, n);
%             cols((c-1)*filter_h*filter_w+1 : c*filter_h*filter_w, (n-1)*HH*WW+1 : n*HH*WW) = im2col(img, [filter_h, filter_w], 'sliding');
%         end
%         
%     end
    
end