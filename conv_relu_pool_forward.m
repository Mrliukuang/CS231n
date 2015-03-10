function [a, cache] = conv_relu_pool_forward(X, W, b, conv_param, pool_param)
    % conv
    a = conv_forward_reshape(X, W, b, conv_param);
    conv_cache = a;
    
    % ReLU
    a = max(0, a);
    relu_cache = a;
    
    % max-pooling
%     a = max_pool_forward(a, pool_param);
%     pool_cache = a;
%     
    cache{1} = conv_cache;
    cache{2} = relu_cache;
%     cache{3} = pool_cache;
end


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


function X_conv = conv_forward_reshape(X, W1, b1, conv_param)
    [H, W, ~, N] = size(X);
    [filter_h, filter_w, ~, filter_n] = size(W1);
    
    % output size
    HH = (H + 2 * conv_param.pad - filter_h) / conv_param.stride + 1;
    WW = (W + 2 * conv_param.pad - filter_w) / conv_param.stride + 1;
    
    
    cols = im_2_col(X, filter_h, filter_w, conv_param);
    W_r = reshape(W1, [], filter_n)';
    X_conv = bsxfun(@plus, W_r * cols, b1);
    X_conv = col_2_im(X_conv, HH, WW, filter_n, N);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stupid method! Use 'conv_forward_reshape()' above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function conved_X = conv_forward(X, W, b, conv_param)
    % X: all input of size H*W*C*N
    % W & b: parameters for filters
    [H_X, W_X, ~, N] = size(X);
    
    % filter size
    [filter_h, filter_w, ~, filter_n] = size(W);
    
    % output size
    H_out = (H_X + 2 * conv_param.pad - filter_h) / conv_param.stride + 1;
    W_out = (W_X + 2 * conv_param.pad - filter_w) / conv_param.stride + 1;
    
    % Xpad: (H+2pad)*(H+2pad)*C*N
    pad = conv_param.pad;
    pad_X = padarray(X, [pad, pad]);
    
    % conv through all images.
    S = conv_param.stride;
    conved_X = zeros(H_out, W_out, filter_n, N);
    
    for n = 1:N     % loop through all images
        for i=1:W_out   % for each patch of Xpad, perform convolution
            for j=1:H_out
                % map output index to input index
                x = i+S-1;
                y = j+S-1;
                % get the patch(cube) of the input image
                cube = pad_X(y : y+filter_h-1, x : x+filter_w-1, :, n);
                % perform convolution on this patch
                conved_X(j,i,:,n) = conv_patch(cube, W, b);
            end
        end
    end
end

function conved_patch = conv_patch(patch, W, b)
    % patch size: [H_filter, W_filter, C]
    [~, ~, ~, filter_n] = size(W);
    conved_patch = zeros(1, filter_n);
    for i=1:filter_n
        filter = W(:,:,:,i);
        conved_patch(:, i) = patch(:)'*filter(:) + b(i);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


