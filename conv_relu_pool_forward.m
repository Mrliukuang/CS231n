function a = conv_relu_pool_forward(X, W, b, conv_param, pool_param)
    % conv
    a = conv_forward(X, W, b, conv_param);
    % ReLU
    s = max(0, a);
    % max-pool
    out = max_pool_forward(s);



   
    
    
    
end

function out = max_pool_forward(s)
    




end


function conved_X = conv_forward(X, W, b, conv_param)
     % X: all input of size H*W*C*N
    % W & b: parameters for filters
    [H_X, W_X, ~, N] = size(X);
    
    % output size = X size
    H_out = H_X;
    W_out = W_X;
    
    % filter size
    [filter_h, filter_w, ~, filter_n] = size(W);
    
    % Xpad: (H+2pad)*(H+2pad)*C*N
    pad = conv_param.pad;
    pad_X = padarray(X, [pad, pad]);
    
    % conv through all images.
    S = conv_param.stride;
    conved_X = zeros(H_out, W_out, filter_n, N);
    
    for n = 1:N     % loop through all images
        fprintf('convoluting image %d...\n', n);
               
        for i=1:W_out   % for each patch of Xpad, perform convolution
            for j=1:H_out
                % map output index to input index
                x = i+S-1;
                y = j+S-1;
                % get the patch(cube) of the input image
                cube = pad_X(y : y+filter_h-1, x : x+filter_w-1, :, n);
                % perform convolution on this patch
                conved_X(i,j,:,n) = conv_patch(cube, W, b);
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

