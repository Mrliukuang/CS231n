function [X_conv, cols] = conv_forward(X, conv_layer)
    % TODO. try 'convn' in matlab
    W1 = conv_layer.W;
    b1 = conv_layer.b;
    pad = conv_layer.pad;
    stride = conv_layer.stride;
    
    [H, W, ~, N] = size(X);
    [filter_h, filter_w, ~, filter_n] = size(W1);
    
    % output size
    HH = (H + 2 * pad - filter_h) / stride + 1;
    WW = (W + 2 * pad - filter_w) / stride + 1;
    
    if ~isa(X, 'double')
        X = double(X);
    end
    
    cols = im_2_col(X, filter_h, filter_w, pad, stride);
    W_r = reshape(W1, [], filter_n)';
    X_conv = bsxfun(@plus, W_r * cols, b1);
    
    % the reshaped X_conv is of size [HH, WW, filter_n, N]
    t = reshape(X_conv', N, []);
    X_conv = reshape(t', [HH, WW, filter_n, N]);
 
end







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stupid method! Use 'conv_forward_reshape()' above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function conved_X = conv_forward_loop(X, W, b, conv_param)
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
