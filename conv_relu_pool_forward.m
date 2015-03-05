function [out, cache] = conv_relu_pool_forward(X, W1, b, conv_param, pool_param)
    % X: H*W*C*N
    [H, W, C, N] = size(X);
    
    % Xpad: (H+2pad)*(H+2pad)*C*N
    pad = conv_param.pad;
    Xpad = padarray(X, [pad, pad]);
    
    % conv through all images.
    N=2;
    C=1;
    for n = 1:N
        for c = 1:C
            one_out = conv_forward(Xpad(:,:,C,N), W1);
            
            
            
            
        end
    end
end

function conved_img = conv_forward(image, W)
    [image_size, ~] = size(image);
    [W_size, ~] = size(W);
    conved_size = (image_size-W_size)+1;
    
    conved_img = zeros(conved_size, conved_size);
    for i = 1:conved_size
        for j = 1:conved_size
            patch = image(i:i+W_size-1, j:j+W_size-1);
        end
    end
end

