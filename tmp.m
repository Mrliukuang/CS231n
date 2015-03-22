[H, W, C, N] = size(a);
layer_size = H*W;

filter_h = 3;
filter_w = 3;

stride = 3;

HH = (H-filter_h)/stride+1;
WW = (W-filter_w)/stride+1;

% the index accross C-index
cind = 1 : C*layer_size : 1+(N-1)*C*layer_size;
hind = (0:filter_h-1)';
% t of size [filter_h, N]
t = bsxfun(@plus, cind, hind);

% t2 of size [filter_w*filter_h, N]
t2 = zeros(filter_w*filter_h, N);
for i = 0:filter_w-1
    t2(i*filter_h+(1:filter_h), :) = t + i*H;
end

% t3 of size [C*filter_w*filter_h, N]
t3 = zeros(C*filter_w*filter_h, N);
for i = 0:C-1
    t3(i*filter_h*filter_w+(1:filter_h*filter_w), :) = t2 + i*layer_size;
end

% t4 of size [C*filter_w*filter_h, N * HH]
t4 = zeros(C*filter_w*filter_h, N*HH);
for i = 0:HH-1
    t4(:, i*N+(1:N)) = t3 + i*stride;
end

% t5 of size [C*filter_w*filter_h, N*HH*WW]
t5 = zeros(C*filter_w*filter_h, N*HH*WW);
for i = 0:WW-1
    t5(:, i*N*HH+(1:N*HH)) = t4 + i*stride*H;
end




















rind = 1:stride:1+(HH-1)*stride;
cind = (0:filter_h-1)';

% t of size [filter_h, HH]
% t = bsxfun(@plus, rind, cind);
t = rind(ones(filter_h, 1), :) + cind(:, ones(HH, 1));



% t of size [C*filter_h, HH]
t = zeros(C*filter_h, HH);
cs = 1:filter_h;
for i=0:C-1
    t(i*filter_h+cs, :) = t_layer + layer_size*i;
end

% tt of size [C*filter_h*filter_w, HH]
tt = zeros(C*filter_h*filter_w, HH);



rows = 1:filter_h;
for i = 0:C*filter_w-1
    tt(i*filter_h+rows, :, :) = t + H*i;
end

ttt = zeros(filter_h*filter_w, HH*WW);
cols = 1:WW;
for i = 0:WW-1
    ttt(:, i*HH+cols, :) = tt + H*stride*i;
end

a(ttt)










