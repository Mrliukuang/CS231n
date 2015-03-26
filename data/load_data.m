N = 35887;
tr_num = 28709;
val_num = 3589;
te_num = 3589;
usage = zeros(N, 1, 'uint8');
y = uint8(emotion(2:end));

X = zeros(48,48,1,N, 'uint8');
parfor i = 2:N+1
    if mod(i, 10) == 0
        fprintf('#%d\n', i);
    end
    str = pixels{i};
    C = textscan(str,'%s ');
    C = C{1};
    
    im = zeros(2304, 1, 'uint8');
    for j = 1:2304
        im(j) = uint8(str2num(C{j}));
    end
    
    im = reshape(im, [48,48]);
    X(:,:,:,i-1) = im';
end

