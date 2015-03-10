function X = col_2_im(cols, HH, WW, filter_n, N)
    % the reshaped X is of size [HH, WW, filter_n, N]
    t = reshape(cols', N, []);
    X = reshape(t', [HH, WW, filter_n, N]);
end