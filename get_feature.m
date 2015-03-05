function X = get_feature(data)
    [~, N] = size(data);
    X = zeros(324, N);
    for i = 1:N
        fprintf('processing image %d\n', i);
        im = reshape(data(:, i), [32, 32, 3]);
        % X(:, i) = color_histgram_hsv(im);
        X(:, i) = extractHOGFeatures(im);
    end
end