function hist_im = color_histgram_hsv(im)
%% Compute color histogram for an image using hue.    

    nbin = 20;  % number of histogram bins. (default: 10)
    hsv = rgb2hsv(im);
    hist_im = imhist(hsv(:,:,1), nbin);
    hist_im = hist_im / sum(hist_im);      % or use featureNormalize.m
    
end