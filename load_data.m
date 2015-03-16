function [trdata, trlables, tedata, telabels] = load_data()
    trdata = [];
    trlables = [];
    for i = 1:5
        load(['../cifar-10-batches-mat/data_batch_', int2str(i), '.mat']);
        trdata = [trdata; data];
        trlables = [trlables; labels];
    end
    
    load('../cifar-10-batches-mat/test_batch.mat');
    tedata = data;
    telabels = labels;
end