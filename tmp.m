siz = [4,4,4,4];

ind = sub2ind(siz, [2;4],[3;4],[1;2],[4;2])

get_ind = @(siz, x1, x2, x3, x4) x1 + (x2-1)*siz(1) + (x3-1)*siz(1)*siz(2) + (x4-1)*siz(1)*siz(2)*siz(3);
get_ind(siz, [2;4],[3;4],[1;2],[4;2])

















