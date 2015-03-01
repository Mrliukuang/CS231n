function rel_error = cmp_gradients(fa, fn)
    % relative error = |fa-fn| / (|fa| + |fn|)
    rel_error = norm(fa - fn)/(norm(fa) + norm(fn) + eps);
end