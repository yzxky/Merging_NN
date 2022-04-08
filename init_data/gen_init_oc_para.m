function gen_init_oc_para(filename, alpha)
    load([filename, '.mat'])
    beta = alpha * 4^2 / (2 * (1 - alpha));
    para = [];
    for i = 1:size(init_queue, 1)
        para = [para; solveUnconstrainedOpt(init_queue(i, 3), init_queue(i, 4), 400, beta)];
    end
    save([filename, '_ocpar_025.mat'], 'para')
    