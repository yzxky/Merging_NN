function [parameter] = solveUnconstrainedOpt(t0, v0, L, beta)
    syms a b c d t1
    e1 = @(t0, v0, Li, beta)[1/2 * a * t0^2 + b * t0 + c - v0, ...
                       1/6 * a * t0^3 + 1/2 * b * t0^2 + c * t0 + d, ...
                       1/6 * a * t1^3 + 1/2 * b * t1^2 + c * t1 + d - Li, ...
                       a * t1 + b, ...
                       beta + 1/2 * a^2 * t1^2 + a * b * t1 + a * c];
    [ai, bi, ci, di, t1i] = solve(e1(t0, v0, L, beta), [a, b, c, d, t1]);
    ai = double(ai);
    bi = double(bi);
    ci = double(ci);
    di = double(di);
    t1i = double(t1i);
    idx = find(imag(t1i) == 0 & real(t1i) > t0 & abs(real(ai)) < 10^6);
    parameter = [ai(idx), bi(idx), ci(idx), di(idx), t1i(idx)];
    if length(idx) ~= 1
        disp('error in unconstrained opt')
    end

