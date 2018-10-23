function [bestEpsilon bestF1] = selectThreshold(y_cv, p_cv)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(p_cv) - min(p_cv)) / 2000;
for epsilon = min(p_cv):stepsize:max(p_cv)
    
    % compare prob(cv) to current value of epsilon
    predictions = (p_cv < epsilon);
    
    % compute true pos, false pos, false neg, precision, recall, F score
    tp = (y_cv==1) & (predictions==1);
    fp = (y_cv==0) & (predictions==1);
    fn = (y_cv==1) & (predictions==0); 
    prec = sum(tp) / (sum(tp) + sum(fp));
    rec = sum(tp) / (sum(tp) + sum(fn));
    F1 = (2 * prec * rec) / (prec + rec);
   
    % update best values of F and epsilon
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
