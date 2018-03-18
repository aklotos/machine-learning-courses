function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

% ====== f1-score ======

function res = tp(predval, yval)
    res = sum((predval == 1) & (yval == 1));
end

function res = fp(predval, yval)
    res = sum((predval == 1) & (yval == 0));
end

function res = fn(predval, yval)
    res = sum((predval == 0) & (yval == 1));
end

function res = prec(predval, yval)
    res = tp(predval, yval) / (tp(predval, yval) + fp(predval, yval)); 
end

function res = rec(predval, yval)
    res = tp(predval, yval) / (tp(predval, yval) + fn(predval, yval));
end

function res = f1score(predval, yval)
    res = 2 * prec(predval, yval) * rec(predval, yval) ...
        / (prec(predval, yval) + rec(predval, yval));
end

% ======================

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    predval = pval < epsilon;
    F1 = f1score(predval, yval);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
