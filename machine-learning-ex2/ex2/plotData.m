function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

data = [X y];
k_o = find(y == 0);
k_plus = find(y == 1);
plot(X(k_o,1), X(k_o,2), 'ko');
plot(X(k_plus,1), X(k_plus,2), 'k+');

% =========================================================================



hold off;

end
