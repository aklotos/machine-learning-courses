function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

function res = sigmoidFunc(el)
    res = 1 / (1 + exp(-el));
end

[m,n] = size(z);
for i = 1:m
   for j = 1:n
      g(i, j) = sigmoidFunc(z(i,j));
   end
end

% =============================================================

end
