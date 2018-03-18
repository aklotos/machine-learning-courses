function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

function [Y] = transformY(y, num_labels)
    [y_size, ~] = size(y);
    Y = zeros(y_size, num_labels);
    for i = 1:y_size
        res = y(i);
        if (res == 0)
            Y(i, num_labels) = 1;
        else
            Y(i, res) = 1;
        end
    end
end

function [res] = addBias(matrix)
    [row_num, ~] = size(matrix);
    res = [ones(row_num, 1) matrix];
end

function [a_next, z_next] = activate(a, Theta)
    a = addBias(a);
    z_next = a * Theta';
    a_next = sigmoid(z_next);
end

function res = regSumOfSquares(matrix)
    matrix(:, 1) = 0;
    res = sum(sum(matrix .^ 2));
end

%%% Cost function %%%
Y = transformY(y, num_labels);
[H_X_2] = activate(X, Theta1);
[H_X_3] = activate(H_X_2, Theta2);
H_X = H_X_3;

J = 1/m * sum(sum( ... 
    -Y .* log(H_X) - (1 - Y) .* log(1 - H_X) ... 
));
reg_J = lambda / (2*m) * (regSumOfSquares(Theta1) + regSumOfSquares(Theta2));
J = J + reg_J;

%%% Backpropagation %%%

function [RegRandTheta] = regGradTheta(Theta)
    RegRandTheta = Theta;
    RegRandTheta(:, 1) = 0;
end

for i = 1:m
    a_1 = X(i, :);
    [a_2, z_2] = activate(a_1, Theta1);
    [a_3, ~] = activate(a_2, Theta2);
    d_3 = (a_3 - Y(i, :))';
    d_2 = (Theta2' * d_3) .* [0; sigmoidGradient(z_2')];
    d_2  = d_2(2:end);
    
    Theta1_grad = Theta1_grad + d_2 * addBias(a_1);
    Theta2_grad = Theta2_grad + d_3 * addBias(a_2);
end
RegGradTheta1 = lambda / m * regGradTheta(Theta1);
RegGradTheta2 = lambda / m * regGradTheta(Theta2);
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad + RegGradTheta1;
Theta2_grad = Theta2_grad + RegGradTheta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
