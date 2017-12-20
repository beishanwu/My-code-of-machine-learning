function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));%是一个方阵

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%计算偏导函数
% Note: grad should have the same dimensions as theta
%
% 这里有一点很重要
% X是m*n+1维度的，theta是n+1*1,也就是一个列向量，从而保证了两个进行矩阵相乘的合规
% 并且X*theta是矩阵乘法这点非常重要
% 按照公式θTx也是这样的
J= -1 * sum( y .* log( sigmoid(X*theta) ) + (1 - y ) .* log( (1 - sigmoid(X*theta)) ) ) / m ;
%这里一定要理解得非常清楚，尤其是这里的X转置
grad = ( X' * (sigmoid(X*theta) - y ) )/ m ;



% =============================================================

end
