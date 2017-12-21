function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
%theta和x都已经在外部进行了纬度扩展操作
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%这里主要是推荐使用vectorized（矢量化）
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
%这个公式是正则化的
%说实话，从实验三的论文到这个程序的大篇幅提示，都是在推荐使用正则化
%实际上，从实验一开始，始终都非常注重这一点，始终都在优先使用vectorized，这个程序与实验二中正则化的代价函数是一样的
%本质也应该是一样的one-vs-all和one-vs-one,思想上的区别不在代价函数，而是代价函数的数量
temp=[0;theta(2:end)];    % 先把theta(1)拿掉，不参与正则化
J= -1 * sum( y .* log( sigmoid(X*theta) ) + (1 - y ) .* log( (1 - sigmoid(X*theta)) ) ) / m  + lambda/(2*m) * temp' * temp ;
grad = ( X' * (sigmoid(X*theta) - y ) )/ m + lambda/m * temp ;





% =============================================================

grad = grad(:);

end
