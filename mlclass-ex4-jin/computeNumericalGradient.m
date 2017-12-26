function numgrad = computeNumericalGradient(J, theta)
%COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
%and gives us a numerical estimate of the gradient.
% 使用有限差异，得到梯度的数值估计
%   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
%   gradient of the function J around theta. Calling y = J(theta) should
%   return the function value at theta.

% Notes: The following code implements numerical gradient checking, and 
%        returns the numerical gradient.It sets numgrad(i) to (a numerical 
%        approximation of) the partial derivative of J with respect to the 
%        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
%        be the (approximately) the partial derivative of J with respect 
%        to theta(i).)
% J关于theta的偏导数
%                
% 这个部分是梯度检验中计算数值梯度的
% 这个程序提供了一个很好的设计思路和模块
% 梯度的本质也是偏导数
numgrad = zeros(size(theta));
perturb = zeros(size(theta));
e = 1e-4;
% numel是返回元素个数
% 从外部调用形式可以看到，J是传入的函数句柄，通过对theta的某一个进行一点+-偏差，计算这一个的偏导
for p = 1:numel(theta)
    % Set perturbation vector微变向量
    perturb(p) = e;
    loss1 = J(theta - perturb);
    loss2 = J(theta + perturb);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
end

end
