function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%返回结果中，第i个分类器在第i行
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables获取X行列
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
%这里之所以是n+1列，是对x0的
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
%X也需要加一列x0
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

options = optimset('GradObj', 'on', 'MaxIter', 50);

initial_theta = zeros(n + 1, 1);
%这就是标签值K的遍历，这里是必须使用循环的，所谓的向量化不是指这里
for c = 1:num_labels

all_theta(c,:) = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
            initial_theta, options);
        %这个地方使用的y == c会生成一个逻辑MAT，在这里是非常巧妙的用法
        %本质思路是，对每一个分类器，对应的正确标签值标记为1，其余所有的情况都应标记为0，对每一个分类器都如何训练，从而得到K种不同分类器
end
% =========================================================================
end
