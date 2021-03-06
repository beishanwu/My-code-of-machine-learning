function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

%数据集行数
m = size(X, 1);
%这是标签数，也是分类器个数
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);%同行数的列矩阵

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

%取最大预测值作为该标签分类
[a,p] = max(sigmoid( X * all_theta'),[],2) ;    % 返回每行最大值的索引位置，也就是预测的数字
%这里的2表示在每一行中找最大值，如果是1则是从每一列中找最大值，返回最大值和索引，都是向量输出（因为输入的是矩阵）
%注意这里的X * all_theta'的原理
%all_theta'是一个K列的矩阵
%在all_theta是一个K行的矩阵，每一行都是一个最优化计算后得到的theta值，这K行按标签顺序排放
%这里就可以解释将0标签设置为10的用意。
%K=10的标签在all_theta的最后一行。使用max（）函数值时，结果索引是从1开始的（MATLAB中没有零索引）
%所以为了程序的简洁，将0设置为10。
%这是有一个完整过程的。从数据集到theta优化计算，到最后的预测。

%预测的思路都是一样的，没有特别区分训练集合测试集。





% =========================================================================


end
