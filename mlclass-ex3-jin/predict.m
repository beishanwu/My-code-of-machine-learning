function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);%数据集大小
num_labels = size(Theta2, 1);%标签数量

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix  -jin
% X增加了一列
X = [ones(m, 1) X];%这是神经网络规则，在计算下一级时上一级要加上bias

a2 = sigmoid(X * Theta1');   % 第二层激活函数输出 5000by401*401by25---5000by25
a2 = [ones(m, 1) a2];        % 第二层加入b  5000by26
a3 = sigmoid(a2 * Theta2');  % 5000by26*26by10---5000by10
[aa,p] = max(a3,[],2);               % 返回每行最大值的索引位置，也就是预测的数字

%说一下一些注意点：首先是神经网络规则，在计算下一级时上一级要加上bias
%然后在实验三文档中提到了“输出结果应该是一个列向量”，我个人认为这是一个很经典的表述，但是有些东西要厘清
%就这个程序而言，你会发现不是这样的，无论是最后还是中间计算结果是5000by10这种形式。怎么理解呢。
%是这样的。首先就这个程序而言，这种方式的计算结果对每一个输入数据，结果是行向量，依次是该层每个神经单元的计算结果
%这个从思想上与列向量结果是等价的。
%另外一点是，结果是一个矩阵而不是向量。这个怎么理解呢。
%是这样的，这个按上节的说法就是vector了。这种方式能简化计算。
%整体理解就是，每一个输出结果是一个行向量，5000个输入example就有5000行的结果

%总体感受上来说，神经网络的前向传播算法在就简单矩阵的计算


% =========================================================================


end
