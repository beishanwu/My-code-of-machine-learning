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
%   parameters是需要别展开为向量化的 并且似乎需要返回回去

%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%   返回的参数是一个偏导数展开向量

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% 这个Theta在作为输入参数时是将Theta1和Theta2整合在一起了，这样可能是减少输入参数的个数吧
% 然后在这个程序内部需要将整合的分开成Theta1和Theta2
%这里的矩阵拆解公式可以说得上是Theta的标准公式了
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));%创建同等大小矩阵
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% 在变量J中的代价函数
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%     通过checkNNGradients进行检验
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%               要将y 逻辑值化
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%         这个提示是说建议第一次使用的时候使用for循环进行反向传播算法
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%           在part2的基础上操作

%% 对y进行处理 Y(find(y==3))= [0 0 1 0 0 0 0 0 0 0]; 用于 Feedforward cost function 1和2   
%这里是考虑了后续的扩展性，其实这点我个人觉得还是比较重要的，因为神经网络似乎不是一个从一开始就可以确定结构的东西
Y=[];
E = eye(num_labels);    % 要满足K可以是任意，则不能写eye(10)！！
for i=1:num_labels
    Y0 = find(y==i);    % 找到等于y=i的序列号,替换向量
    Y(Y0,:) = repmat(E(i,:),size(Y0,1),1);%Y 的维度是5000*10，从结果来看我是完全能够理解的，但是这句代码的实现原理我非常不理解？？？
    %这么做的原因是原始y直接标记结果的，而我们过渡值需要的是对于到每个输出单元的结果，并且需要是逻辑值
end

%% unregularized Feedforward cost function lambda=0
% % 计算前向传输 Add ones to the X data matrix  -jin
% X = [ones(m, 1) X];
% a2 = sigmoid(X * Theta1');   % 第二层激活函数输出
% a2 = [ones(m, 1) a2];        % 第二层加入b
% a3 = sigmoid(a2 * Theta2');  5000by10
% 
% cost = Y .* log(a3) + (1 - Y ) .* log( (1 - a3));  % cost是m*K(5000*10)的结果矩阵  sum(cost(:))全部求和
% J= -1 / m * sum(cost(:)); 

%这里已经是非常全面的矢量化操作了
% a3的尺寸是5000by10，其中5000是i的遍历，10是k的遍历

%% regularized Feedforward cost function lambda=1
% 计算前向传输 Add ones to the X data matrix  -jin
% 这是有正则化的
X = [ones(m, 1) X];
a2 = sigmoid(X * Theta1');   % 第二层激活函数输出
a2 = [ones(m, 1) a2];        % 第二层加入b
a3 = sigmoid(a2 * Theta2');  

temp1 = [zeros(size(Theta1,1),1) Theta1(:,2:end)];   % 先把theta(1)拿掉，不参与正则化，设置为0
temp2 = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
temp1 = sum(temp1 .^2);     % 计算每个参数的平方，再就求和
temp2 = sum(temp2 .^2);

% 部分注释内容看上面的非正则化部分
%这个部分是带有正则化的，但是并不是一定会正则化，因为lambda参数是外部传入的，取决于外部设置
cost = Y .* log(a3) + (1 - Y ) .* log( (1 - a3));  % cost是m*K(5000*10)的结果矩阵  sum(cost(:))全部求和
J= -1 / m * sum(cost(:)) + lambda/(2*m) * ( sum(temp1(:))+ sum(temp2(:)) );  
%正则化的区别是在最后加入了这一部分

%% 计算 Gradient 
delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

% 因为a_1等已经转置为列向量形式，也就是神经网络部分支持标准格式，所有后面部分都是直接按照公式来的
for t = 1:m
   % step 1 前向计算
   a_1 = X(t,:)'; %取出一个训练数据，并转置为列向量         
%        a_1 = [1 ; a_1];%这里注释掉了，因为在输出X已经在前面加了一列了
   z_2 = Theta1 * a_1;   %25*1
   a_2 = sigmoid(z_2);  
      a_2 = [1 ; a_2];%需要增加一行
   z_3 = Theta2 * a_2;
   a_3 = sigmoid(z_3);
   % step 2 计算δ3
   err_3 = zeros(num_labels,1);%可以看到定义的是一个列向量
   for k = 1:num_labels     
      err_3(k) = a_3(k) - (y(t) == k);%逻辑值化
   end
   % step 3 计算δ2
   err_2 = Theta2' * err_3;                % err_2有26行！！！这个地方要特别注意
   err_2 = err_2(2:end) .* sigmoidGradient(z_2);   % 去掉第一个误差值，减少为25. sigmoidGradient(z_2)只有25行！！！
   % step 4 累计梯度
   delta_2 = delta_2 + err_3 * a_2';
   delta_1 = delta_1 + err_2 * a_1';
end

% step 5    计算偏导数（这个是带正则化的）
Theta1_temp = [zeros(size(Theta1,1),1) Theta1(:,2:end)];%将第一列设置为0，因为正则化时j不能为0
Theta2_temp = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta1_grad = 1 / m * delta_1 + lambda/m * Theta1_temp;
Theta2_grad = 1 / m * delta_2 + lambda/m * Theta2_temp ;
      
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];%合并输出


end
