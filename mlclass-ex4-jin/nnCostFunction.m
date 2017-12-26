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
%   parameters����Ҫ��չ��Ϊ�������� �����ƺ���Ҫ���ػ�ȥ

%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%   ���صĲ�����һ��ƫ����չ������

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% ���Theta����Ϊ�������ʱ�ǽ�Theta1��Theta2������һ���ˣ����������Ǽ�����������ĸ�����
% Ȼ������������ڲ���Ҫ�����ϵķֿ���Theta1��Theta2
%����ľ����⹫ʽ����˵������Theta�ı�׼��ʽ��
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));%����ͬ�ȴ�С����
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% �ڱ���J�еĴ��ۺ���
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%     ͨ��checkNNGradients���м���
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%               Ҫ��y �߼�ֵ��
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%         �����ʾ��˵�����һ��ʹ�õ�ʱ��ʹ��forѭ�����з��򴫲��㷨
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%           ��part2�Ļ����ϲ���

%% ��y���д��� Y(find(y==3))= [0 0 1 0 0 0 0 0 0 0]; ���� Feedforward cost function 1��2   
%�����ǿ����˺�������չ�ԣ���ʵ����Ҹ��˾��û��ǱȽ���Ҫ�ģ���Ϊ�������ƺ�����һ����һ��ʼ�Ϳ���ȷ���ṹ�Ķ���
Y=[];
E = eye(num_labels);    % Ҫ����K���������⣬����дeye(10)����
for i=1:num_labels
    Y0 = find(y==i);    % �ҵ�����y=i�����к�,�滻����
    Y(Y0,:) = repmat(E(i,:),size(Y0,1),1);%Y ��ά����5000*10���ӽ������������ȫ�ܹ����ģ������������ʵ��ԭ���ҷǳ�����⣿����
    %��ô����ԭ����ԭʼyֱ�ӱ�ǽ���ģ������ǹ���ֵ��Ҫ���Ƕ��ڵ�ÿ�������Ԫ�Ľ����������Ҫ���߼�ֵ
end

%% unregularized Feedforward cost function lambda=0
% % ����ǰ���� Add ones to the X data matrix  -jin
% X = [ones(m, 1) X];
% a2 = sigmoid(X * Theta1');   % �ڶ��㼤������
% a2 = [ones(m, 1) a2];        % �ڶ������b
% a3 = sigmoid(a2 * Theta2');  5000by10
% 
% cost = Y .* log(a3) + (1 - Y ) .* log( (1 - a3));  % cost��m*K(5000*10)�Ľ������  sum(cost(:))ȫ�����
% J= -1 / m * sum(cost(:)); 

%�����Ѿ��Ƿǳ�ȫ���ʸ����������
% a3�ĳߴ���5000by10������5000��i�ı�����10��k�ı���

%% regularized Feedforward cost function lambda=1
% ����ǰ���� Add ones to the X data matrix  -jin
% ���������򻯵�
X = [ones(m, 1) X];
a2 = sigmoid(X * Theta1');   % �ڶ��㼤������
a2 = [ones(m, 1) a2];        % �ڶ������b
a3 = sigmoid(a2 * Theta2');  

temp1 = [zeros(size(Theta1,1),1) Theta1(:,2:end)];   % �Ȱ�theta(1)�õ������������򻯣�����Ϊ0
temp2 = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
temp1 = sum(temp1 .^2);     % ����ÿ��������ƽ�����پ����
temp2 = sum(temp2 .^2);

% ����ע�����ݿ�����ķ����򻯲���
%��������Ǵ������򻯵ģ����ǲ�����һ�������򻯣���Ϊlambda�������ⲿ����ģ�ȡ�����ⲿ����
cost = Y .* log(a3) + (1 - Y ) .* log( (1 - a3));  % cost��m*K(5000*10)�Ľ������  sum(cost(:))ȫ�����
J= -1 / m * sum(cost(:)) + lambda/(2*m) * ( sum(temp1(:))+ sum(temp2(:)) );  
%���򻯵�������������������һ����

%% ���� Gradient 
delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

% ��Ϊa_1���Ѿ�ת��Ϊ��������ʽ��Ҳ���������粿��֧�ֱ�׼��ʽ�����к��沿�ֶ���ֱ�Ӱ��չ�ʽ����
for t = 1:m
   % step 1 ǰ�����
   a_1 = X(t,:)'; %ȡ��һ��ѵ�����ݣ���ת��Ϊ������         
%        a_1 = [1 ; a_1];%����ע�͵��ˣ���Ϊ�����X�Ѿ���ǰ�����һ����
   z_2 = Theta1 * a_1;   %25*1
   a_2 = sigmoid(z_2);  
      a_2 = [1 ; a_2];%��Ҫ����һ��
   z_3 = Theta2 * a_2;
   a_3 = sigmoid(z_3);
   % step 2 �����3
   err_3 = zeros(num_labels,1);%���Կ����������һ��������
   for k = 1:num_labels     
      err_3(k) = a_3(k) - (y(t) == k);%�߼�ֵ��
   end
   % step 3 �����2
   err_2 = Theta2' * err_3;                % err_2��26�У���������ط�Ҫ�ر�ע��
   err_2 = err_2(2:end) .* sigmoidGradient(z_2);   % ȥ����һ�����ֵ������Ϊ25. sigmoidGradient(z_2)ֻ��25�У�����
   % step 4 �ۼ��ݶ�
   delta_2 = delta_2 + err_3 * a_2';
   delta_1 = delta_1 + err_2 * a_1';
end

% step 5    ����ƫ����������Ǵ����򻯵ģ�
Theta1_temp = [zeros(size(Theta1,1),1) Theta1(:,2:end)];%����һ������Ϊ0����Ϊ����ʱj����Ϊ0
Theta2_temp = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta1_grad = 1 / m * delta_1 + lambda/m * Theta1_temp;
Theta2_grad = 1 / m * delta_2 + lambda/m * Theta2_temp ;
      
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];%�ϲ����


end
