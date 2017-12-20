function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
% 我自己写的
% n=size(y);
% for i=1:n
%     if (y(i) == 1)
%     plot(X(i,1),X(i,2),'k+')
%     else
%     plot(X(i,1),X(i,2),'ko')
%     end
% end

% Find Indices of Positive and Negative Examples
pos = find(y == 1); %y是一个列向量，find找出y里面所有满足的条件值的索引，从而控制X的读取
neg = find(y == 0);%整体上与前面的if条件查找意义是一样的
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);
% plot(X(pos, 1), X(pos, 2), 'k+');
% plot(X(neg, 1), X(neg, 2), 'ko');

% =========================================================================



hold off;

end
