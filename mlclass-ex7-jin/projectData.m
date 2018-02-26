function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%
% 是按列选取的
     U_reduce = U(:, 1:K);
     Z =X * U_reduce;
%      可以看到这里的Z的公式与笔记上的内容是不一致的
% 原因在于X的构造形式与笔记上是不同的
% 这里的X是m*n,U_reduce是n*k,从而得到的Z是m*k,可以正好是将特征缩减到k个的结果
     
% =============================================================

end
