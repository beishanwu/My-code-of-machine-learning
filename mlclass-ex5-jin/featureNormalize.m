function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% mean把矩阵中每一列做一个向量，求平均值，返回行向量
mu = mean(X);%平均值，8个
X_norm = bsxfun(@minus, X, mu);%减平均值，减去平均值后还是很大的数据，所以还要进行处理

sigma = std(X_norm);%求标准差，行向量，8个值
X_norm = bsxfun(@rdivide, X_norm, sigma);%左除，这个时候值的范围就比较合适了


% ============================================================

end
