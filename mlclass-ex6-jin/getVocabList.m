function vocabList = getVocabList()
%GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
%cell array of the words
%   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
%   and returns a cell array of the words in vocabList.


%% Read the fixed vocabulary list
fid = fopen('vocab.txt');

% Store all dictionary words in cell array vocab{}
n = 1899;  % Total number of words in the dictionary
% 一共是这么多单词
% For ease of implementation, we use a struct to map the strings =>
% integers将字符串变换为整数
% In practice, you'll want to use some form of hashmap哈希表
vocabList = cell(n, 1);
% 构建空的元胞数组
for i = 1:n
    % Word Index (can ignore since it will be = i)
    fscanf(fid, '%d', 1);%%d是读取的整数
    % Actual Word
    vocabList{i} = fscanf(fid, '%s', 1);%%s存入对应的单词列表
end
fclose(fid);

end
