function [h, display_array] = displayData(X, example_width)
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

% Set example_width automatically if not passed in
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end

% Gray Image
colormap(gray);

% Compute rows, cols
[m n] = size(X);
example_height = (n / example_width);%这是一幅图像的行数

% Compute number of items to display
%显示的时候图像是进行了排列的，这里要特别注意这里几个行列的意义
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% Between images padding
%图像直接的填充，如图的就是图像间的黑线，这是一个挺好的做法
pad = 1;

% Setup blank display
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
%遍历写入图像
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, %这是在内存循环中
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch补丁
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;%是进行了归一化的，在每一个图像内部归一化
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, %这是在外层循环中，因为break一次不能跳出两层循环
		break; 
	end
end

% Display Image
h = imagesc(display_array, [-1 1]);%在-1到1的尺度内显示

% Do not show axis
axis image off

drawnow;%更新图形窗口和执行挂起的回调 

end
