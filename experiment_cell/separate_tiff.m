function separate_tiff(input_folder, output_folder)
cd(input_folder);
img_list = dir('*.tif');

if ~exist(output_folder, 'dir')
	mkdir(output_folder)
end

for i = 1:length(img_list)
	img_name = img_list(i).name;
	info = imfinfo(img_name);
	num_images = numel(info);
	for k = 1:num_images
    	A = imread(img_name, k, 'Info', info);
    	imwrite(A, [output_folder img_name '_' num2str(k,'%02d') '.tif']);
	end
end

end