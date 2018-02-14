%% matlab file of image 
folder = '/Users/xvz5220-admin/Dropbox/cell_tracking_data/data_output/01_09/';
img_name = [folder 't000.tif_09.png'];
I = imread(img_name);
bw = im2bw(I, graythresh(I));
imshow(bw);

bw2 = imfill(bw,'holes'); % fill in the holes
bw3 = imopen(bw2, ones(2,2)); % 
bw4 = bwareaopen(bw3, 10); % 

L = bwlabel(bw4, 4); % from black and white to the labels
Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');

