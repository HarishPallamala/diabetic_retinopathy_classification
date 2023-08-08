clear all;
close all;
clc;
input_imgs_dir = 'D:\DRImp\NewDataset\ABNCLR'; % give the path of the folder where your images are present

segmented_images = 'D:\DRImp\SegmentedImgs'; % give the path of the folder where you want to save images

image = dir([input_imgs_dir,filesep,'\*.jpg']);

N = 1; % total images in the folder

for i = 1:N
current_file = [input_imgs_dir,filesep,image(i).name] ; % each image file
[filepath,file_name,extension] = fileparts(current_file) ; % get name of the image file and extension
output_image_dir = [segmented_images,filesep,[file_name,extension]] ; % write image file on this name here
K = imread(current_file) ; % read image into
K = imresize(K,[336 448]); %as mentioned in paper
segmented_img = vesselSegPC(K);
imwrite(segmented_img,output_image_dir);
end