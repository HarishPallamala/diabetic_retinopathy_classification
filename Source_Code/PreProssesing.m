clc
clear all
close all
input_imgs_dir = 'D:\DRImp\NewDataset\ABNCLR'; % give the path of the folder where your images are present

pre_prosessed_imgs = 'D:\DRImp\NewDataset'; % give the path of the folder where you want to save images

image = dir([input_imgs_dir,filesep,'\*.jpg']) ; % give extension of your images

N = 1409; % total images in the foder

for i = 1:N
current_file = [input_imgs_dir,filesep,image(i).name] ; % each image file
[filepath,file_name,extension] = fileparts(current_file) ; % get name of the image file and extension
output_img_dir = [pre_prosessed_imgs,filesep,[file_name,extension]] ; % write image file on this name here
K = imread(current_file) ; % read image into
K = imresize(K,[512 512]);
I = rgb2gray(K); %512*512 size pxls
imwrite(I,output_img_dir) ; % write the image in the said folder on the name
end



