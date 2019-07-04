% close all;
clear all;

run /home/c/matconvnet_test/matconvnet-1.0-beta25/matlab/vl_setupnn;

addpath('utils')


  load('HCNN_original.mat');

use_cascade = 0;

use_gpu = 0;
up_scale =2;
shave = 1;
result_sr=[];
result_bi=[];
folder='Set5';
filepaths = dir(fullfile(folder,'*.bmp'));
for kk=1:length(filepaths)
im_gt = imread(fullfile(folder,filepaths(kk).name));

im_gt = modcrop(im_gt,up_scale);
  im_l = imresize(im_gt,1/up_scale,'bicubic');%%%for set5 and set14

im_gt = double(im_gt);
  im_l  = double(im_l) / 255.0;

[H,W,C] = size(im_l);
if C == 3
    im_l_ycbcr = rgb2ycbcr(im_l);
else
    im_l_ycbcr = im_l;
end
im_l_y = im_l_ycbcr(:,:,1);
if use_gpu
    im_l_y = gpuArray(im_l_y);
end
im_h_ycbcr = imresize(im_l_ycbcr,up_scale,'bicubic');
tic;
im_h_y = VDSR_Matconvnet(im_l_y,im_l_ycbcr, model,up_scale,use_cascade);
toc;
if use_gpu
    im_h_y = gather(im_h_y);
end
im_h_y = im_h_y * 255;
im_h_ycbcr = imresize(im_l_ycbcr,up_scale,'bicubic');

if C == 3
    im_b = ycbcr2rgb(im_h_ycbcr) * 255.0;
    im_h_ycbcr(:,:,1) = im_h_y / 255.0;
    im_h  = ycbcr2rgb(im_h_ycbcr) * 255.0;
else
    im_h = im_h_y;
    im_b = im_h_ycbcr * 255.0;
end

figure;I=im_h./max(max(im_h));imshow(I)
 [sr_psnr sr_ssim] = compute_difference(im_h,im_gt,up_scale);

psnr_sr(kk)=sr_psnr;
ssim_sr(kk)=sr_ssim;
% result_bi(kk)=bi_psnr;
end

fprintf('sr_psnr: %f dB\n',sum(psnr_sr)/5);
fprintf('sr_ssim: %f dB\n',sum(ssim_sr)/5);


