close all;
clear;
clc;

wsiname = 'W26-1-1-E.03_0001_PC';

im1 = imread(['D:\GBM_Project\Current_Experiments\PC_Patches\PC_1792_raw_Aug2023\Testing\PC\',wsiname,'.jpg']);
mask = imread(['D:\GBM_Project\Current_Experiments\PC_Patches\PC_1792_raw_Aug2023\Testing\PC_SG\SG_',wsiname,'.jpg']);
predmask = imread(['D:\JournalExperiments\PC\Predictions\IvyGap_SL_Frozen\SG_',wsiname,'.jpg']);

% im1 = imread(['D:\JournalExperiments\PC\TCGA\PC_1792_raw\Testing\PC\',wsiname,'.jpg']);
% mask = imread(['D:\JournalExperiments\PC\TCGA\PC_1792_raw\Testing\PC_SG\SG_',wsiname,'.jpg']);
% predmask = imread(['D:\JournalExperiments\PC\Predictions\TCGA_SL_Frozen\SG_',wsiname,'.jpg']);


% im1 = imread('D:\GBM_Project\Current_Experiments\MV_Patches\MV_896_raw\Testing\MV\W9-1-1-H.2.01_0014_MV.jpg');
% mask = imread('D:\GBM_Project\Current_Experiments\MV_Patches\MV_896_ChA_data_augm\Testing\MV_SG2\SG_W9-1-1-H.2.01_0014_MV.jpg');

mask_boundaries = bwboundaries(mask(:,:,1));
pred_mask_boundaries = bwboundaries(predmask(:,:,1));

figure, 
% subplot(1,3,1);

% imshow(im1);
% title('H&E')
% subplot(1,3,2);
% imshow(mask);
% title('Prediction')
% subplot(1,3,3);
% imshow(im1);
% title('Prediction')
% hold on;

% subplot(1,3,1);
imshow(im1);
title('H&E')
% imshow(mask);
% title('Prediction')
% subplot(1,3,2);
%%

figure, imshow(im1);
title('Ground Truth')

hold on;

numVascMask = size(mask_boundaries,1);

for i=1:numVascMask
    b = mask_boundaries{i};
    plot(b(:,2),b(:,1),'*b');    
end

% subplot(1,3,3);
figure, imshow(im1);
title('Prediction')

hold on;

numVascMask = size(pred_mask_boundaries,1);

for i=1:numVascMask
    b = pred_mask_boundaries{i};
    plot(b(:,2),b(:,1),'*b');    
end

