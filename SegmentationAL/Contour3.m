close all;
clear;
clc;

% im1 = imread('D:\MV\TCGA-GBM_Patches_MV\TCGA-02-0006_004.jpg');
% mask = imread('D:\MV\mask_mv\SG_TCGA-02-0006_004.jpg');

% im1 = imread('D:\GBM_Project\Current_Experiments\PC_Patches\PC_1792_raw\Training\PC\W8-1-1-H.1.01_0001_PC.jpg');
% mask = imread('D:\GBM_Project\Current_Experiments\PC_Patches\PC_1792_raw\Training\PC_SG\SG_W8-1-1-H.1.01_0001_PC.jpg');
% 
% patchname = 'TCGA-02-0258_005';
% im1 = imread('D:\HistoGBM\Discarted TCGA\WSI\TCGA-19-1788-01Z-00-DX1.svs','Index',2);
% mask = imread('D:\HistoGBM\Discarted TCGA\mask\TCGA-19-1788-01Z-00-DX1.png');

im1 = imread('D:\HistoGBM\IvyGap_Curated\MicroV_WSI_backup\WSI\Training\W2-1-1-N.1.01.jpg');
mask = imread('D:\HistoGBM\IvyGap_Curated\MicroV_WSI_backup\SG\Training\SG_W2-1-1-N.1.01.jpg');

[x,y,z]=size(im1);

mask2 = imresize(mask,[x y]);
mask = mask2;
%%

% im1 = imread(['D:\MV\TCGA-GBM_Patches_MV\',patchname,'.jpg']);
% mask = imread(['D:\MV\mask_mv\','SG_',patchname,'.jpg']);

% im1 = imread('D:\GBM_Project\Current_Experiments\MV_Patches\MV_896_raw\Testing\MV\W9-1-1-H.2.01_0014_MV.jpg');
% mask = imread('D:\GBM_Project\Current_Experiments\MV_Patches\MV_896_ChA_data_augm\Testing\MV_SG2\SG_W9-1-1-H.2.01_0014_MV.jpg');

mask_boundaries = bwboundaries(mask(:,:,1));

% figure, 
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

% fig = subplot(1,2,1);
% imshow(im1);
% title('H&E')
% % imshow(mask);
% % title('Prediction')
% subplot(1,2,2);
% imshow(im1);
% title('GT')
% hold on;
% 
% numVascMask = size(mask_boundaries,1);
% 
% for i=1:numVascMask
%     b = mask_boundaries{i};
%     plot(b(:,2),b(:,1),'.b');    
% end
% 
% saveas(fig,'MySimulinkDiagram.bmp');


%%

% fig = subplot(1,2,1);
imshow(im1);
% title('H&E')
% imshow(mask);
% title('Prediction')
% subplot(1,2,2);
% figure();
% imshow(im1);
% title('GT')
hold on;

numVascMask = size(mask_boundaries,1);

for i=1:numVascMask
    b = mask_boundaries{i};
    plot(b(:,2),b(:,1),'.r');    
end


