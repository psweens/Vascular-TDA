clc
clear
close all

main_directory = '/media/sweene01/SSD/VA_Paper_Datasets/Lnet Topological Analysis (Isotropic)';
tumour_list = dir(main_directory);
tumour_list = tumour_list([tumour_list(:).isdir]);
tumour_list = tumour_list(~ismember({tumour_list(:).name},{'.','..'}));

for tumour_idx = 1:size(tumour_list,1)
    
    rootfolder = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name);

    tiff1 = dir(strcat(rootfolder, '/', 'Segmented/Groundtruth*'));
    tiff1 = tiff1(~ismember({tiff1(:).name},{'.','..'}));
    
    tiff2 = dir(strcat(rootfolder, '/', 'Segmented/*AutoThreshSegm.tiff'));
    tiff2 = tiff2(~ismember({tiff2(:).name},{'.','..'}));
    
    %t1 = Tiff(strcat(tiff1(1).folder, '/', tiff1(1).name), 'r+');
    
    options.overwrite = true;
    img = double(loadtiff(strcat(tiff1(1).folder, '/', tiff1(1).name)));
    img = abs(255 * img);
    img = uint8(img);
    saveastiff(img,strcat(tiff1(1).folder, '/', tiff1(1).name),options);
    
    %img2 = loadtiff(strcat(tiff2(1).folder, '/', tiff2(1).name));
    
    %close(t1)
    
end