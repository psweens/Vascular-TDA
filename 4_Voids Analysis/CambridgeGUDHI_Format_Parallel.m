%% Construct alpha complex with GUDHI
% 14.12.2019


%delete(gcp) 

clear all
clc

main_directory = '/media/sweene01/SSD/Lina/ilastik_medfilt_HIF_Pilot'; 
tumour_list = dir(main_directory);
tumour_list = tumour_list([tumour_list(:).isdir]);
tumour_list = tumour_list(~ismember({tumour_list(:).name},{'.','..'}));

for tumour_idx = 1:size(tumour_list,1)
    
    % Data directory
    initial_path = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/skeletons');

    folders = dir(initial_path);
    folders = folders([folders(:).isdir]);
    folders = folders(~ismember({folders(:).name},{'.','..'}));


    parfor data_file_number = 1:size(folders,1)
    %for data_file_number = 1:length(tumours)

        tumour = folders(data_file_number).name;%tumours(data_file_number);
        [pathstr, tumour, ext] = fileparts(tumour);
        %tumour = replace(tumour,'Simple Segmentation','Simple_Segmentation'); 
        

        category = '';%categories(data_file_number);

        folder = strcat('/', folders(data_file_number).name, '/');

        output_path = [strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/','GUDHI_Data/Input/')]
        
        if ~exist(output_path, 'dir')
           mkdir(output_path)
        end

        getTextFilesForGUDHI(char(initial_path), folder, tumour, category, output_path);

        sprintf(['Done with folder: ' char(tumour)])

    end

    %delete(gcp)     
    
end

