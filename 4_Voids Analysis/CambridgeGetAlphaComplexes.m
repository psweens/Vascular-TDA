clc
clear
close

%% Construct alpha complex with GUDHI
% 2.8.2020

main_directory = '/media/sweene01/SSD/Lina/ilastik_medfilt_CyclingHypoxia'; 
tumour_list = dir(main_directory);
tumour_list = tumour_list([tumour_list(:).isdir]);
tumour_list = tumour_list(~ismember({tumour_list(:).name},{'.','..'}));

for tumour_idx = 1:size(tumour_list,1)
    
    initial_path = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/skeletons');

    folders = dir(initial_path);
    folders = folders([folders(:).isdir]);
    folders = folders(~ismember({folders(:).name},{'.','..'}));

    for data_file_number = 1:size(folders,1)

        %category = categories(data_file_number);
    %     tumour = tumours(data_file_number);
        tumour = folders(data_file_number).name;
        %[pathstr, tumour, ext] = fileparts(tumour);

        %point_cloud_filename =  [char(category) '_' char(tumour)];
        point_cloud_filename =  char(tumour);

        input_path = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/','GUDHI_Data/Input/',num2str(point_cloud_filename),'.txt');
        %input_path = char(['GUDHI_data/Input/' num2str(point_cloud_filename) '.txt']);

        output_path = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/','GUDHI_Data/Output/',num2str(point_cloud_filename),'.txt');
        %output_path = ['GUDHI_data/Output/' num2str(point_cloud_filename) '.txt'];
        
        if ~exist(strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/','GUDHI_Data/Output/', 'dir'))
           mkdir(strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/','GUDHI_Data/Output/'))
        end
        
        input_path = replace(input_path, 'Simple Segmentation', 'Simple_Segmentation')
        output_path = replace(output_path, 'Simple Segmentation', 'Simple_Segmentation')

        sprintf(['GUDHI code opened file for ' point_cloud_filename])

        unixstring=['/home/sweene01/anaconda3/envs/ox_topo/bin/alpha_complex_persistence -p 2 -o ' output_path ' ' input_path];
        [status,cmdout] = unix(unixstring);

        gudhi_reformat_output( output_path );

        unixstring = [];
        unixstring=['rm ', output_path];
        [status,cmdout] = unix(unixstring);



        sprintf(['Done with folder' char(folders(data_file_number).name)])


    end
    
end


