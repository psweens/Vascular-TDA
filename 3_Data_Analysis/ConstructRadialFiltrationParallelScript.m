%% Construct radial filtration
% This code constructs a radial filtration based on a blood vessel network point cloud
% Bernadette Stolz
% 28.7.2020

function ConstructRadialFiltrationParallelScript(tumour_list)

    %parpool(28) %adjust to machine

    for tumour_idx = 1:size(tumour_list,1)

        % Data directory
        folder_name = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/Skeletons');
        initial_path = folder_name;

        folders = dir(folder_name);
        folders = folders([folders(:).isdir]);
        folders = folders(~ismember({folders(:).name},{'.','..'}));

        for data_file_number = 1:size(folders,1)

            category = '';
            tumour = folders(data_file_number).name;

            folder = strcat('/', folders(data_file_number).name, '/')
            output_path = strcat(folders(data_file_number).folder, '/', folders(data_file_number).name, '/');

            number_of_divisions = 500; %We set the number of divisions 

            [starts_dim0, ends_dim0, starts_dim1, ends_dim1] = ...
                getRadialFiltrationParallelCambridge(initial_path, folder, category, tumour, output_path, number_of_divisions);

            sprintf(['Done with folder ' char(folder)])

            starts_dim0 = [];
            ends_dim0 = [];
            starts_dim1 = [];
            ends_dim1 = [];


        end
    end

end
