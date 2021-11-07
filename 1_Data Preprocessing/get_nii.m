%% Convert .tif stacks (in uint8 format - you can save the data in this format in ImageJ) 
% into .nii format which is required for the further code
% Bernadette Stolz 2020

% This script requires the tiff2nii function from matlabTools, add this directory to
% your matlab path

% Load details of '.tiff' files from directory into struct

% Need to convert to uint8 format first and then .tif for this to work

function get_nii(tumour_list)

    voxel_size = [1 1 1];

    for tumour_idx = 1:size(tumour_list,1)

        rootfolder = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name);

        file_list = dir(strcat(rootfolder, '/', 'Segmented/*'));
        file_list = file_list(~ismember({file_list(:).name},{'.','..'}));

        new_dir = strcat(rootfolder, '/', 'nii_files/');
        if ~exist(new_dir, 'dir')
           mkdir(new_dir)
        end

        for file = 1:size(file_list,1)

            % Extract file name
            [filepath,name,ext] = fileparts(file_list(file).name);

            % Create name for '.nii' file
            nii_filename = fullfile(new_dir,[name '.nii']);

            % Load '.tiff' and convert to '.nii'
            fprintf('Generating .nii file for %s\n',name)
            nii = tiff2nii(char(fullfile(file_list(file).folder,file_list(file).name)),...
                'Axial', voxel_size);
            save_nii(nii, char(nii_filename))

        %     nii = tiff2nii(char(filenames(file)), 'Axial', [0.02 0.02 0.004])
        %     save_nii(nii, char(nii_filenames(file)))

        end

    end

end