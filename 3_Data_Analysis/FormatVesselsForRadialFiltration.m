%% Format Vessels
% This code takes the output files from Russ' segmentation codes and
% combines them in one .mat file in a similar format of his previous Vessels.mat
% files
% 
% 29.7.2020
% Bernadette Stolz-Pretzer

function FormatVesselsForRadialFiltration(tumour_list)

    for tumour_idx = 1:size(tumour_list,1)

        % Data directory
        folder_name = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/Skeletons');

        %folder_name = strcat(tumour_list(skel_idx).folder, '/', tumour_list(skel_idx).name, '/');

        file_list = dir(folder_name);
        file_list = file_list([file_list(:).isdir]);
        file_list = file_list(~ismember({file_list(:).name},{'.','..'}));

        for i = 1:size(file_list,1)

            Vessels = {};

            component_count = 1;
            branch_count = 1;
            filename = strcat(file_list(i).folder, '/', file_list(i).name, '/Components/Component', num2str(component_count), 'Branch', num2str(branch_count), '.txt');
            %filename = [char(folder_name) char(file_list(i)) 'Components/Component' num2str(component_count) 'Branch' num2str(branch_count) '.txt'];


            output_dir = strcat(file_list(i).folder, '/', file_list(i).name, '/');

            while exist(filename) == 2

                while exist(filename) == 2

                    [component_count,branch_count];
                    %branch_count

                    Vessels{component_count}.Branch{branch_count} = load(filename);

        %             filename2 = ['/Users/user/Desktop/Cambridge Data/' char(file_list(i)) 'AdjMatrix' num2str(component_count) '.mat'];
        %             load(filename2);
        %             Vessels{component_count}.AdjMatrix = b;

                    %Branches{branch_count} = load(filename);

                    branch_count = branch_count + 1;

                    %filename = [char(folder_name) char(file_list(i)) 'Components/Component' num2str(component_count) 'Branch' num2str(branch_count) '.txt'];
                    filename = strcat(file_list(i).folder, '/', file_list(i).name, '/Components/Component', num2str(component_count), 'Branch', num2str(branch_count), '.txt');

                    %filename = [char(file_list(1)) 'Branch' num2str(branch_count) '.txt'];


                end

                component_count = component_count + 1;
                branch_count = 1;

                filename = strcat(file_list(i).folder, '/', file_list(i).name, '/Components/Component', num2str(component_count), 'Branch', num2str(branch_count), '.txt');
                %filename = [char(folder_name) char(file_list(i)) 'Components/Component' num2str(component_count) 'Branch' num2str(branch_count) '.txt'];


            end


            output = [output_dir 'Vessels.mat'];
            save(output, 'Vessels')

            clear Vessels

        end

    end


end
