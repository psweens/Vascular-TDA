function calculateBloodVolume(tumour_list, subdiv, output_path)

    data = zeros(1,subdiv);

    fileID = fopen(strcat(output_path,'/BloodVolume_Summary_mm3.txt'), 'w');
    for i = 1:size(tumour_list,1)

        % Data directory
        path = strcat(tumour_list(i).folder, '/', tumour_list(i).name, '/skeletons/');

        method_list = dir(path);
        method_list = method_list([method_list(:).isdir]);
        method_list = method_list(~ismember({method_list(:).name},{'.','..'}));

        for j = 1:size(method_list,1)

            subpath = strcat(path, method_list(j).name, '/Network_Summary.txt');
            networkData = importdata(subpath,',',0);
            networkData(:,1) = 0.5 * networkData(:,1);
            volume = sum(pi .* (networkData(:,1).^2) .* networkData(:,2)) .* 1e-9;

            data(1,j) = volume;

        end

        fprintf(fileID,'%s, ', tumour_list(i).name);
        for j = 1:subdiv
            fprintf(fileID, '%f, ', data(1,j));
        end
        fprintf(fileID,'\n');

    end
    fclose(fileID);

end
