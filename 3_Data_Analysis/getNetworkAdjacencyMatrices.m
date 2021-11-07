%% Get Adjacency Matrices from Cambridge data


% This code creates adjacency matrices from the extracted vessel networks:
% Output:
%For full network
% AdjacencyMatrix.mat: Adjacency matrix with entries 0,1 where a(i,j) =
% a(j,i) = 1 iff branch point i and branch point j are connected by a
% vessel
% AdjacencyMatrixDiameter.mat Adjacency matrix with diameter entries where a(i,j) =
% a(j,i) = d, d > 0, iff branch point i and branch point j are connected by a
% vessel of diameter d
% AdjacencyMatrixLength.mat with length entries where a(i,j) =
% a(j,i) = l, l > 0, iff branch point i and branch point j are connected by a
% vessel of length l
% AdjacencyMatrixSOAM.mat with sum of angles metric (tortuosity) entries where a(i,j) =
% a(j,i) = s,  s > 0, iff branch point i and branch point j are connected by a
% vessel of tortuosity s
% AdjacencyMatrixCLR.mat chord-length-ratio (tortuosity) entries where a(i,j) =
% a(j,i) = c,  c > 0, iff branch point i and branch point j are connected by a
% vessel of tortuosity c
% Coordinates.mat: List of 3D coordinates of branching points
% Same as above but considering only the largest connected component of the
% network
% LargestComponentAdjacencyMatrix.mat
% LargestComponentAdjacencyMatrixDiameter.mat
% LargestComponentAdjacencyMatrixLength.mat
% LargestComponentAdjacencyMatrixSOAM.mat
% LargestComponentAdjacencyMatrixCLR.mat
% LargestComponentCoordinates.mat

% Bernadette Stolz
% 24.7.2020


%Adapt this to where your data is

%% Segmentation methods

function getNetworkAdjacencyMatrices(tumour_list)

    %parpool(28) %adjust to machine

    for tumour_idx = 1:size(tumour_list,1)

        % Data directory
        folder_name = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/Skeletons');

        file_list = dir(folder_name);
        file_list = file_list([file_list(:).isdir]);
        file_list = file_list(~ismember({file_list(:).name},{'.','..'}));
        for i = 1:size(file_list,1)
            file = strcat(file_list(i).folder, '/', file_list(i).name, '/Components/');
            file_list(i).name = file;
        end


        %%

        for i = 1:size(file_list,1)

            file_list(i).name
            filename = [file_list(i).name 'Summary_Component1.txt'];
            initial_path = file_list(i).name;

            component = 1;

            epsilon_perturbation_percentage_for_ghost_points = 0.05;

            B_binary = {};
            B_diameter = {};
            B_length = {};
            B_soam = {};
            B_clr = {};
            B_coordinates = {};
            max_component_size = 0;
            while exist(filename) == 2

                component_summary_list = load(filename);

                component_summary_matrix = reshape(component_summary_list,[10,size(component_summary_list,1)/10])';

                B = reshape(component_summary_matrix(:,1:6)',[3,size(component_summary_list,1)/5])';

                A_binary_full = [component_summary_matrix(:,[1,2,3,4,5,6])];
                diameter = [component_summary_matrix(:,7)];
                length = [component_summary_matrix(:,8)];
                soam = [component_summary_matrix(:,9)];
                clr = [component_summary_matrix(:,10)];

                points = unique(reshape(component_summary_matrix(:,1:6)',[3,size(component_summary_list,1)/5])','rows');

                row_numbers_A_binary_full = [1:1:size(A_binary_full,1)];

                for point_index = 1:size(points,1)

                    point = points(point_index,:);

                    tf =ismember(A_binary_full(:,1:3),point,'rows');

                    if sum(tf) ~= 0

                        index_1 = row_numbers_A_binary_full(tf);
                        A_binary(index_1,1) = point_index;
                        A_diameter(index_1,1) = point_index;
                        A_diameter(index_1,3) = diameter(index_1);
                        A_length(index_1,1) = point_index;
                        A_length(index_1,3) = length(index_1);
                        A_soam(index_1,1) = point_index;
                        A_soam(index_1,3) = soam(index_1);
                        A_clr(index_1,1) = point_index;
                        A_clr(index_1,3) = clr(index_1);

                    end


                    tf =ismember(A_binary_full(:,4:6),point,'rows');


                    if sum(tf) ~= 0

                        index_2 = row_numbers_A_binary_full(tf);
                        A_binary(index_2,2) = point_index;
                        A_diameter(index_2,2) = point_index;
                        A_length(index_2,2) = point_index;
                        A_soam(index_2,2) = point_index;
                        A_clr(index_2,2) = point_index;

                    end



                end


                coords = points;

                % We introduce ghost points in cases where Russ' code gives more than
                % one edge between two nodes

                ghost_points_per_component(component) = 0;

                if size(unique(A_binary,'rows'),1) ~= size(A_binary,1)

                    [b,m,n] = unique(A_binary,'rows');

                    i=true(size(A_binary,1),1);
                    i(m)=false;

                    c = unique(A_binary(i,:),'rows');

                    %ghost_points_per_component(component) = 0;

                    for rows_in_c = 1:size(c,1)

                        current_row = c(rows_in_c,:);

                        tf = ismember(A_binary,current_row,'rows');
                        affected_rows_in_A_binary = row_numbers_A_binary_full(tf);
                        number_of_ghost_points =  size(affected_rows_in_A_binary,2)-1;

                        branch_point_index_1 = A_binary(affected_rows_in_A_binary(1),1);
                        branch_point_1 = points(branch_point_index_1,:);

                        branch_point_index_2 = A_binary(affected_rows_in_A_binary(1),2);
                        branch_point_2 = points(branch_point_index_2,:);

            %             [VesselIndex_1,BranchIndex_1] = findVesselAndBranchIndicesofPointInCellArray(Vessels,branch_point_1);
            %             [VesselIndex_2,BranchIndex_2] = findVesselAndBranchIndicesofPointInCellArray(Vessels,branch_point_2);

                        ghost_points_per_component(component) = ghost_points_per_component(component) + number_of_ghost_points;

                        if number_of_ghost_points > 2

                            %keyboard

                        end

                        for ghost_point_index = 1:number_of_ghost_points

                            point_index = point_index + 1;

                            epsilon = norm(branch_point_1 - branch_point_2)*epsilon_perturbation_percentage_for_ghost_points;

                            coords(point_index,:) = (branch_point_1 + branch_point_2)/2+epsilon/sqrt(3)*ones(1,3);

                            A_binary(end+1,:) = [point_index,A_binary(affected_rows_in_A_binary(ghost_point_index),2)];
                            A_diameter(end+1,:) = [point_index,A_binary(affected_rows_in_A_binary(ghost_point_index),2),diameter(affected_rows_in_A_binary(ghost_point_index))];
                            A_length(end+1,:) = [point_index,A_binary(affected_rows_in_A_binary(ghost_point_index),2),length(affected_rows_in_A_binary(ghost_point_index))];
                            A_soam(end+1,:) = [point_index,A_binary(affected_rows_in_A_binary(ghost_point_index),2),soam(affected_rows_in_A_binary(ghost_point_index))];
                            A_clr(end+1,:) = [point_index,A_binary(affected_rows_in_A_binary(ghost_point_index),2),clr(affected_rows_in_A_binary(ghost_point_index))];

                            A_binary(affected_rows_in_A_binary(ghost_point_index),2) = point_index;
                            A_diameter(affected_rows_in_A_binary(ghost_point_index),2) = point_index;
                            A_length(affected_rows_in_A_binary(ghost_point_index),2) = point_index;
                            A_soam(affected_rows_in_A_binary(ghost_point_index),2) = point_index;
                            A_clr(affected_rows_in_A_binary(ghost_point_index),2) = point_index;

                        end






                    end


                end

                if max([max_component_size,point_index]) ~= max_component_size

                    max_component_number = component;

                    max_component_size = point_index;

                    inlet = B(1,:);

                    [tf,i] = ismember(points,inlet,'rows');

                    x = [1:1:point_index];

                    inlet_node_nr = x(tf);


                end


                A_binary = [A_binary, ones(size(A_binary,1))];


                % create sparse matrices for all types

                B_binary{component} = sparse([A_binary(:,1);A_binary(:,2)],[A_binary(:,2);A_binary(:,1)],[A_binary(:,3),A_binary(:,3)]);
                B_diameter{component} = sparse([A_diameter(:,1);A_diameter(:,2)],[A_diameter(:,2);A_diameter(:,1)],[A_diameter(:,3);A_diameter(:,3)]);
                B_length{component} = sparse([A_length(:,1);A_length(:,2)],[A_length(:,2);A_length(:,1)],[A_length(:,3);A_length(:,3)]);
                B_soam{component} = sparse([A_soam(:,1);A_soam(:,2)],[A_soam(:,2);A_soam(:,1)],[A_soam(:,3);A_soam(:,3)]);
                B_clr{component} = sparse([A_clr(:,1);A_clr(:,2)],[A_clr(:,2);A_clr(:,1)],[A_clr(:,3);A_clr(:,3)]);
                B_coordinates{component} = coords;


                component = component + 1;
                filename = [initial_path, 'Summary_Component' num2str(component) '.txt'];

                clear points
                clear component_summary_matrix
                clear A_binary_full
                clear diameter
                clear length
                clear soam
                clear clr
                clear A_binary
                clear A_diameter
                clear A_length
                clear A_soam
                clear A_clr
                clear number_of_ghost_points
                clear coords

            end

            % Make block matrix using B = blkdiag(A1,A2,A3)
            full_adjacency_matrix = full(blkdiag(B_binary{:}))-diag(full(blkdiag(B_binary{:})));
            full_adjacency_matrix_diameter = full(blkdiag(B_diameter{:}))-diag(full(blkdiag(B_diameter{:})));
            full_adjacency_matrix_length = full(blkdiag(B_length{:}))-diag(full(blkdiag(B_length{:})));
            full_adjacency_matrix_soam = full(blkdiag(B_soam{:}))-diag(full(blkdiag(B_soam{:})));
            full_adjacency_matrix_clr = full(blkdiag(B_clr{:}))-diag(full(blkdiag(B_clr{:})));
            full_coordinates = cell2mat(B_coordinates');


            max_component_number
            largest_component_adjacency_matrix = full(B_binary{max_component_number})-diag(full(B_binary{max_component_number}));
            largest_component_adjacency_matrix_diameter = full(B_diameter{max_component_number})-diag(full(B_diameter{max_component_number}));
            largest_component_adjacency_matrix_length = full(B_length{max_component_number})-diag(full(B_length{max_component_number}));
            largest_component_adjacency_matrix_soam = full(B_soam{max_component_number})-diag(full(B_soam{max_component_number}));
            largest_component_adjacency_matrix_clr = full(B_clr{max_component_number})-diag(full(B_clr{max_component_number}));
            largest_component_coordinates =  B_coordinates{max_component_number};

            ghost_points = ghost_points_per_component(max_component_number)
            nodes = max_component_size
            edges = nnz(largest_component_adjacency_matrix)/2

            save([initial_path,'AdjacencyMatrix.mat'],'full_adjacency_matrix')
            save([initial_path,'AdjacencyMatrixDiameter.mat'],'full_adjacency_matrix_diameter')
            save([initial_path,'AdjacencyMatrixLength.mat'],'full_adjacency_matrix_length')
            save([initial_path,'AdjacencyMatrixSOAM.mat'],'full_adjacency_matrix_soam')
            save([initial_path,'AdjacencyMatrixCLR.mat'],'full_adjacency_matrix_clr')
            save([initial_path,'Coordinates.mat'],'full_coordinates')

            save([initial_path, 'LargestComponentAdjacencyMatrix.mat'],'largest_component_adjacency_matrix')
            save([initial_path, 'LargestComponentAdjacencyMatrixDiameter.mat'],'largest_component_adjacency_matrix_diameter')
            save([initial_path, 'LargestComponentAdjacencyMatrixLength.mat'],'largest_component_adjacency_matrix_length')
            save([initial_path, 'LargestComponentAdjacencyMatrixSOAM.mat'],'largest_component_adjacency_matrix_soam')
            save([initial_path, 'LargestComponentAdjacencyMatrixCLR.mat'],'largest_component_adjacency_matrix_clr')

            save([initial_path,'LargestComponentCoordinates.mat'],'largest_component_coordinates')

        end



    end

end



