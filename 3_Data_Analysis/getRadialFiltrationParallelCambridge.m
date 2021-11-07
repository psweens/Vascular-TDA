function  [starts_dim0, ends_dim0, starts_dim1, ends_dim1] = getRadialFiltrationParallelCambridge(initial_path, folder, category, tumour, output_path, number_of_divisions)

    import edu.stanford.math.plex4.*;
    javaaddpath('../3_Data_Analysis/lib/javaplex.jar');
    
    vessel_file_path = [char(initial_path) char(folder) 'Vessels.mat'];
    
    load(vessel_file_path);
    
    N_vessels = length(Vessels);

    [VesselBranchingPoints,VesselNonBranchingPoints] = ...
        constructVesselPointCloudAllPoints(vessel_file_path);
 
    VesselPointCloud = [VesselBranchingPoints;VesselNonBranchingPoints];
    
     % We define a tumour center

    tumour_center = [(max(VesselPointCloud(:,1))-min(VesselPointCloud(:,1)))/2+min(VesselPointCloud(:,1)),...
        (max(VesselPointCloud(:,2))-min(VesselPointCloud(:,2)))/2+min(VesselPointCloud(:,2)),...
        (max(VesselPointCloud(:,3))-min(VesselPointCloud(:,3)))/2+min(VesselPointCloud(:,3))];

    center_of_mass = [sum(VesselPointCloud(:,1))/size(VesselPointCloud,1),...
    	sum(VesselPointCloud(:,2))/size(VesselPointCloud,1),...
        sum(VesselPointCloud(:,3))/size(VesselPointCloud,1)];
    
    h = figure;
    scatter3(VesselPointCloud(:,1),VesselPointCloud(:,2),VesselPointCloud(:,3))
    hold on
    scatter3(tumour_center(1),tumour_center(2),tumour_center(3),'y', 'filled')
    hold on
    scatter3(center_of_mass(1),center_of_mass(2),center_of_mass(3),'r', 'filled')
    % hold on 
    % scatter3(points_within_radius(:,1),points_within_radius(:,2),points_within_radius(:,3),'y')
    legend('Blood vessel points', 'Coordinates center point', 'Center of mass')
    title(['Tumour ' char(tumour), category], 'Interpreter', 'none','Fontsize',16)

    image_title = [output_path 'Centers.pdf'];
    image_titlefig = [output_path 'Centers.fig'];

    saveas(h,image_title)
    saveas(h,image_titlefig)
    
    
    
     % We look at the distances of all points to the center point and rank them
    % in order

    Distance_matrix_squared = DistanceMatrix2Point_clouds(VesselPointCloud,center_of_mass);

    Distances_to_center = sqrt(Distance_matrix_squared);

    [sorted_Distances_to_center,index_in_Vessel_Point_cloud] = sort(Distances_to_center);

    filtration_end_distance_to_center = max(max(sorted_Distances_to_center))+1; %change this according to maximal tumour size in data set

  % We prepare the javaplex input

    stream = api.Plex4.createExplicitSimplexStream();

    % We increase the radius creating the filtration

    minimal_radius = filtration_end_distance_to_center/number_of_divisions;
    distance_vector_index = 1;
    filtration_step = 0;
    VesselPointCloud_point_indices_in_filtration = [];
    old_radius = 0;
    
    
    
    for radius = (filtration_end_distance_to_center/2+minimal_radius):minimal_radius:filtration_end_distance_to_center
       
       % Find points that are within that radius of the center and and add
    % them to the stream
        if size(VesselPointCloud_point_indices_in_filtration,1) ~= size(VesselPointCloud,1)    

            if sorted_Distances_to_center(distance_vector_index) <= radius

                sorted_point_indices_within_radius = find(old_radius < sorted_Distances_to_center & sorted_Distances_to_center <= radius);
                distance_vector_index = distance_vector_index + length(sorted_point_indices_within_radius);

                points_within_radius = VesselPointCloud(index_in_Vessel_Point_cloud(sorted_point_indices_within_radius),:);

                for point_number = 1:size(points_within_radius,1)

                    point = points_within_radius(point_number,:);
                    
                    % depending whether we had to stretch data or not we
                    % use a different neighbour function

                    direct_neighbours = getNeighbours_new(point,Vessels,VesselBranchingPoints);

                    if isempty(direct_neighbours) == 0

                        [tf,direct_neighbour_VesselPointCloud_indices] = ismember(direct_neighbours,VesselPointCloud,'rows');

                        [tf2,point_row_index_in_VesselPointCloud] = ismember(point,VesselPointCloud,'rows');

                        % Check whether the neighbour points are part of the filtration
                        % already

                        if sum(ismember(direct_neighbour_VesselPointCloud_indices,VesselPointCloud_point_indices_in_filtration)) == 0

                            stream.addVertex(point_row_index_in_VesselPointCloud, radius);
                            VesselPointCloud_point_indices_in_filtration = [VesselPointCloud_point_indices_in_filtration;point_row_index_in_VesselPointCloud];

                        else

                            % Find neighbours that are in stream, connect with edge

                            neighbour_indices_in_stream = [];

                            for j = 1:length(direct_neighbour_VesselPointCloud_indices)

                                isinstream_candidate = direct_neighbour_VesselPointCloud_indices(j);
                                isinstream = find(VesselPointCloud_point_indices_in_filtration==isinstream_candidate);
                                neighbour_indices_in_stream = [neighbour_indices_in_stream;VesselPointCloud_point_indices_in_filtration(isinstream)];

                            end

                            for k = 1:length(neighbour_indices_in_stream)

                                stream.addElement([point_row_index_in_VesselPointCloud,neighbour_indices_in_stream(k)], radius);
                                if k == 1
                                    stream.addVertex(point_row_index_in_VesselPointCloud, radius);
                                    VesselPointCloud_point_indices_in_filtration = [VesselPointCloud_point_indices_in_filtration;point_row_index_in_VesselPointCloud];
                                end

                            end

                            if isempty(neighbour_indices_in_stream) == 1
                                'keyboard'
                            end

                        end

                    else

                            [tf2,point_row_index_in_VesselPointCloud] = ismember(point,VesselPointCloud,'rows');
                            stream.addVertex(point_row_index_in_VesselPointCloud, radius);
                            VesselPointCloud_point_indices_in_filtration = [VesselPointCloud_point_indices_in_filtration;point_row_index_in_VesselPointCloud];

                    end

                    direct_neighbours = [];



                end

                points_within_radius = [];
                sorted_point_indices_within_radius = [];

            end

        end

        old_radius = radius;
        filtration_step = filtration_step + 1;

        if size(VesselPointCloud_point_indices_in_filtration,1)+1 ~= distance_vector_index
            keyboard
        end


    end
    
    

    stream.finalizeStream();%added

    stream.validateVerbose()
    
    
    % get persistence algorithm over Z/2Z
    persistence = api.Plex4.getModularSimplicialAlgorithm(3, 2);

    % compute and print the intervals
    intervals = persistence.computeIntervals(stream);

    % % compute and print the intervals annotated with a representative cycle
    Annotatedintervals = persistence.computeAnnotatedIntervals(stream);

    % % get the infinite barcodes
    infinite_barcodes = intervals.getInfiniteIntervals();
    
    %%% We print the folder
    
    folder
    
    %%%
    
    % 
    % % print out betti numbers in form {dimension: betti number}
    betti_numbers_string = infinite_barcodes.getBettiNumbers()
    
    % create the barcode plots
    options.filename = ['Radial filtration on Tumour ' char(tumour)];
    options.max_filtration_value = radius;
    options.max_dimension = 1;
    h3 = plot_barcodes_javaPlex(intervals, options);

    image_title2 = [output_path 'BarcodeRadial.pdf'];
    saveas(h3,image_title2)

    image_title3 = [output_path 'BarcodeRadial.fig'];
    saveas(h3,image_title3)


    [starts_dim0, ends_dim0] = starts_and_ends(intervals, 0);
    [starts_dim1, ends_dim1] = starts_and_ends(intervals, 1);
    
    dim0 = [starts_dim0', ends_dim0'];
    dim1 = [starts_dim1', ends_dim1'];

    fileID = fopen([output_path 'BarcodeDim0.txt'],'w');
    fmt = '%d %d\n';
    fprintf(fileID,fmt, dim0');
    fclose(fileID);

    fileID = fopen([output_path 'BarcodeDim1.txt'],'w');
    fmt = '%d %d\n';
    fprintf(fileID,fmt, dim1');
    fclose(fileID);

    %Betti curves
    filtration_step = 1;

    for radius = minimal_radius:minimal_radius:filtration_end_distance_to_center

        %Dim0

        Betti0(filtration_step) = sum(dim0(:,1) <= radius & (radius <= dim0(:,2)| dim0(:,2) == -1));

        %Dim1

        Betti1(filtration_step) = sum(dim1(:,1) <= radius & (radius <= dim1(:,2)| dim1(:,2) == -1));

        filtration_step = filtration_step + 1;

    end
    
    
    save([output_path 'Betti0.mat'],'Betti0')
    save([output_path 'Betti1.mat'],'Betti1')

    image_title6 = [output_path 'Betti0Curve.pdf'];
    image_title7 = [output_path 'Betti0Curve.fig'];

    image_title4 = [output_path 'Betti1Curve.pdf'];
    image_title5 = [output_path 'Betti1Curve.fig'];


    set(0, 'DefaulttextInterpreter', 'none')

    supertitle = [char(category)  ' tumour ' char(tumour)];
    
   % supertitle = [char(category)  ' tumour ' char(tumour) ', Day ' char(day) ];

    h1 = figure;
    plot(minimal_radius:minimal_radius:filtration_end_distance_to_center,Betti0)
    xlabel('Distance to tumour center')
    ylabel('Betti 0')
    title('Betti 0 curve')
    suptitle(supertitle)

    h2 = figure;
    plot(minimal_radius:minimal_radius:filtration_end_distance_to_center,Betti1)
    xlabel('Distance to tumour center')
    ylabel('Betti 1')
    title('Betti 1 curve')
    suptitle(supertitle)

    saveas(h1,image_title6)
    saveas(h1,image_title7)

    saveas(h2,image_title4)
    saveas(h2,image_title5)
    
    
    
    clear Vessels
    
    VesselBranchingPoints = [];
    VesselNonBranchingPoints = [];
    VesselPointCloud = [];
    Distance_matrix_squared = [];
    Distances_to_center = [];
    sorted_Distances_to_center = [];
    index_in_Vessel_Point_cloud = [];
    VesselPointCloud_point_indices_in_filtration = [];
    Betti0 = [];
    Betti1 = [];


end