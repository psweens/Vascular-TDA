function getTextFilesForGUDHI(initial_path, folder, tumour, category, output_path)


    vessel_file_path = [char(initial_path) char(folder) 'Vessels.mat'];
    
    load(vessel_file_path);
    
    N_vessels = length(Vessels);

    [VesselBranchingPoints,VesselNonBranchingPoints] = ...
        constructVesselPointCloudAllPoints(vessel_file_path);
    
    point_cloud = [VesselBranchingPoints;VesselNonBranchingPoints];
  
    point_cloud_filename =  [char(tumour)]; %[char(category) '_' char(tumour)]
    point_cloud_filename2 =  [num2str(point_cloud_filename) '.txt'];
    point_cloud_filename3 =  [output_path num2str(point_cloud_filename) '.txt']
    
    [number_of_points,point_cloud_dimension] = size(point_cloud);
    
    fileID = fopen(point_cloud_filename3,'w')
    fprintf(fileID,'OFF\n');
    fprintf(fileID,'%d 0 0\n',number_of_points );
    fprintf(fileID,'%f %f %f\n',point_cloud');
    fclose(fileID);


end