function [Distance_matrix_square] = DistanceMatrix2Point_clouds(point_cloud_1,point_cloud_2)

% gives the distance matrix square!!

    N = size(point_cloud_1,1);
    M = size(point_cloud_2,1);
    
    Distance_matrix_square = real(sum(point_cloud_1.^2,2)*ones(1,M) + ones(N,1)*sum(point_cloud_2.^2,2)' -2.*point_cloud_1*point_cloud_2');
    
end
