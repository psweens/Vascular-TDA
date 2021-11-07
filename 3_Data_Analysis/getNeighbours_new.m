function [direct_neighbours] = getNeighbours_new(point,cellarray,VesselBranchingPoints)           

    Vessels = cellarray; 
    direct_neighbours = [];
    [VesselIndex,BranchIndex] = findVesselAndBranchIndicesofPointInCellArray(Vessels,point);
    
     % We check the indices of the first point
            
    branch_points = Vessels{VesselIndex(1)}.Branch{BranchIndex(1)};
    [tf,point_row_index_on_branch] = ismember(point,branch_points,'rows'); 

    if ismember(point,VesselBranchingPoints,'rows') == 0
            
        if size(VesselIndex,1) > 1
            
            sprintf('There is a nonbranching point that is member of multiple branches or vessels.')
            keyboard
            
        end
        
        direct_neighbours = [direct_neighbours;branch_points(point_row_index_on_branch-1,:);branch_points(point_row_index_on_branch+1,:)];
                
    else  
        
       for branch = 1:size(BranchIndex,1)
            
            % We check the indices of the Branching point
            
            branch_points = Vessels{VesselIndex(branch)}.Branch{BranchIndex(branch)};
            [tf,point_row_index_on_branch] = ismember(point,branch_points,'rows');  
           
           % starting point
            if point_row_index_on_branch == 1 & size(branch_points,1) ~= 1
                
                direct_neighbours = [direct_neighbours;branch_points(2,:)];
                
                %end point
            elseif point_row_index_on_branch == size(branch_points,1) & size(branch_points,1) ~= 1
                
                direct_neighbours = [direct_neighbours;branch_points(end-1,:)];
          
            elseif size(branch_points,1) ~= 1
                
                direct_neighbours = [direct_neighbours;branch_points(point_row_index_on_branch-1,:);branch_points(point_row_index_on_branch+1,:)];
            
            end
            
            branch_points = [];
     
       end
        
    end
 
    
    direct_neighbours = unique(direct_neighbours,'rows');
    
end