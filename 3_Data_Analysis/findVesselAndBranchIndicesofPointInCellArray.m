function [VesselIndex,BranchIndex] = findVesselAndBranchIndicesofPointInCellArray(cellarray,point)
    
    VesselIndex = [];
    BranchIndex = [];
    
    Vessels = cellarray;
    
    N_vessels = numel(Vessels);
    

    for vessel = 1:N_vessels
        
        for branch = 1:numel(Vessels{vessel}.Branch)
            
            if ismember(point,Vessels{vessel}.Branch{branch}, 'rows')
                VesselIndex = [VesselIndex;vessel];
                BranchIndex = [BranchIndex;branch];
    
            end
            
        end
        
    end

end