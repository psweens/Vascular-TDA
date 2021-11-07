function [starts_dim, ends_dim] = starts_and_ends(intervals, dimension)

% This function collects all start and end points of the intervals of a
% filtration at a specific dimension
% Bernadette Stolz
% 26th May 2015
% Input arguments: itervals - persistence intervals from javaPlex
%                    max_filtration_value - from the filtration used
%                   dimension of interest
% Output arguments: starts in vector, ends in vector
%(end = infty is coded as end = max_filtration_value + 1)
                

intervals_dim=intervals.getIntervalsAtDimension(dimension);

% Get start and endpoints for dimension:

ends_dim = zeros(1,intervals_dim.size());
starts_dim = zeros(1,intervals_dim.size());


for index = 1:intervals_dim.size()
    
    if intervals_dim.get(index-1).isRightInfinite() == 1
        ends_dim(index) = -1;
        starts_dim(index) = intervals_dim.get(index-1).getStart();
    else
        ends_dim(index) = intervals_dim.get(index-1).getEnd();
        starts_dim(index) = intervals_dim.get(index-1).getStart();
    end
    
    length_dim(index) = ends_dim(index) - starts_dim(index);
end


end