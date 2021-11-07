function outline = makeMultiOutline( inputData )
%MAKEMULTIOUTLINE Create outlines of all connected areas of the same
%Brightness in a 3D dataset.
%   Creates 2D outlines working on a plane-by-plane basis across the second
%   dimension (coronal planes if data follows .nii convention)

outline = zeros(size(inputData));

% the following (commented out) code is more accurate since it creates 
% correct 3D outlines, but runs excruciatingly slow
%
%allVals = unique(inputData);
%disp(['Found ' num2str(numel(allVals)) ' areas']);
% for i = 1:numel(allVals)
%     disp(['Calculating ' num2str(i)]);
%     newOutline = bwperim(inputData==allVals(i), 26)*allVals(i);
%     outline = outline+newOutline;
% end

for z = 1:size(outline,2)
    allVals = unique(inputData(:,z,:));
    disp(['Found ' num2str(numel(allVals)) ' areas on slice ' num2str(z)]);
    currSliceOut = squeeze(outline(:,z,:));
    currSliceIn = squeeze(inputData(:,z,:));
    parfor i = 1:numel(allVals)
        currSliceOut = currSliceOut+bwperim(currSliceIn==allVals(i), 8)*allVals(i);
    end
    outline(:,z,:) = currSliceOut;
end