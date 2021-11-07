function nii = tiff2nii(tiff, view, voxelSize, varargin)
%TIFF2NII Loads a 3D tiff file and converts it to the Nifty format
%   tiff: Path to the tiff stack (char) or matlab-loaded tiff stack ([Y X Z]
%   axis order).
%   view: View defined by the XY axis, either 'Coronal', 'Sagittal' or
%   'Axial'
%   voxelSize: Size of a voxel (in mm) in X, Y and Z (e.g. [1 1 5])
%   'flipX', 'flipY', 'flipZ': This tool expects tiffs to be oriented with
%   the dorsal (coronal/saggital view) or anterior (axial view) side on top
%   and the z axis going from anterior to posterior (coronal view), dorsal
%   to ventral (axial view) or left to right (sagittal view). If any of the
%   axes don't match that convention you can use the flags here to flip the
%   relevant axes. These parameters are optional.

    flipX=ismember('flipX', varargin);
    flipY=ismember('flipY', varargin);
    flipZ=ismember('flipZ', varargin);

    if ischar(tiff)
        tiffImg = loadTiffStack(tiff);
          
    elseif isnumeric(tiff)
        tiffImg = tiff;
    end
    
    if flipX
        tiffImg = flip(tiffImg,2);
    end
    
    if flipY
        tiffImg = flip(tiffImg,1);
    end
    
    if flipZ
        tiffImg = flip(tiffImg,3);
    end
    
    %matlab loads tiffs in the order [Y X Z]
    %nifty expects them to be in [X Y Z] with the XY plane containing the
    %Axial view
    switch lower(view)
        case 'axial'
            permuteVect=[2 1 3];
        case 'coronal'
            permuteVect=[2 3 1];
        case 'sagittal'
            permuteVect=[3 2 1]; %first 2 1 3, then 3 1 2
        otherwise
            error('View needs to be either ''Axial'', ''Coronal'' or ''Sagittal''');
            
    end
    tiffImg = permute(tiffImg, permuteVect);
    voxelSize = voxelSize(permuteVect);
    % Z axis of .nii is inverted (starts at bottom of brain pointing
    % upwards)
    tiffImg = flip(tiffImg,3);
    
    nii=make_nii(tiffImg, voxelSize);
end

