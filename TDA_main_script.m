%% Script to run topological data analysis code
clc
clear
close

% Set directory of tumour folders
main_directory = '';

% Checking and removing redundant directories
tumour_list = dir(main_directory);
tumour_list = tumour_list([tumour_list(:).isdir]);
tumour_list = tumour_list(~ismember({tumour_list(:).name},{'.','..','TDA_Summary'}));

% Generate '.nii' files for skeletonisation
get_nii(tumour_list);

% Generate skeletons externally on Python IDE using getSkeleton.py in folder 2_Data_Extraction

% Analyse vascular skeletons
FormatVesselsForRadialFiltration(tumour_list);
ConstructRadialFiltrationParallelScript(tumour_list);
getNetworkAdjacencyMatrices(tumour_list);
getVesselStats(tumour_list);
    
% Script retrieves the data analysed obove and outputs tables containing vascular descriptor values in the folder 'TDA_Summary'
organiseVesselStats(tumour_list, main_directory);
