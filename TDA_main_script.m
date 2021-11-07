%% Script to run topological data analysis code
clc
clear
close

% load_javaplex
% import edu.stanford.math.plex4.*;

% Set directory of tumour folders
main_directory = '/media/sweene01/SSD/VA_Paper_Datasets/Lnet_Topological_Analysis_Isotropic/';
python_path = '/home/sweene01/anaconda3/envs/ox_topo/bin/python3.8';
pe = pyenv;
if pe.Status == "NotLoaded"
    pe = pyenv("ExecutionMode","OutOfProcess","Version",python_path);
end
% pyenv("Version",python_path);
% pyenv("ExecutionMode","OutOfProcess")

%addCondaEnv(python_libs);
%pyenv('Version', '~/anaconda3/envs/ox_topo/bin/python');

% Checking and removing redundant directories
tumour_list = dir(main_directory);
tumour_list = tumour_list([tumour_list(:).isdir]);
tumour_list = tumour_list(~ismember({tumour_list(:).name},{'.','..','TDA_Summary'}));

% Generate '.nii' files for skeletonisation
%get_nii(tumour_list);

% Generate skeletons
%system(['python 2_Data_Extraction/getSkeleton.py ',main_directory]);
%pe.list({'hi'})
%pyrunfile('2_Data_Extraction/getSkeleton.py',main_directory)

% Analyse vascular skeletons
%FormatVesselsForRadialFiltration(tumour_list);
%ConstructRadialFiltrationParallelScript(tumour_list);
%getNetworkAdjacencyMatrices(tumour_list);
getVesselStats(tumour_list);
    
organiseVesselStats(tumour_list, main_directory);
