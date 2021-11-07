% This script prepares the javaplex library for use

clc; clear all; close all;
% clear import;

javaaddpath('./3_Data_Analysis/lib/javaplex.jar');
import edu.stanford.math.plex4.*;

javaaddpath('./3_Data_Analysis/lib/plex-viewer.jar');
import edu.stanford.math.plex_viewer.*;

%cd './3_Data_Analysis/utility';
addpath(pwd);
%cd '..';


