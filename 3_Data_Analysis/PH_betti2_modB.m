function [beta0, beta1, biggest0, biggest1] = PH_betti2(C, thresholds)
% Computes the 0-th and 1st Betti numbers (the number of connected components and cycles) 
% and the size of the largest cycle over the range of thresholds in connectivity matrix C
%
% C           : Weighted connectivity matrix. 
% thresholds  : Range of thresholds to use in C. For instance [0:0.01:1] will tresholds C
%               between [0,1] at 0.01 increment. If the total number of thresholds is bigger than 200, 
%               MATLAB may not be able to compute it. 
% beta0, beta1: 0th and 1st Betti numbers
% biggest0    : the size of the largest number of components
% biggest1    : the size of the largest cycle, in terms of the number of nodes (or eddges). 
%               Not implemented yet
%
% The mathematical details of the code is given in the following paper
%
% [1] Chung, M.K., Vilalta, V.G., Lee, H., Rathouz, P.J., Lahey, B.B., Zald, D.H. 
%     2017 Exact topological inference for paired brain networks via persistent 
%     homology. Information Processing in Medical Imaging (IPMI) 10265:299-310
%     http://pages.stat.wisc.edu/%7Emchung/papers/chung.2017.IPMI.pdf
%
% [2]  Chung, M.K., Luo, Z., Leow, A.D., Alexander, A.L., Richard, D.J., Goldsmith, H.H. 
%      2018 Exact Combinatorial Inference for Brain Images, Medical Image Computing and 
%      Computer Assisted Intervention (MICCAI), 11070:629-637
%      http://pages.cs.wisc.edu/~mchung/papers/chung.2018.MICCAI.pdf
%
% [3] Chung, M.K. Lee, H., Gritsenko, A., DiChristofano, A., Pluta, D. 
% Ombao, H. Solo, V. Topological Brain Network Distances, ArXiv 1809.03878
% http://arxiv.org/abs/1809.03878
%
% If you are using this code, please reference one of the above paper. 
%
% (C) 2017- Moo K. Chung, Hyekyung Lee                           
%      University of Wisconsin-Madison
%      Seuoul National University
%      mkchung@wisc.edu
%
%     
%    
%
% 2017 December 18. errors fixed
% 2018 Jun.  16.  nargin added
% 2018 Jul.  28.  Betti-1 number computation added. This part is based on work with 
%                 Alex DiChristofano and Andrey Gritsenko
% 2018 Aug.  11.  con2adj error fixed. code simplified using built-in MATLAB function
% 2018 Sept. 12., 2019 July  05. Additional documentation added


if nargin<=1
    %if threshold is not given, thresholds are automatically set between
    %the minimum and maximum of edge weghts at 0.01 increment.
    maxC= max(max(C));
    minC = min(min(C));
    thresholds= minC:0.01:maxC;
end


%-------------------------------
% It requries computing beta0 first. Then computes beta1 using beta0
% beta1 is a function of beta0. Note 
% Euler characteristic = beta0 - beta1 = # of nodes - # of edges. 
% Thus, beta1 = beta0 - # of nodes + # of edges


beta0 =[];
biggest0=[];

beta1 =[];
biggest1=[];

n_nodes = size(C,1);

%C=abs(C); %removes possible negative values in the connectivity matrix. 

for rho=thresholds  %this range needs to be changed depending on the value of C
    %computest Beti-0
    %computes adjacency matrix    
    adj = sparse(C>rho); %introduces diagonal entries
    adj = adj - diag(diag(adj)); %removes diagonal entries
    
    % we don't want any nodes showing up as separate connected components
    adj( ~any(adj,2), : ) = [];  %rows
    adj( :, ~any(adj,1) ) = [];  %columns
    
    n_nodes = size(adj,1);
    
    %[n_components,S] = conncomp(adj); %faster routine
    [n_components,S] = graphconncomp(adj); %built-in MATLAB function is slow
    %n_components is the number of components
    %S is a vector indicating to which component each node belongs
    beta0=[beta0 n_components]; %Betti_0: the number of connected components
    
    nn = hist(S,[1:n_components]); 
    % nn contains the number of nodes in each connected component 
    biggest0 = [biggest0 max(nn)];

    %computes Beti-1
   % n_edges=sum(sum(adj))/2;
    n_edges = nnz(adj)/2;
    n_cycle = n_components - n_nodes + n_edges;
    beta1=[beta1 n_cycle];
    
   
    %if n_cycle<=0
    %    n_cycle=0; % the number of cycle may go below zero numerically. 
    %end;
    
end

biggest1=[]; %This is not yet implemnented. 

