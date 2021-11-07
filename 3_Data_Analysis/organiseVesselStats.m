function organiseVesselStats(tumour_list, main_directory)

    dir_check = ismember({tumour_list(:).name},{'TDA_Summary'});
    output_path = strcat(main_directory, '/TDA_Summary');
    if sum(dir_check) == 1
      idx = find(dir_check == 1);
      tumour_list(idx) = [];
    else 
        mkdir(output_path)
    end

    maxval = 0;
    for i=1:size(tumour_list,1)

        subfolder_list = dir(strcat(main_directory, '/', tumour_list(i).name, '/skeletons'));
        subfolder_list = subfolder_list([subfolder_list(:).isdir]);
        subfolder_list = subfolder_list(~ismember({subfolder_list(:).name},{'.','..'}));
        if maxval < size(subfolder_list,1)
           maxval = size(subfolder_list,1); 
        end

    end

    n = maxval + 1;
    m = size(tumour_list,1);

    data = zeros(size(tumour_list,1),17,n);
    bettiMaxComponentLoops = cell(m,n);
    bettiMaxComponentCCs = cell(m,n);
    bettiFNLoops = cell(m,n);
    bettiFNCCs = cell(m,n);

    sizeMaxComponentNodes = cell(m,n);
    sizeMaxComponentEdges = cell(m,n);
    sizeFNNodes = cell(m,n);
    sizeFNEdges = cell(m,n);

    lengthFN = cell(m,n);
    lengthFNstd = cell(m,n);
    diameterFN = cell(m,n);
    diameterFNstd = cell(m,n);

    clrFN = cell(m,n);
    clrFNstd = cell(m,n);
    soamFN = cell(m,n);
    soamFNstd = cell(m,n);

    mean_void = cell(m,n);
    median_void = cell(m,n);
    std_void = cell(m,n);

    void = false;

    dataorder = cell(m,n);

    % 
    % for j = 2:n
    %     bettiMaxComponentLoops{1,j} = subfolder_list(j-1).name;
    %     bettiMaxComponentCCs{1,j} = subfolder_list(j-1).name;
    %     bettiFNLoops{1,j} = subfolder_list(j-1).name;
    %     bettiFNCCs{1,j} = subfolder_list(j-1).name;
    %     sizeMaxComponentNodes{1,j} = subfolder_list(j-1).name;
    %     sizeMaxComponentEdges{1,j} = subfolder_list(j-1).name;
    %     sizeFNNodes{1,j} = subfolder_list(j-1).name;
    %     sizeFNEdges{1,j} = subfolder_list(j-1).name;
    %     lengthFN{1,j} = subfolder_list(j-1).name;
    %     lengthFNstd{1,j} = subfolder_list(j-1).name;
    %     diameterFN{1,j} = subfolder_list(j-1).name;
    %     diameterFNstd{1,j} = subfolder_list(j-1).name;
    %     clrFN{1,j} = subfolder_list(j-1).name;
    %     clrFNstd{1,j} = subfolder_list(j-1).name;
    %     soamFN{1,j} = subfolder_list(j-1).name;
    %     soamFNstd{1,j} = subfolder_list(j-1).name;
    %     mean_void{1,j} = subfolder_list(j-1).name;
    %     median_void{1,j} = subfolder_list(j-1).name;
    %     std_void{1,j} = subfolder_list(j-1).name;
    % end

    [status, list] = system( main_directory );
    result = textscan( main_directory, '%s', 'delimiter', '\n' );
    fileList = result{1};

    for i = 1:m

        subfolder_list = dir(strcat(main_directory, '/', tumour_list(i).name, '/Skeletons'));
        subfolder_list = subfolder_list([subfolder_list(:).isdir]);
        subfolder_list = subfolder_list(~ismember({subfolder_list(:).name},{'.','..'}));

        figure_path = strcat(tumour_list(i).folder, '/', tumour_list(i).name, '/TDA/');
        void_path = strcat(tumour_list(i).folder, '/', tumour_list(i).name, '/','GUDHI_Data/');

        load(strcat(figure_path,'data_log.mat'));   

        bettiMaxComponentLoops{i,1} = tumour_list(i).name;
        bettiMaxComponentCCs{i,1} = tumour_list(i).name;
        bettiFNLoops{i,1} = tumour_list(i).name;
        bettiFNCCs{i,1} = tumour_list(i).name;
        sizeMaxComponentNodes{i,1} = tumour_list(i).name;
        sizeMaxComponentEdges{i,1} = tumour_list(i).name;
        sizeFNNodes{i,1} = tumour_list(i).name;
        sizeFNEdges{i,1} = tumour_list(i).name;
        lengthFN{i,1} = tumour_list(i).name;
        lengthFNstd{i,1} = tumour_list(i).name;
        diameterFN{i,1} = tumour_list(i).name;
        diameterFNstd{i,1} = tumour_list(i).name;
        clrFN{i,1} = tumour_list(i).name;
        clrFNstd{i,1} = tumour_list(i).name;
        soamFN{i,1} = tumour_list(i).name;
        soamFNstd{i,1} = tumour_list(i).name;
        dataorder{i,1} = tumour_list(i).name;


        for j = 2:(1+size(subfolder_list,1))
            bettiMaxComponentLoops{i,j} = csvmat(1,j-1);
            bettiMaxComponentCCs{i,j} = csvmat(2,j-1);
            bettiFNLoops{i,j} = csvmat(3,j-1);
            bettiFNCCs{i,j} = csvmat(4,j-1);
            sizeMaxComponentNodes{i,j} = csvmat(5,j-1);
            sizeMaxComponentEdges{i,j} = csvmat(6,j-1);
            sizeFNNodes{i,j} = csvmat(7,j-1);
            sizeFNEdges{i,j} = csvmat(8,j-1);
            lengthFN{i,j} = csvmat(9,j-1);
            lengthFNstd{i,j} = csvmat(10,j-1);
            diameterFN{i,j} = csvmat(11,j-1);
            diameterFNstd{i,j} = csvmat(12,j-1);
            clrFN{i,j} = csvmat(13,j-1);
            clrFNstd{i,j} = csvmat(14,j-1);
            soamFN{i,j} = csvmat(15,j-1);
            soamFNstd{i,j} = csvmat(16,j-1);
            dataorder{i,j} = subfolder_list(j-1).name;
        end

        writecell(bettiMaxComponentLoops,strcat(output_path,'/','loops_MaxComponent_EdgeNorm.csv'))
        writecell(bettiMaxComponentCCs,strcat(output_path,'/','CC_MaxComponent_EdgeNorm.csv'))
        writecell(bettiFNLoops,strcat(output_path,'/','loops_FullNetwork_EdgeNorm.csv'))
        writecell(bettiFNCCs,strcat(output_path,'/','CC_FullNetwork_EdgeNorm.csv'))
        writecell(sizeMaxComponentNodes,strcat(output_path,'/','sizeNodes_MaxComponent.csv'))
        writecell(sizeMaxComponentEdges,strcat(output_path,'/','sizeEdges_MaxComponent.csv'))
        writecell(sizeFNNodes,strcat(output_path,'/','sizeNodes_FullNetwork.csv'))
        writecell(sizeFNEdges,strcat(output_path,'/','sizeEdges_FullNetwork.csv'))
        writecell(lengthFN,strcat(output_path,'/','lengths_FullNetwork.csv'))
        writecell(lengthFNstd,strcat(output_path,'/','lengths_MaxComponent.csv'))
        writecell(diameterFN,strcat(output_path,'/','diameters_FullNetwork.csv'))
        writecell(diameterFNstd,strcat(output_path,'/','diameters_MaxComponent.csv'))
        writecell(clrFN,strcat(output_path,'/','clr_FullNetwork.csv'))
        writecell(clrFNstd,strcat(output_path,'/','clr_MaxComponent.csv'))
        writecell(soamFN,strcat(output_path,'/','soam_FullNetwork.csv'))
        writecell(soamFNstd,strcat(output_path,'/','soam_MaxComponent.csv'))
        writecell(dataorder,strcat(output_path,'/','dataReadIn.csv'))

        if void
            mean_void{i,1} = tumour_list(i).name;
            median_void{i,1} = tumour_list(i).name;
            median_void{i,1} = tumour_list(i).name;

            for j = 2:n
                load(strcat(void_path,'mean_SoV.mat'));
                mean_void{i,j} = mean_SoV(1,j-1);
                load(strcat(void_path,'median_SoV.mat'));
                median_void{i,j} = median_size_of_voids(1,j-1);
                load(strcat(void_path,'std_SoV.mat'));
                std_void{i,j} = std_void(1,j-1);
            end

            writecell(mean_void,strcat(output_path,'/','mean_SoV.csv'))
            writecell(median_void,strcat(output_path,'/','median_SoV.csv'))
            writecell(std_void,strcat(output_path,'/','std_SoV.csv'))
        end

    end

    calculateBloodVolume(tumour_list, n-1, output_path);

end
