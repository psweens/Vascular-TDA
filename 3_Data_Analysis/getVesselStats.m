%% Obtain initial vessel stats (TDA and non-TDA) for Cambridge data based on adjacency matrices

% Output: Number of Loops, number of connected components, vessel lengths,
% vessel diameters, clr, SOAM

% Bernadette Stolz
% 24.7.2020

function getVesselStats(tumour_list)

    name = 'Segmentation Methods';

    %parpool(28) %adjust to machine

    for tumour_idx = 1:size(tumour_list,1)

        % Data directory
        figure_path = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/TDA/');
        if ~exist(figure_path)
               mkdir(figure_path)
        end
        folder_name = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/Skeletons');


        file_list = dir(folder_name);
        file_list = file_list([file_list(:).isdir]);
        file_list = file_list(~ismember({file_list(:).name},{'.','..'}));
        save_list = file_list;

        folders = string(size(file_list));
        for i = 1:size(file_list,1)
            file_list(i).name = strcat(file_list(i).folder, '/', file_list(i).name, '/Components/');
            save_list(i).name = strcat(save_list(i).folder, '/', save_list(i).name, '/Analysis/');
            folders(i) = file_list(i).name;
        end

        for folder_index = 1:size(file_list,1)

            initial_path = file_list(folder_index).name;

            % Make analysis folder
            mkdir(save_list(folder_index).name);

            max_component_size = 0;

            max_component_number = 0;

            load([initial_path,'AdjacencyMatrix.mat']) 
            load([initial_path,'LargestComponentAdjacencyMatrix.mat']) 

            load([initial_path,'AdjacencyMatrixLength.mat']) 
            load([initial_path,'LargestComponentAdjacencyMatrixLength.mat'])  

            load([initial_path,'AdjacencyMatrixDiameter.mat']) 
            load([initial_path,'LargestComponentAdjacencyMatrixDiameter.mat'])  

            load([initial_path,'AdjacencyMatrixCLR.mat']) 
            load([initial_path,'LargestComponentAdjacencyMatrixCLR.mat'])  

            load([initial_path,'AdjacencyMatrixSOAM.mat']) 
            load([initial_path,'LargestComponentAdjacencyMatrixSOAM.mat'])  


            nodes_max_component(folder_index) = size(largest_component_adjacency_matrix,1);
            edges_max_component(folder_index) = nnz(largest_component_adjacency_matrix)/2;

            length_max_component(folder_index) = mean (nonzeros(largest_component_adjacency_matrix_length));
            diameter_max_component(folder_index) = mean (nonzeros(largest_component_adjacency_matrix_diameter));
            clr_max_component(folder_index) = mean (nonzeros(largest_component_adjacency_matrix_clr));
            SOAM_max_component(folder_index) = mean (nonzeros(largest_component_adjacency_matrix_soam));

            nodes_full_network(folder_index) = size(full_adjacency_matrix,1);
            edges_full_network(folder_index) = nnz(full_adjacency_matrix)/2;


            length_full_network(folder_index) = mean(nonzeros(full_adjacency_matrix_length));
            diameter_full_network(folder_index) = mean(nonzeros(full_adjacency_matrix_diameter));

            clr_full_network(folder_index) = mean(nonzeros(full_adjacency_matrix_clr));
            SOAM_full_network(folder_index) = mean(nonzeros(full_adjacency_matrix_soam));

            std_length_full_network(folder_index) = std(nonzeros(full_adjacency_matrix_length));
            std_diameter_full_network(folder_index) = std(nonzeros(full_adjacency_matrix_diameter));
            std_clr_full_network(folder_index) = std(nonzeros(full_adjacency_matrix_clr));
            std_SOAM_full_network(folder_index) = std(nonzeros(full_adjacency_matrix_soam));

            thresholds= 0;
            [beta0_max_component(folder_index), beta1_max_component(folder_index), biggest0, biggest1] = PH_betti2_modB(largest_component_adjacency_matrix, thresholds);
            [beta0_full_network(folder_index), beta1_full_network(folder_index), biggest0, biggest1] = PH_betti2_modB(full_adjacency_matrix, thresholds);

            % We normalise to reduce the effect of different network sizes

            beta0_max_component(folder_index) = beta0_max_component(folder_index)/edges_max_component(folder_index);
            beta1_max_component(folder_index) = beta1_max_component(folder_index)/edges_max_component(folder_index);

            beta0_full_network(folder_index) = beta0_full_network(folder_index)/edges_full_network(folder_index);
            beta1_full_network(folder_index) = beta1_full_network(folder_index)/edges_full_network(folder_index);


            clear largest_component_adjacency_matrix
            clear full_adjacency_matrix

        end

        h1 = figure('Name','Biological')
        plot(0:1:size(file_list,1)-1,beta1_max_component(1:size(file_list,1)),'b','LineWidth',2)
        hold on
        plot(0:1:size(file_list,1)-1,beta0_max_component(1:size(file_list,1)),'r','LineWidth',2)
        legend('Number of loops','Number of connected components','Fontsize',20)
        set(gca,'FontSize',18)
        xlabel('Mask number')
        ylabel('Betti numbers/number of edges')
        title([name,': Betti numbers, largest component'],'Fontsize',20)

        image_title1 = [char(figure_path) 'BettiNumbersMaxComponent.fig'];
        image_title2 = [char(figure_path) 'BettiNumbersMaxComponent.pdf'];


            set(h1,'Units','Inches');
            pos = get(h1,'Position');
            set(h1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
            print(h1,image_title2,'-dpdf','-r0')

            saveas(h1,image_title1)

            csvmat(1,:) = beta1_max_component(1:size(file_list,1));
            csvmat(2,:) = beta0_max_component(1:size(file_list,1));

        %close(h1)



        h2 = figure('Name','Biological')
        plot(0:1:size(file_list,1)-1,beta1_full_network(1:size(file_list,1)),'b','LineWidth',2)
        hold on
        plot(0:1:size(file_list,1)-1,beta0_full_network(1:size(file_list,1)),'r','LineWidth',2)
        legend('Number of loops','Number of connected components','Fontsize',20)
        set(gca,'FontSize',18)
        xlabel('Mask number')
        ylabel('Betti numbers/number of edges')
        title([name,': Betti numbers, full network'],'Fontsize',20)

        image_title3 = [char(figure_path) 'BettiNumbersFullNetwork.fig'];
        image_title4 = [char(figure_path) 'BettiNumbersFullNetwork.pdf'];


            set(h2,'Units','Inches');
            pos = get(h2,'Position');
            set(h2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
            print(h2,image_title4,'-dpdf','-r0')

            saveas(h2,image_title3)

            csvmat(3,:) = beta1_full_network(1:size(file_list,1));
            csvmat(4,:) = beta0_full_network(1:size(file_list,1));

            close(h2)

        h3 = figure('Name','Biological')
        plot(0:1:size(file_list,1)-1,nodes_max_component(1:size(file_list,1)),'b','LineWidth',2)
        hold on
        plot(0:1:size(file_list,1)-1,edges_max_component(1:size(file_list,1)),'r','LineWidth',2)
        legend('Nodes','Edges','Fontsize',20)
        set(gca,'FontSize',18)
        xlabel('Mask number')
        title([name,': Network summaries, largest component'],'Fontsize',20)

        image_title5 = [char(figure_path) 'SizeMaxComponent.fig'];
        image_title6 = [char(figure_path) 'SizeMaxComponent.pdf'];


            set(h3,'Units','Inches');
            pos = get(h3,'Position');
            set(h3,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
            print(h3,image_title6,'-dpdf','-r0')

            saveas(h3,image_title5)

            csvmat(5,:) = nodes_max_component(1:size(file_list,1));
            csvmat(6,:) = edges_max_component(1:size(file_list,1));

            close(h3)


        h4 = figure('Name','Biological')
        plot(0:1:size(file_list,1)-1,nodes_full_network(1:size(file_list,1)),'b','LineWidth',2)
        hold on
        plot(0:1:size(file_list,1)-1,edges_full_network(1:size(file_list,1)),'r','LineWidth',2)
        legend('Nodes','Edges','Fontsize',20)
        set(gca,'FontSize',18)
        xlabel('Mask number')
        title([name,': Network summaries, full network'],'Fontsize',20)


        image_title7 = [char(figure_path) 'SizeFullNetwork.fig'];
        image_title8 = [char(figure_path) 'SizeFullNetwork.pdf'];


            set(h4,'Units','Inches');
            pos = get(h4,'Position');
            set(h4,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
            print(h4,image_title8,'-dpdf','-r0')

            saveas(h4,image_title7)

            csvmat(7,:) = nodes_full_network(1:size(file_list,1));
            csvmat(8,:) = edges_full_network(1:size(file_list,1));

            close(h4)



        h5 = figure('Name','Biological')
        errorbar(0:1:size(file_list,1)-1,length_full_network(1:size(file_list,1)), std_length_full_network(1:size(file_list,1)),'b','LineWidth',2)
        set(gca,'FontSize',18)
        xlabel('Mask number')
        ylabel('Length')
        title([name,': Lengths, full network'],'Fontsize',20)


        image_title9 = [char(figure_path) 'LengthFullNetwork.fig'];
        image_title10 = [char(figure_path) 'LengthFullNetwork.pdf'];


            set(h5,'Units','Inches');
            pos = get(h5,'Position');
            set(h5,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
            print(h5,image_title10,'-dpdf','-r0')

            saveas(h5,image_title9)

            csvmat(9,:) = length_full_network(1:size(file_list,1));
            csvmat(10,:) = length_max_component(1:size(file_list,1));

            close(h5)


        h6 = figure('Name','Biological')
        errorbar(0:1:size(file_list,1)-1,diameter_full_network(1:size(file_list,1)), std_diameter_full_network(1:size(file_list,1)),'b','LineWidth',2)
        set(gca,'FontSize',18)
        xlabel('Mask number')
        ylabel('Diameter')
        title([name,': Diameters, full network'],'Fontsize',20)


        image_title11 = [char(figure_path) 'DiametersFullNetwork.fig'];
        image_title12 = [char(figure_path) 'DiametersFullNetwork.pdf'];


            set(h6,'Units','Inches');
            pos = get(h6,'Position');
            set(h6,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
            print(h6,image_title12,'-dpdf','-r0')

            saveas(h6,image_title11)

            csvmat(11,:) = diameter_full_network(1:size(file_list,1));
            csvmat(12,:) = diameter_max_component(1:size(file_list,1));

            close(h6)


        h7 = figure('Name','Biological')
        errorbar(0:1:size(file_list,1)-1,clr_full_network(1:size(file_list,1)), std_clr_full_network(1:size(file_list,1)),'b','LineWidth',2)
        set(gca,'FontSize',18)
        xlabel('Mask number')
        ylabel('Chord-length-ratio')
        title([name,': CLR, full network'],'Fontsize',20)


        image_title13 = [char(figure_path) 'CLRFullNetwork.fig'];
        image_title14 = [char(figure_path) 'CLRFullNetwork.pdf'];


            set(h7,'Units','Inches');
            pos = get(h7,'Position');
            set(h7,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
            print(h7,image_title14,'-dpdf','-r0')

            saveas(h7,image_title13)

            csvmat(13,:) = clr_full_network(1:size(file_list,1));
            csvmat(14,:) = clr_max_component(1:size(file_list,1));

            close(h7)


        h9 = figure('Name','Biological')
        errorbar(0:1:size(file_list,1)-1,SOAM_full_network(1:size(file_list,1)),std_SOAM_full_network(1:size(file_list,1)),'b','LineWidth',2)
        set(gca,'FontSize',18)
        xlabel('Mask number')
        ylabel('Sum-of-angles-metric')
        title([name,': SOAM, full network'],'Fontsize',20)


        image_title17 = [char(figure_path) 'SOAMFullNetwork.fig'];
        image_title18 = [char(figure_path) 'SOAMFullNetwork.pdf'];


            set(h9,'Units','Inches');
            pos = get(h9,'Position');
            set(h9,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
            print(h9,image_title18,'-dpdf','-r0')

            saveas(h9,image_title17)

            csvmat(15,:) = SOAM_full_network(1:size(file_list,1)) ./ length_full_network(1:size(file_list,1));
            csvmat(16,:) = SOAM_max_component(1:size(file_list,1))./ length_full_network(1:size(file_list,1));

            save(strcat(figure_path,'data_log'),'csvmat');

            close(h9)

    end

end
