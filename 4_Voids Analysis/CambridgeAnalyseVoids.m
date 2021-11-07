%% Analyse the size of voids in Cambridge data set
% Bernadette Stolz
% 2.8.2020

close all
clear all
clc

main_directory = '/media/sweene01/SSD/Lina/ilastik_medfilt_Test_retest/ilastik_medfilt_Test_retest_Lina'; 
tumour_list = dir(main_directory);
tumour_list = tumour_list([tumour_list(:).isdir]);
tumour_list = tumour_list(~ismember({tumour_list(:).name},{'.','..'}));

folders=[];
for tumour_idx = 1:size(tumour_list,1)
    
    initial_path = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/skeletons');

    folders = dir(initial_path)
    folders = folders([folders(:).isdir]);
    folders = folders(~ismember({folders(:).name},{'.','..'}));
    folders={folders.name};
    folders=sort(folders);
    
    alpha_complex_path = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/','GUDHI_Data/Output/');

    mean_SoV = zeros(size(1,size(folders,2)));
    for data_file_number = 1:size(folders,2)

       % category = categories(data_file_number);
        %tumour = tumours(data_file_number);
        %folder = folders(data_file_number);

        folder = char(folders(data_file_number));%.name;
        tumour = char(folders(data_file_number));%.name;
        %[pathstr, tumour, ext] = fileparts(tumour);
        tumour = replace(tumour,'Simple Segmentation','Simple_Segmentation'); 

        
        if isfile([alpha_complex_path char(tumour) '_2.txt'])
            filename =  [alpha_complex_path char(tumour) '_2.txt'];
        elseif isfile([alpha_complex_path char(tumour) '_1.txt'])
            filename =  [alpha_complex_path char(tumour) '_1.txt'];
        elseif isfile([alpha_complex_path char(tumour) '_0.txt'])
            filename =  [alpha_complex_path char(tumour) '_0.txt'];
        end

        alpha = load(filename);

        size_of_voids = alpha(:,2) - alpha(:,1);
        
        mean_SoV(1,data_file_number) = mean(size_of_voids);

        % This can be worth playing around with: Mean, median, maximum etc

    %     mean_size_of_voids(data_file_number) = mean(size_of_voids);
    %     
        std_size_of_voids(data_file_number) = std(size_of_voids);

        median_size_of_voids(data_file_number) = median(size_of_voids);

        q1_size_of_voids(data_file_number) = quantile(size_of_voids,0.25);
        q2_size_of_voids(data_file_number) = quantile(size_of_voids,0.75);


        clear alpha
        clear size_of_voids

    end


    h1 = figure('Name','Biological')
    errorbar(1:1:4,median_size_of_voids(1:4), median_size_of_voids(1:4)-...
        q1_size_of_voids(1:4),median_size_of_voids(1:4)+q2_size_of_voids(1:4),'b','LineWidth',2)
    set(gca,'FontSize',18)
    xlabel('Mask number')
    ylabel('Size of voids')
    title('Test retest: Voids','Fontsize',20)

    image_title1 = 'Test_retest_Voids.fig';
    image_title2 = 'Test_retest_Voids.pdf';

        set(h1,'Units','Inches');
        pos = get(h1,'Position');
        set(h1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
        print(h1,image_title2,'-dpdf','-r0')

        saveas(h1,image_title1)



    h2 = figure('Name','Biological')
    errorbar(1:1:4,median_size_of_voids(1:4), median_size_of_voids(1:4)-...
        q1_size_of_voids(1:4),median_size_of_voids(1:4)+q2_size_of_voids(1:4),'b','LineWidth',2)
    set(gca,'FontSize',18)
    xlabel('Mask number')
    ylabel('Size of voids')
    title('Time Series: Voids','Fontsize',20)

    image_title1 = 'Time_Series_Voids.fig';
    image_title2 = 'Time_Series_Voids.pdf';

        set(h2,'Units','Inches');
        pos = get(h2,'Position');
        set(h2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
        print(h2,image_title2,'-dpdf','-r0')

        saveas(h2,image_title1)

    % Mean

    % h1 = figure('Name','Biological')
    % errorbar(1:1:4,mean_size_of_voids(1:4), std_size_of_voids(1:4),'b','LineWidth',2)
    % set(gca,'FontSize',18)
    % xlabel('Mask number')
    % ylabel('Size of voids')
    % title('Test retest: Voids','Fontsize',20)
    % 
    % image_title1 = 'Test_retest_Voids.fig';
    % image_title2 = 'Test_retest_Voids.pdf';
    % 
    %     set(h1,'Units','Inches');
    %     pos = get(h1,'Position');
    %     set(h1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    %     print(h1,image_title2,'-dpdf','-r0')
    % 
    %     saveas(h1,image_title1)




    %     
    %     
    % h2 = figure('Name','Biological')
    % errorbar(1:1:4,mean_size_of_voids(5:8), std_size_of_voids(5:8),'b','LineWidth',2)
    % set(gca,'FontSize',18)
    % xlabel('Mask number')
    % ylabel('Size of voids')
    % title('Time Series: Voids','Fontsize',20)
    % 
    % image_title3 = 'Time_Series_Voids.fig';
    % image_title4 = 'Time_Series_Voids.pdf';
    % 
    %     set(h2,'Units','Inches');
    %     pos = get(h2,'Position');
    %     set(h2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    %     print(h2,image_title2,'-dpdf','-r0')
    % 
    %     saveas(h2,image_title1)
    % 
    
    outputpath = strcat(tumour_list(tumour_idx).folder, '/', tumour_list(tumour_idx).name, '/','GUDHI_Data/');
    save(strcat(outputpath,'mean_SoV'),'mean_SoV');
    save(strcat(outputpath,'median_SoV'),'median_size_of_voids');
    save(strcat(outputpath,'std_SoV'),'std_size_of_voids');
    
    
end
