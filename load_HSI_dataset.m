function [hsData, gtLabel, rgbImg, numClasses] = load_HSI_dataset(dataset_name,base_dir)

    switch dataset_name
        case 'Indian_Pines'
            hcube = hypercube("indian_pines.dat");
            gtLabel = load(fullfile(base_dir,'indian_pines_gt.mat'));
            gtLabel = gtLabel.indian_pines_gt;
            band = [104:108 150:163 220]; % Water absorption bands
            newhcube = removeBands(hcube,BandNumber=band);
            numClasses = 16;
            rgbImg = colorize(newhcube,method="rgb");
            % hsData = newhcube.DataCube;
            hsData = gather(newhcube);
            % hyperspectralViewer(hcube);

        case 'Salinas'
            gtLabel = load(fullfile(base_dir,'Salinas_gt.mat'));
            gtLabel = gtLabel.salinas_gt;
            % load(fullfile(base_dir,'Salinas.mat'));
            load(fullfile(base_dir,'Salinas_corrected.mat'));
            % keyboard
            hsData = salinas_corrected;
            r = hsData(:,:,27);
            g = hsData(:,:,16);
            b = hsData(:,:,8);
            rgb = cat(3,r,g,b);
            rgbImg = uint8(255*rgb/8698);
            numClasses = 16;
    
        case 'SalinasA'
            gtLabel = load(fullfile(base_dir,'SalinasA_gt.mat'));
            % keyboard
            gtLabel = gtLabel.salinasA_gt;
            load(fullfile(base_dir,'SalinasA.mat'));
            hsData = salinasA;
            r = hsData(:,:,27);
            g = hsData(:,:,16);
            b = hsData(:,:,8);
            rgb = cat(3,r,g,b);
            rgbImg = uint8(255*rgb/8698);
            numClasses = 16; 

        case 'Pavia'
            load(fullfile(base_dir,'Pavia.mat'));
            hsData = pavia;
            r = hsData(:,:,27);
            g = hsData(:,:,16);
            b = hsData(:,:,8);
            rgb = cat(3,r,g,b);
            rgbImg = uint8(255*rgb/8698);
            gtLabel = load(fullfile(base_dir,'pavia_gt.mat'));
            gtLabel = gtLabel.pavia_gt;
            numClasses = 9;
    
        case 'PaviaU'
            % hcube = hypercube("paviaU.dat");
            % gtLabel = load(fullfile(base_dir,'paviaU_gt.mat'));
            % gtLabel = gtLabel.paviaU_gt;
            % newhcube = hcube;
            % numClasses = 9;
            % rgbImg = colorize(newhcube,method="rgb");
            % % hsData = newhcube.DataCube;
            % % hsData = newhcube.DataCube;
            % hsData = gather(newhcube);
            load(fullfile(base_dir,'PaviaU.mat'));
            % keyboard
            hsData = paviaU;
            r = hsData(:,:,27);
            g = hsData(:,:,16);
            b = hsData(:,:,8);
            rgb = cat(3,r,g,b);
            rgbImg = uint8(255*rgb/8698);
            gtLabel = load(fullfile(base_dir,'PaviaU_gt.mat'));
            gtLabel = gtLabel.paviaU_gt;
            numClasses = 9;            
    
        case 'KSC'
            load(fullfile(base_dir,'KSC.mat'));
            hsData = KSC;
            r = hsData(:,:,27);
            g = hsData(:,:,16);
            b = hsData(:,:,8);
            rgb = cat(3,r,g,b);
            rgbImg = uint8(255*rgb/8698);
            gtLabel = load(fullfile(base_dir,'KSC_gt.mat'));
            gtLabel = gtLabel.KSC_gt;
            numClasses = 13;    
    
        case 'Botswana'
            load(fullfile(base_dir,'Botswana.mat'));
            hsData = Botswana;
            r = hsData(:,:,27);
            g = hsData(:,:,16);
            b = hsData(:,:,8);
            rgb = cat(3,r,g,b);
            rgbImg = uint8(255*rgb/8698);
            gtLabel = load(fullfile(base_dir,'Botswana_gt.mat'));
            gtLabel = gtLabel.Botswana_gt;
            numClasses = 14;        
       end
    
end