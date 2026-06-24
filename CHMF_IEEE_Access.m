clear all
close all
clc
rng(13); 

base_dir = 'D:\RnD\Frameworks\Datasets\HyperspectralImages';
dataset_name = ["Indian_Pines", "Salinas", "SalinasA", "PaviaC", "PaviaU", "KSC", "Botswana", "Houston13", "Houston18"];

plot_results = 0;
stats = [];
chf_enable = 1; %Enable/disable Cyclic Heterogenous Filtering
for dataset_idx=1:7
    [hsData, gtLabel, rgbImg, numClasses] = load_HSI_dataset(dataset_name(dataset_idx),base_dir);
    [M,N,C] = size(hsData);
    tic
    hsData = PCA_downsample(hsData,round(C*0.7));

    %Ablation study as reported in the paper
    % hsData = PCA_filtering(hsData,round(C*0.6));
    % hsDataFiltered = Spatial_Filtering_Baseline_Gaussian(hsData,chf_enable);
    % hsDataFiltered = Spatial_Filtering_Baseline_Median(hsData,chf_enable);
    % hsDataFiltered = Spatial_Filtering_Baseline_Bilateral(hsData,chf_enable);
    hsDataFiltered = Spatial_Filtering_Baseline_GIF(hsData,chf_enable);
    t_preprocess = toc;

    % hsDataFiltered = PCA_downsample(hsDataFiltered,round(C*0.7));
    for k=1:10
        trainRatio=0.1;
        [TrainAccuracy,TestAccuracy_Overall,TestAccuracy_Average,TestKappa,perClassAcc,t_classify] = trainSVM(hsData,hsDataFiltered, gtLabel, rgbImg, numClasses, trainRatio, plot_results);
        fprintf('%s:Spatial Filtering -Training(%.2f) Accuracy: %.4f,Test(%.2f) Accuracy: %.4f\n', dataset_name(dataset_idx),trainRatio*100,TrainAccuracy,(1-trainRatio)*100,TestAccuracy_Overall);
        stats(dataset_idx,k,1) = TrainAccuracy;
        stats(dataset_idx,k,2) = TestAccuracy_Overall;
        stats(dataset_idx,k,3) = TestAccuracy_Average;
        stats(dataset_idx,k,4) = TestKappa;
        stats(dataset_idx,k,5) = 1000*(t_preprocess + t_classify)/(M*N*(1-trainRatio)); %milliseconds per sample
    end
    % keyboard
end
dataset_stats = zeros(dataset_idx,10);
for d = 1:dataset_idx
    dataset_stats(d,1) = mean(stats(d,:,1)); 
    dataset_stats(d,2) = var(stats(d,:,1)); 
    dataset_stats(d,3) = mean(stats(d,:,2)); 
    dataset_stats(d,4) = var(stats(d,:,2)); 
    dataset_stats(d,5) = mean(stats(d,:,3)); 
    dataset_stats(d,6) = var(stats(d,:,3)); 
    dataset_stats(d,7) = mean(stats(d,:,4)); 
    dataset_stats(d,8) = var(stats(d,:,4));
    dataset_stats(d,9) = mean(stats(d,:,5)); 
    dataset_stats(d,10) = var(stats(d,:,5));
end

function hsDataFiltered = Spatial_Filtering_Baseline_GIF(hsData,chf_enable)
[M,N,C] = size(hsData);
[pca1, mnf1] =getPCA_MNF(hsData);
mnf1 = abs(mnf1); %if PCA-based spectral smoothing is done, mnf1 could have tiny imaginary parts!
% mnf_matlab = hypermnf(hsData,20);
% mnf1 = mnf_matlab(:,:,1);
guide = mnf1;
% guide = pca1;

hsDataFiltered = zeros(size(hsData));
for band = 1:C
    bandImage = hsData(:,:,band); 
    if(~chf_enable)
        bandImageFiltered = imguidedfilter(bandImage, guide, 'NeighborhoodSize',31, 'DegreeOfSmoothing',0.75e7);       
    else
        if(rem(band,5)==1)
            bandImageFiltered = imguidedfilter(bandImage, guide, 'NeighborhoodSize',3, 'DegreeOfSmoothing',0.75e7);
        elseif(rem(band,5)==2)
            bandImageFiltered = imguidedfilter(bandImage, guide, 'NeighborhoodSize',5, 'DegreeOfSmoothing',0.75e7);
        elseif(rem(band,5)==3)
            bandImageFiltered = imguidedfilter(bandImage, guide, 'NeighborhoodSize',13, 'DegreeOfSmoothing',0.75e7);
        elseif(rem(band,5)==4)
            bandImageFiltered = imguidedfilter(bandImage, guide, 'NeighborhoodSize',21, 'DegreeOfSmoothing',0.75e7);
        else
            bandImageFiltered = imguidedfilter(bandImage, guide, 'NeighborhoodSize',31, 'DegreeOfSmoothing',0.75e7);
        end
    end

    bandImageGray = mat2gray(bandImageFiltered);
    hsDataFiltered(:,:,band) = uint8(bandImageGray*255);
    % subplot(1,2,1)
    % imshow(mat2gray(bandImage))
    % subplot(1,2,2)
    % imshow(mat2gray(bandImageFiltered))  
    % pause
end
end

function hsDataFiltered = Spatial_Filtering_Baseline_Bilateral(hsData,chf_enable)
[M,N,C] = size(hsData);
hsDataFiltered = zeros(size(hsData));
for band = 1:C
    bandImage = hsData(:,:,band); 
    if(~chf_enable)
        bandImageFiltered = imbilatfilt(bandImage,0.75e7,10);  
    else
        if(rem(band,5)==1)
            bandImageFiltered = imbilatfilt(bandImage,0.75e7,3);
        elseif(rem(band,5)==2)
            bandImageFiltered = imbilatfilt(bandImage,0.75e7,4);
        elseif(rem(band,5)==3)
            bandImageFiltered = imbilatfilt(bandImage,0.75e7,7);
        elseif(rem(band,5)==4)
            bandImageFiltered = imbilatfilt(bandImage,0.75e7,8);
        else
            bandImageFiltered = imbilatfilt(bandImage,0.75e7,9);
        end
    end

    bandImageGray = mat2gray(bandImageFiltered);
    hsDataFiltered(:,:,band) = uint8(bandImageGray*255);
    % subplot(1,2,1)
    % imshow(mat2gray(bandImage))
    % subplot(1,2,2)
    % imshow(mat2gray(bandImageFiltered))  
    % pause
end
end


function hsDataFiltered = Spatial_Filtering_Baseline_Median(hsData,chf_enable)
[M,N,C] = size(hsData);
hsDataFiltered = zeros(size(hsData));
for band = 1:C
    bandImage = hsData(:,:,band); 
    if(~chf_enable)
        bandImageFiltered = medfilt2(bandImage,[7 7],"symmetric");  
    else
        if(rem(band,5)==1)
            bandImageFiltered = medfilt2(bandImage,[7 7],"symmetric");
        elseif(rem(band,5)==2)
            bandImageFiltered = medfilt2(bandImage,[9 9],"symmetric");
        elseif(rem(band,5)==3)
            bandImageFiltered = medfilt2(bandImage,[11 11],"symmetric");
        elseif(rem(band,5)==4)
            bandImageFiltered = medfilt2(bandImage,[13 13],"symmetric");
        else
            bandImageFiltered = medfilt2(bandImage,[15 15],"symmetric");
        end
    end

    bandImageGray = mat2gray(bandImageFiltered);
    hsDataFiltered(:,:,band) = uint8(bandImageGray*255);
    % subplot(1,2,1)
    % imshow(mat2gray(bandImage))
    % subplot(1,2,2)
    % imshow(mat2gray(bandImageFiltered))  
    % pause
end
end

function hsDataFiltered = Spatial_Filtering_Baseline_Gaussian(hsData,chf_enable)
[M,N,C] = size(hsData);
hsDataFiltered = zeros(size(hsData));
for band = 1:C
    bandImage = hsData(:,:,band); 
    if(~chf_enable)
        bandImageFiltered = imgaussfilt(bandImage,1.0); 
    else
        if(rem(band,5)==1)
            bandImageFiltered = imgaussfilt(bandImage,2.0);
        elseif(rem(band,5)==2)
            bandImageFiltered = imgaussfilt(bandImage,4.0);
        elseif(rem(band,5)==3)
            bandImageFiltered = imgaussfilt(bandImage,5.0);
        elseif(rem(band,5)==4)
            bandImageFiltered = imgaussfilt(bandImage,7.0);
        else
            bandImageFiltered = imgaussfilt(bandImage,9.0);
        end
    end

    bandImageGray = mat2gray(bandImageFiltered);
    hsDataFiltered(:,:,band) = uint8(bandImageGray*255);
    % subplot(1,2,1)
    % imshow(mat2gray(bandImage))
    % subplot(1,2,2)
    % imshow(mat2gray(bandImageFiltered))  
    % pause
end
end



