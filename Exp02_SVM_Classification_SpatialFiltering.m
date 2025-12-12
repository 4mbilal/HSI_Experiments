clear all
close all
clc
rng(13); 

% base_dir = 'D:\RnD\Frameworks\Datasets\HyperspectralImages';
base_dir = 'E:\Datasets\HyperspectralImages';
dataset_name = ["Indian_Pines", "Salinas", "SalinasA", "Pavia", "PaviaU", "KSC", "Botswana"];

plot_results = 0;

for dataset_idx=2:2
    [hsData, gtLabel, rgbImg, numClasses] = load_HSI_dataset(dataset_name(dataset_idx),base_dir);
    % hsData = Spectral_Filtering(hsData);
    hsDataFiltered = Spatial_Filtering(hsData,rgbImg);
    % disp(".")
    for trainRatio=0.1:0.1:0.1
        [TrainAccuracy,TestAccuracy] = trainSVM(hsDataFiltered, gtLabel, rgbImg, numClasses, trainRatio, plot_results);
        fprintf('%s:Spatial Filtering -Training(%.2f) Accuracy: %.4f,Test(%.2f) Accuracy: %.4f\n', dataset_name(dataset_idx),trainRatio*100,TrainAccuracy,(1-trainRatio)*100,TestAccuracy);
    end
end

function hsDataFiltered = Spatial_Filtering(hsData,rgbImg)
% [pca1, mnf1] =getPCA_MNF(hsData);
% mnf1 = mat2gray(mnf1);
% pca1 = mat2gray(pca1);
% subplot(1,2,1)
% imshow(mnf1)
% subplot(1,2,2)
% imshow(pca1)
% keyboard
[M,N,C] = size(hsData);
hsDataFiltered = zeros(size(hsData));
BW = 2;
for band = 1:C
    bandImage = hsData(:,:,band); 
    BW1 = max(band-BW,1);
    BW2 = min(band+BW,C);
    pca_band = hsData(:,:,BW1:BW2);
    [pca1, mnf1] =getPCA_MNF(pca_band);
    % bandImageFiltered = imgaussfilt(bandImage,3.65); 
    % bandImageFiltered = medfilt2(bandImage,[11 11],"symmetric"); 
    % bandImageFiltered = imbilatfilt(bandImage,0.75e7,4.5);
    % bandImageFiltered = imguidedfilter(bandImage, rgb2gray(rgbImg), 'NeighborhoodSize',9, 'DegreeOfSmoothing',0.75e7); %Indian Pines
    % bandImageFiltered = imguidedfilter(bandImage, pca1, 'NeighborhoodSize',9, 'DegreeOfSmoothing',0.75e7); %Indian Pines
    bandImageFiltered = imguidedfilter(bandImage, mnf1, 'NeighborhoodSize',29, 'DegreeOfSmoothing',0.75e7); %Indian Pines
    bandImageGray = mat2gray(bandImageFiltered);
    hsDataFiltered(:,:,band) = uint8(bandImageGray*255);
    % subplot(1,2,1)
    % imshow(mat2gray(bandImage))
    % subplot(1,2,2)
    % imshow(bandImageGray)    
    % pause
end
end

function hsSmooth = Spectral_Filtering(hsData)
% hsData: M x N x C
[M,N,C] = size(hsData);

% Parameters (tune these)
window = 5;      % odd, e.g., 5–11 depending on how smooth you want
poly    = 3;     % polynomial order (1–3 typical)

% Reshape to (C x (M*N)) to apply filter per spectrum
X = reshape(hsData, [], C)';  % C x (M*N)

% Savitzky–Golay smoothing (requires Signal Processing Toolbox)
Xsg = sgolayfilt(X, poly, window, [], 1);  % filter along rows (spectral axis)

hsSmooth = reshape(Xsg', M, N, C);
% keyboard
end


function [TrainAccuracy,TestAccuracy] = trainSVM(hsData, gtLabel, rgbImg, numClasses, trainRatio, plot_results)
[M,N,C] = size(hsData);
DataVector = reshape(hsData,[M*N C]);

gtVector = gtLabel(:);

gtLocs = find(gtVector~=0);
classLabel = gtVector(gtLocs);

cv = cvpartition(classLabel,HoldOut=1-trainRatio);

locTrain = gtLocs(cv.training);
locTest = gtLocs(~cv.training);

% svmMdl = fitcecoc(DataVector(locTrain,:),gtVector(locTrain,:));
tSVM = templateSVM('KernelFunction','linear','BoxConstraint',1.0);
svmMdl = fitcecoc(DataVector(locTrain,:),gtVector(locTrain,:),'Learners',tSVM,'Coding','onevsone');

% svmMdl = fitcecoc(DataVector(locTrain,:),gtVector(locTrain,:),'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus','UseParallel',true,'MaxObjectiveEvaluations',10,'Kfold',2));


[TrainPredictions,~] = predict(svmMdl,DataVector(locTrain,:));
TrainAccuracy = sum(TrainPredictions == gtVector(locTrain))/numel(locTrain);

[TestPredictions,~] = predict(svmMdl,DataVector(locTest,:));
TestAccuracy = sum(TestPredictions == gtVector(locTest))/numel(locTest);

if(plot_results)
    svmPredLabel = gtLabel;
    svmPredLabel(locTest) = TestPredictions;
    
    cmap = parula(numClasses);
    figure
    tiledlayout(1,3,TileSpacing="loose")
    nexttile
    imshow(rgbImg)
    title("RGB Image")
    nexttile
    imshow(gtLabel,cmap)
    title("Ground Truth Map")
    nexttile
    imshow(svmPredLabel,cmap)
    colorbar
    title("SVM Classification Map")
    
    
    fig = figure;
    confusionchart(gtVector(locTest),TestPredictions,ColumnSummary="column-normalized")
    fig_Position = fig.Position;
    fig_Position(3) = fig_Position(3)*1.5;
    fig.Position = fig_Position;
    title("Confusion Matrix: SVM Classification Results")
    
    figure
    Train_labels = gtVector(locTrain,:);
    Test_labels = gtVector(locTest,:);
    subplot(1,2,1)
    hist(double(Train_labels),16)
    title("Train Labels Histogram")
    subplot(1,2,2)
    hist(double(Test_labels),16)
    title("Test Labels Histogram")  
end
end


