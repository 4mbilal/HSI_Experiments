clear all
close all
clc
rng(13); 

% base_dir = 'D:\RnD\Frameworks\Datasets\HyperspectralImages';
base_dir = 'E:\Datasets\HyperspectralImages';
dataset_name = ["Indian_Pines", "Salinas", "SalinasA", "Pavia", "PaviaU", "KSC", "Botswana"];

plot_results = 0;

for dataset_idx=1:1
    [hsData, gtLabel, rgbImg, numClasses] = load_HSI_dataset(dataset_name(dataset_idx),base_dir);
    for trainRatio=0.1:0.1:0.9
        [TrainAccuracy,TestAccuracy] = trainSVM(hsData, gtLabel, rgbImg, numClasses, trainRatio, plot_results);
        fprintf('%s:Training(%.2f) Accuracy: %.4f,Test(%.2f) Accuracy: %.4f\n', dataset_name(dataset_idx),trainRatio*100,TrainAccuracy,(1-trainRatio)*100,TestAccuracy);
    end
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

svmMdl = fitcecoc(DataVector(locTrain,:),gtVector(locTrain,:));
% tSVM = templateSVM('KernelFunction','linear','BoxConstraint',0.10);
% svmMdl = fitcecoc(DataVector(locTrain,:),gtVector(locTrain,:),'Learners',tSVM,'Coding','onevsall');


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
