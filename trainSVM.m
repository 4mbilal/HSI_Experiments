function [TrainAccuracy,TestAccuracy_Overall,TestAccuracy_Average,TestKappa,perClassAcc,t_test] = trainSVM(hsData,hsDataFiltered, gtLabel, rgbImg, numClasses, trainRatio, plot_results)
[M,N,C] = size(hsDataFiltered);
% DataVector = reshape(hsData,[M*N C]);
DataVectorFiltered = reshape(hsDataFiltered,[M*N C]);

% DataVectorFiltered = DataVectorFiltered/10000;
% DataVectorFiltered = [DataVectorFiltered;DataVectorFiltered.^2];

gtVector = gtLabel(:);

gtLocs = find(gtVector~=0);
classLabel = gtVector(gtLocs);

% Percentage Split
cv = cvpartition(classLabel,HoldOut=1-trainRatio);
% Sample-based split
% cv = fixedClassPartition(classLabel, 5);

locTrain = gtLocs(cv.training);
locTest = gtLocs(~cv.training);

% trainD = DataVectorFiltered(locTrain,:);
% trainL = gtVector(locTrain,:);
% 
% keyboard

% svmMdl = fitcecoc(DataVector(locTrain,:),gtVector(locTrain,:));
tSVM = templateSVM('KernelFunction','linear','BoxConstraint',1.0);
% tSVM = templateSVM('KernelFunction','gaussian','BoxConstraint',1,'Standardize',true);
% tSVM = templateSVM(...
%     'KernelFunction', 'gaussian', ...
%     'PolynomialOrder', [], ...
%     'KernelScale', 0.5, ...%6.5, ...
%     'BoxConstraint', 0.3, ...%5.5, ...
%     'Standardize', false);


svmMdl = fitcecoc(DataVectorFiltered(locTrain,:),gtVector(locTrain,:),'Learners',tSVM,'Coding','onevsone');

% keyboard
% svmMdl = fitcecoc(DataVector(locTrain,:),gtVector(locTrain,:),'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus','UseParallel',true,'MaxObjectiveEvaluations',10,'Kfold',2));

% [TestPredictions,probs_nan] = NaN_classifier(DataVector(locTrain,:),gtVector(locTrain,:), DataVector(locTest,:), gtVector(locTest,:));
% 
tic;
[TrainPredictions,~] = predict(svmMdl,DataVectorFiltered(locTrain,:));
t_test = toc;
TrainAccuracy = sum(TrainPredictions == gtVector(locTrain))/numel(locTrain);
% TrainAccuracy = 1;
% 
[TestPredictions,~] = predict(svmMdl,DataVectorFiltered(locTest,:));
TestAccuracy_Overall = sum(TestPredictions == gtVector(locTest))/numel(locTest);

[TestAccuracy_Average,TestKappa,perClassAcc] = getAverageAccuracy(TestPredictions,gtVector(locTest));


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

function [AA,kappa,perClassAcc] = getAverageAccuracy(TestPredictions,TestLabels)
% Ensure column vectors
TestPredictions = TestPredictions(:);
TestLabels      = TestLabels(:);

% Get list of unique classes
classes = unique(TestLabels);

% Preallocate
perClassAcc = zeros(length(classes),1);

% Compute per-class accuracy
for i = 1:length(classes)
    c = classes(i);
    idx = (TestLabels == c);              % samples belonging to class c
    perClassAcc(i) = sum(TestPredictions(idx) == c) / sum(idx);
end

% Average Accuracy
AA = mean(perClassAcc);

% Confusion matrix
C = confusionmat(TestLabels, TestPredictions);

% Total samples
N = sum(C(:));

% Observed agreement (OA)
p_o = trace(C) / N;

% Expected agreement
row_marginals = sum(C, 2);   % true class counts
col_marginals = sum(C, 1);   % predicted class counts
p_e = sum(row_marginals .* col_marginals') / N^2;

% Cohen's kappa
kappa = (p_o - p_e) / (1 - p_e);

end


function C = fixedClassPartition(labels, k)
% fixedClassPartition  Create a cvpartition-like object with exactly k samples per class
%
%   C = fixedClassPartition(labels, k)
%
%   C.training : logical vector (true = training sample)
%   C.test     : logical vector (true = test sample)

labels = labels(:);
classes = unique(labels);

trainIdx = false(size(labels));

for i = 1:length(classes)
    c = classes(i);
    idx = find(labels == c);          % all samples of class c
    idx = idx(randperm(length(idx))); % shuffle
    n = min(k, numel(idx));           % safety check
    trainIdx(idx(1:n)) = true;
end

C.training = trainIdx;
C.test     = ~trainIdx;

end
