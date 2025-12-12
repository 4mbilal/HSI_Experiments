clear all
close all
clc
rng(13); 

% 0.70284 - baseline svm
% 0.52732 - baseline svm + scaling!!! too bad. why?
% 0.71563 - bandpooling 0.98
% 0.58261- bandpooling 0.9
% 0.69449 - bandpooling 0.999
% 0.72712 - bandpooling 0.99 

% 0.7101 - PCA_filtering K=100
% 0.74761 - PCA_filtering K=150
% 0.75444 - PCA_filtering K=170
% 0.69178 - PCA_filtering K=180
% 0.75976 - PCA_filtering K=175

% 0.76442 - PCA_downsample K=175
% 0.76485 - PCA_downsample K=150
% 0.76117 - PCA_downsample K=125
% 0.76518 - PCA_downsample K=155
% 0.7655 - PCA_downsample K=160

% 0.76388 - SG_filtering win-9,poly-2
% 0.77472 - SG_filtering win-9,poly-3
% 0.71065 - SG_filtering win-7,poly-2
% 0.73309 - SG_filtering win-7,poly-3
% 0.77613 - SG_filtering win-5,poly-3

% 0.78231 - SG_filtering win-5,poly-3 followed by PCA_downsample K=160

% 0.98124 - Spatial Filtering round(imgaussfilt(bandImage,3.75));

base_dir = 'D:\RnD\Frameworks\Datasets\HyperspectralImages';
dataset_name = ["Indian_Pines", "Salinas", "SalinasA", "Pavia", "PaviaU", "KSC", "Botswana"];

[hsData, gtLabel, rgbImg, numClasses] = load_HSI_dataset(dataset_name(1),base_dir);
% hsData = spatial_filtering(hsData,rgbImg);
hsData = PCA_spatial_filtering(hsData,rgbImg);
% hsData = filt_1(hsData,5,3);
% hsData = spatial_filtering(hsData);

% hsData = PCA_filtering(hsData,170);
% hsData = PCA_downsample(hsData,160);
% hsData = single(hsData);
% hsData = band_pooling(hsData);

% [M,N,C] = size(hsData);
% 
% cmap = parula(numClasses);
% figure
% tiledlayout(1,3,TileSpacing="loose")
% nexttile
% imshow(rgbImg)
% title("RGB Image")
% nexttile
% imshow(gtLabel,cmap)
% title("Ground Truth Map")
% 
% nexttile
% 
% for band = 1:C
%     bandImage = hsData(:,:,band); 
%     m = max(bandImage,[],"all");
%     imshow(uint8(255*bandImage/m))
%     pause(0.5)
% end

% for band = 1:C
%     bandImage = hsData(:,:,band); 
%     m = max(bandImage,[],"all");
%     subplot(1,2,1)
%     imshow(uint8(255*bandImage/m))
%     bandImage = hsData_filt(:,:,band); 
%     m = max(bandImage,[],"all");
%     subplot(1,2,2)
%     imshow(uint8(255*bandImage/m))
%     pause(0.5)
% end

% hist(hsData(:))
% hsData = hsData/max(hsData,[],"all");
% figure
% hist(hsData(:))
% figure
trainSVM(hsData,gtLabel,numClasses,rgbImg)

function trainSVM(hsData,gtLabel,numClasses,rgbImg)
[M,N,C] = size(hsData);
DataVector = reshape(hsData,[M*N C]);

gtVector = gtLabel(:);

gtLocs = find(gtVector~=0);
classLabel = gtVector(gtLocs);

per = 0.1; % Training percentage
cv = cvpartition(classLabel,HoldOut=1-per);

locTrain = gtLocs(cv.training);
locTest = gtLocs(~cv.training);
% svmMdl = fitcecoc(DataVector(locTrain,:),gtVector(locTrain,:));

% svmMdl = fitcecoc(DataVector(locTrain,:),gtVector(locTrain,:),'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus'))

% [svmLabelOut,~] = predict(svmMdl,DataVector(locTest,:));

[svmLabelOut,probs] = NaN_classifier(DataVector(locTrain,:),gtVector(locTrain,:), DataVector(locTest,:), gtVector(locTest,:));
% keyboard
% KNN model
% knnMdl = fitcknn(DataVector(locTrain,:),gtVector(locTrain,:),NumNeighbors=5,Standardize=true);
% [svmLabelOut,~] = predict(knnMdl,DataVector(locTest,:));

svmAccuracy = sum(svmLabelOut == gtVector(locTest))/numel(locTest);
disp(["Overall Accuracy (OA) of the test data using SVM = ",num2str(svmAccuracy)])

svmPredLabel = gtLabel;
svmPredLabel(locTest) = svmLabelOut;

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
confusionchart(gtVector(locTest),svmLabelOut,ColumnSummary="column-normalized")
fig_Position = fig.Position;
fig_Position(3) = fig_Position(3)*1.5;
fig.Position = fig_Position;
title("Confusion Matrix: SVM Classification Results")

end

function hsSmooth = filt_1(hsData,window,poly)
% hsData: M x N x C
[M,N,C] = size(hsData);

% Parameters (tune these)
% window = 7;      % odd, e.g., 5–11 depending on how smooth you want
% poly    = 2;     % polynomial order (1–3 typical)

% Reshape to (C x (M*N)) to apply filter per spectrum
X = reshape(hsData, [], C)';  % C x (M*N)

% Savitzky–Golay smoothing (requires Signal Processing Toolbox)
Xsg = sgolayfilt(X, poly, window, [], 1);  % filter along rows (spectral axis)

hsSmooth = reshape(Xsg', M, N, C);
% keyboard
end

function hsPCA = PCA_downsample(hsData,K)
[M,N,C] = size(hsData);
% Compute PCA on spectra (pixels x bands)
X = reshape(hsData, [], C);           % (M*N) x C

% Center data
mu = mean(X,1);
Xc = X - mu;

% PCA via SVD
[U,S,V] = svd(Xc, 'econ');
% K = 30;                                % choose components (e.g., 20–50)

Xpca = U(:,1:K) * S(1:K,1:K);          % (M*N) x K features
hsPCA = reshape(Xpca, M, N, K);
% keyboard
end

function hsRec = PCA_filtering(hsData,K)
% Dimensions
[M,N,C] = size(hsData);

% Reshape into 2D: (pixels x bands)
X = reshape(hsData, [], C);

% Center the data
mu = mean(X,1);
Xc = X - mu;

% PCA via SVD
[U,S,V] = svd(Xc, 'econ');

% Choose smaller K (e.g., 10 instead of 30)
% K = 10;

% --- Dimensionality reduction ---
Xpca = Xc * V(:,1:K);        % (M*N) x K scores

% --- Reconstruction ---
Xrec = Xpca * V(:,1:K)';     % back to (M*N) x C
Xrec = Xrec + mu;            % add mean back

% Reshape to cube
hsRec = reshape(Xrec, M, N, C);

% --- Reconstruction error ---
err = X - Xrec;              % pixel-wise error
rmse = sqrt(mean(err(:).^2)); % root mean square error

fprintf('Reconstruction RMSE with K=%d: %.4f\n', K, rmse);
% keyboard
end


% Band pooling by correlation
% - Cluster highly correlated bands: Group adjacent (or globally similar) bands and replace each group with an average or a representative band.
% - Benefit: Keeps spectral shape while reducing dimensionality and noise

function hsPooled = band_pooling(hsData)
[M,N,C] = size(hsData);
% Correlation matrix across bands
X = reshape(hsData, [], C);
R = corrcoef(X);                       % C x C
% keyboard

% Simple adjacent pooling: threshold on correlation
corrThresh = 0.99;                     % tighten/relax based on your data
groups = {};
visited = false(1,C);
for b = 1:C
    if visited(b), continue; end
    idx = find(R(b,:) >= corrThresh);
    groups{end+1} = idx;
    visited(idx) = true;
end

% Aggregate by mean (you can also use median or weighted mean)
numGroups = numel(groups);
hsPooled = zeros(M, N, numGroups, 'like', hsData);
for g = 1:numGroups
    hsPooled(:,:,g) = mean(hsData(:,:,groups{g}), 3);
end
end

function hsDataFiltered = spatial_filtering(hsData,rgbImg)
[pca1, mnf1] =getPCA_MNF(hsData);

[M,N,C] = size(hsData);
for band = 1:C
    bandImage = hsData(:,:,band); 
    % imagesc(bandImage)
    % pause(1)
    % if((band>=1)&(band<=150))
        % bandImageFiltered = imgaussfilt(bandImage,2); 
        % bandImageFiltered = medfilt2(bandImage,[9 9],"symmetric"); 
        % bandImageFiltered = imbilatfilt(bandImage,0.7e7,5);
        bandImageFiltered2 = round(imgaussfilt(bandImage,3.75));
        % bandImageFiltered = imguidedfilter(bandImage, rgb2gray(rgbImg),
        % 'NeighborhoodSize',17, 'DegreeOfSmoothing',0.7e7); % Salinas
        bandImageFiltered = imguidedfilter(bandImage, rgb2gray(rgbImg), 'NeighborhoodSize',11, 'DegreeOfSmoothing',1e7); %Indian Pines
        % bandImageFiltered = imguidedfilter(bandImage, pca1, 'NeighborhoodSize',11, 'DegreeOfSmoothing',1e7); %Indian Pines
        % bandImageFiltered = imguidedfilter(bandImage, pca1, 'NeighborhoodSize',17, 'DegreeOfSmoothing',1e7); %Salinas
        % bandImageFiltered = imguidedfilter(bandImage, mnf1, 'NeighborhoodSize',11, 'DegreeOfSmoothing',1e7); %Indian Pines
        % bandImageFiltered = imguidedfilter(bandImage, mnf1, 'NeighborhoodSize',17, 'DegreeOfSmoothing',1e7); %Salinas
        % bandImageFiltered = bandImage;

    % else
        % bandImageFiltered = bandImage; 
    % end
    % subplot(1,3,1)
    % imagesc(bandImage)
    % subplot(1,3,2)
    % imagesc(bandImageFiltered)
    % subplot(1,3,3)
    % imagesc(bandImageFiltered2)
    % pause
    % bandImageFiltered = imbilatfilt(bandImage,5,3);
    % bandImageFiltered = bandImage; 
    % bandImageGray = mat2gray(bandImageFiltered);
    % bandImageGray = (bandImageFiltered);
    mx = max(bandImageFiltered,[],"all");
    mn = min(bandImageFiltered,[],"all");
    % mn = 0;
    % mx = 255;
    % bandImageGray = (bandImageFiltered-mn)/(mx-mn);
    hsDataFiltered(:,:,band) = bandImageFiltered;
end
end

function hsDataFiltered = PCA_spatial_filtering(hsData,rgbImg)
[M,N,C] = size(hsData);
hsDataFiltered = hsData;
nB = 5;
K = 1;
for band = nB+1:C-nB
    PCAband = hsData(:,:,band-nB:band+nB);
    [pca1, mnf1] =getPCA_MNF(PCAband);

    % X = reshape(PCAband, [], 2*nB+1)';
    % mu = mean(X,1);
    % Xc = X - mu;
    % [U,S,V] = svd(Xc, 'econ');
    % % keyboard
    % Xpca = Xc * V(:,1:K);        % (M*N) x K scores
    % Xrec = Xpca * V(:,1:K)';     % back to (M*N) x C
    % Xrec = Xrec + mu;            % add mean back
    % PCAbandRec = reshape(Xrec', M, N, 2*nB+1);
    % err = X - Xrec;              % pixel-wise error
    % rmse = sqrt(mean(err(:).^2)); % root mean square error
    % fprintf('Reconstruction RMSE with K=%d: %.4f\n', K, rmse);
    
    % bandImageFiltered1 = round(PCAbandRec(:,:,nB+1));
    bandImageFiltered2 = round(imgaussfilt(hsData(:,:,band),3.75));

    bandImageFiltered1 = imguidedfilter(hsData(:,:,band), mnf1, 'NeighborhoodSize',11, 'DegreeOfSmoothing',1e3); %Indian Pines
    % bandImageFiltered1 = mean(hsData(:,:,band-nB:band+nB),3);
    % bandImageFiltered1 = round(imgaussfilt(bandImageFiltered1,3.75));
    % bandImageFiltered1 = TwoDpca(hsData(:,:,band));
    % bandImageFiltered1 = TwoDpca(bandImageFiltered1')';
    % bandImageFiltered1 = round(imgaussfilt(bandImageFiltered1,1.5));

    % se = strel("square",4);
    % bandImageFiltered1 = imdilate(hsData(:,:,band),se);
    % d = dct2(hsData(:,:,band));
    % d(5:end,5:end) = 0;
    % bandImageFiltered1 = idct2(d);

    % keyboard

        
    % subplot(1,3,1)
    % imagesc(hsData(:,:,band))
    % subplot(1,3,2)
    % imagesc(bandImageFiltered1)
    % subplot(1,3,3)
    % imagesc(bandImageFiltered2)
    % keyboard
    % pause
    % bandImageFiltered = imbilatfilt(bandImage,5,3);
    % bandImageFiltered = bandImage; 
    % bandImageGray = mat2gray(bandImageFiltered);
    % bandImageGray = (bandImageFiltered);
    % mx = max(bandImageFiltered1,[],"all");
    % mn = min(bandImageFiltered1,[],"all");
    % mn = 0;
    % mx = 255;
    % bandImageFiltered1 = (bandImageFiltered1-mn)/(mx-mn);
    hsDataFiltered(:,:,band) = bandImageFiltered1;
end
end

function Xrec=TwoDpca(in)
    [r,c] = size(in);
    X = in;
    mu = mean(X,1);
    Xc = X - mu;
    [U,S,V] = svd(Xc, 'econ');
    K = round(length(S)*0.04);
    Xpca = Xc * V(:,1:K);        % (M*N) x K scores
    Xrec = Xpca * V(:,1:K)';     % back to (M*N) x C
    Xrec = Xrec + mu;            % add mean back

    % err = X - Xrec;              % pixel-wise error
    % rmse = sqrt(mean(err(:).^2)); % root mean square error
    % fprintf('Reconstruction RMSE with K=%d: %.4f\n', K, rmse);

end