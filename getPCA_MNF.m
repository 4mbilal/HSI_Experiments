function [pca1, mnf1] =getPCA_MNF(hsData)
[M,N,C] = size(hsData);
X = reshape(hsData, [], C);
X_mean = mean(X, 1);
X_centered = X - X_mean;
[coeff, score, ~] = pca(X_centered);
pca1 = reshape(score(:,1), M, N);


% Estimate noise using local differences (basic method)
noise = zeros(size(X));
for b = 1:C
    img = reshape(hsData(:,:,b), M,N);
    noise_band = img - imfilter(img, fspecial('average', [3 3]), 'replicate');
    % keyboard
    noise(:, b) = reshape(noise_band, [], 1);
end

% Covariance matrices
Rn = cov(noise);         % Noise covariance
Rx = cov(X_centered);    % Signal covariance

% Step 1: Noise whitening
[E_n, D_n] = eig(Rn);
Wn = E_n * diag(1./sqrt(diag(D_n))) * E_n';  % Whitening matrix

X_whitened = X_centered * Wn';

% Step 2: PCA on whitened data
[coeff_mnf, score_mnf] = pca(X_whitened);

% MNF-1 image
mnf1 = reshape(score_mnf(:,1), M,N);