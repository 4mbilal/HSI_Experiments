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