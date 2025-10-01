% NOTES:
% - Tunable parameters are at the top.
% - For the full 32M dataset you'll likely need to run on a subset
%   or use tall arrays / datastores and more advanced scaling.
% - Requires: Statistics and Machine Learning Toolbox (kmeans, svds)

clear; clc;
% File names
ratingsFile = 'ratings.csv';
moviesFile = 'movies.csv';
% tagsFile = 'tags.csv';

% Subsampling / memory controls (set large to use more data)
maxUsers = Inf; % set to Inf to use all users (may not fit memory)
maxMovies = Inf; % set to Inf to use all movies
minRatingsPerUser = 5; % filter out very inactive users
minRatingsPerMovie = 5; % filter out very unpopular movies

% Model params
kLatent = 20; % number of PCA components (max = numGenres)
numClusters = 10; % desired user clusters
kmeansReplicates = 8;

%% 1) Load CSVs
fprintf('Loading CSV files...\n');
ratings = readtable(ratingsFile);
movies = readtable(moviesFile);
% tags = readtable(tagsFile); % optional (not used directly below)
fprintf('Step 1 Done.\n');

%% 2) Map IDs to compact indices
fprintf('Mapping IDs...\n');
[uniqueUsers, ~, userIdx] = unique(ratings.userId);
[uniqueMovies, ~, movieIdx] = unique(ratings.movieId);
numUsersAll = numel(uniqueUsers);
numMoviesAll = numel(uniqueMovies);
fprintf('Total users: %d, Total movies: %d, Total ratings: %d\n', numUsersAll, numMoviesAll, height(ratings));
fprintf('Step 2 Done.\n');

%% 3) Build sparse user-movie matrix (full dataset indices)
fprintf('Building sparse rating matrix...\n');
Rfull = sparse(userIdx, movieIdx, ratings.rating, numUsersAll, numMoviesAll);
fprintf('Step 3 Done.\n');

%% 4) Filter users & movies by activity
userCounts = sum(Rfull~=0, 2);
movieCounts = sum(Rfull~=0, 1);
activeUsers = find(userCounts >= minRatingsPerUser);
popularMovies = find(movieCounts >= minRatingsPerMovie);


% Optionally restrict to top N users/movies by counts
if ~isinf(maxUsers) && numel(activeUsers) > maxUsers
[~, ix] = sort(userCounts(activeUsers), 'descend');
activeUsers = activeUsers(ix(1:maxUsers));
end
if ~isinf(maxMovies) && numel(popularMovies) > maxMovies
[~, ix2] = sort(movieCounts(popularMovies), 'descend');
popularMovies = popularMovies(ix2(1:maxMovies));
end
fprintf('Step 4 Done.\n');

%% 5) Subset rating matrix for analysis
R = Rfull(activeUsers, popularMovies);
clear Rfull userCounts movieCounts;
fprintf('Step 5 Done.\n');

%% 6) Dimensionality reduction via PCA on user-genre matrix
fprintf('Step 6 (PCA) starting...\n');

% map selected movies to movie rows & genres
selectedMovieIds = uniqueMovies(popularMovies);
[~, movieRows] = ismember(selectedMovieIds, movies.movieId);
movieGenres = movies.genres(movieRows);

% build movie x genre binary matrix G
allTokens = cellfun(@(s) strsplit(s,'|'), movieGenres, 'UniformOutput', false);
genreList = unique([allTokens{:}]);
numGenres = numel(genreList);
G = false(numel(selectedMovieIds), numGenres);
for i = 1:numel(selectedMovieIds)
    toks = allTokens{i};
    [tf, loc] = ismember(toks, genreList);
    G(i, loc(tf)) = true;
end

% aggregate: sum ratings per user per genre, then average by counts
Ugenre = R * double(G);                 % sums (numUsers x numGenres)
userGenreCounts = (R ~= 0) * double(G); % counts (numUsers x numGenres)
Ugenre = full(Ugenre); % Convert to full for PCA (user x genre is usually small enough to fit in memory)
userGenreCounts = full(userGenreCounts);
userGenreCounts = max(userGenreCounts, 1);   % replace zeros with 1 safely
Ugenre = Ugenre ./ userGenreCounts;          % average rating per genre

% PCA (small matrix), keep <= numGenres components
kLatent = min(kLatent, numGenres);
fprintf('Running PCA for k=%d...\n', kLatent);
[coeff, score, latentVals] = pca(Ugenre, 'NumComponents', kLatent);

% keep variable names compatible with downstream code
userFeatures = score;                       % numUsers x kLatent
U = userFeatures;
% latentVals are variances (eigenvalues). sqrt gives singular-value-like scaling for compatibility
S = diag(sqrt(latentVals(1:kLatent)));     % approximate scaling matrix 
V = coeff(:,1:kLatent);                     % genre loadings

fprintf('Step 6 Done.\n');

%% 7) Train/test split on observed ratings (hold-out) 
fprintf('Step 7: creating train/test split...\n');
test_frac = 0.20;        % fraction of observed ratings to hold out for testing
rng(1);                  % reproducible

% get all observed entries from R (R is numUsers x numMovies sparse)
[u_idx, m_idx, r_vals] = find(R);
nObs = numel(r_vals);
perm = randperm(nObs);
nTest = round(test_frac * nObs);
testPos = perm(1:nTest);
trainPos = perm(nTest+1:end);

test_u = u_idx(testPos);
test_m = m_idx(testPos);
test_r = r_vals(testPos);

train_u = u_idx(trainPos);
train_m = m_idx(trainPos);
train_r = r_vals(trainPos);

% build sparse training matrix
R_train = sparse(train_u, train_m, train_r, size(R,1), size(R,2));

% quick sanity
fprintf('Observed ratings: %d, Train: %d, Test: %d\n', nObs, numel(train_r), numel(test_r));
fprintf('Step 7 Done.\n');

%% Step 8 (modified): User × genre with shrinkage (smoothing)
fprintf('Step 8: computing user-genre with shrinkage (smoothing)...\n');

% Sums and counts (use original train matrix)
sums_genre = R_train * double(G);                 % numUsers x numGenres : sum of ratings per user-genre
counts_genre = (R_train ~= 0) * double(G);        % numUsers x numGenres : counts per user-genre
sums_genre = full(sums_genre);
counts_genre = full(counts_genre);

% shrinkage parameter alpha (tunable)
alpha = 10;   % start with something like 5..50 depending on dataset size

% compute shrunk user-genre mean:
% shrunk_mean = (sum + alpha * globalMean) / (count + alpha)
% note: globalMean should be computed on training ratings
globalMean = sum(train_r) / max(numel(train_r),1);
Ugenre_shrunk = (sums_genre + alpha * globalMean) ./ (counts_genre + alpha);

% For downstream use also keep a version of counts (used as weights)
userGenreCounts_train = counts_genre;  % numUsers x numGenres

fprintf('Computed shrunk user-genre means (alpha = %g)\n', alpha);
fprintf('Step 8 Done.\n');

%% Step 9 (tempered biases + weighted genre): Predict with tempered user/movie biases + weighted genres + shrinkage
fprintf('Step 9: predicting test ratings with tempered biases + weighted genres + shrinkage...\n');

% hyperparams (tweak these)
beta         = 0.5;   % weighting power for counts (0 => equal weights, 1 => linear)
lambda_genre = 0.7;   % blend: how much to trust genre-based prediction vs baseline (0..1)
lambda_user  = 0.5;   % scale for user bias (0..1)
lambda_movie = 0.5;   % scale for movie bias (0..1)

% precompute user/movie baseline stats (ensure full arrays)
userCount = sum(R_train ~= 0, 2);                        % numUsers x 1 (sparse-friendly)
userSum   = full(sum(R_train, 2));                       % numUsers x 1
userMean  = userSum ./ max(userCount,1);                 % numUsers x 1
userMean(userCount==0) = globalMean;                     % cold users -> global

movieCount = full(sum(R_train ~= 0, 1));                 % 1 x numMovies
movieSum   = full(sum(R_train, 1));                      % 1 x numMovies
movieMeanVec = movieSum ./ max(movieCount,1);            % 1 x numMovies
movieMeanVec(movieCount==0) = globalMean;                % cold movies -> global

% baseline biases (full numeric)
b_u_vec = full(userMean - globalMean);      % numUsers x 1
b_m_vec = full(movieMeanVec - globalMean);  % 1 x numMovies

% ensure Ugenre_shrunk and userGenreCounts_train are full numeric matrices
Ugenre_shrunk_full = full(Ugenre_shrunk);
userGenreCounts_full = full(userGenreCounts_train);

nTest = numel(test_r);
preds = zeros(nTest,1);

for t = 1:nTest
    u = test_u(t);
    m = test_m(t);

    % tempered baseline (global + scaled user bias + scaled movie bias)
    b_u = 0; b_m = 0;
    if userCount(u) > 0
        b_u = lambda_user * b_u_vec(u);
    end
    b_m = lambda_movie * b_m_vec(m);

    baseline = globalMean + b_u + b_m;

    % genre-based prediction (weighted using counts^beta, values from Ugenre_shrunk)
    movieGenresIdx = find(G(m, :));
    if isempty(movieGenresIdx)
        % no genre info -> use baseline
        p_genre = NaN;
    else
        vals = Ugenre_shrunk_full(u, movieGenresIdx);                 % 1 x nGenresMovie
        counts = userGenreCounts_full(u, movieGenresIdx);            % 1 x nGenresMovie
        weights = counts .^ beta;
        wsum = sum(weights);
        if wsum > 0
            p_genre = sum(weights .* vals) / wsum;
        else
            p_genre = NaN;
        end
    end

    % final prediction: baseline + lambda_genre * (genre_pred - globalMean)
    if isnan(p_genre)
        p = baseline;
    else
        p = baseline + lambda_genre * (p_genre - globalMean);
    end

    % clip to allowed rating range
    p = min(max(p, 0.5), 5.0);
    preds(t) = p;
end

% compute errors & metrics
errors = preds - double(test_r);
rmse = sqrt(mean(errors.^2));
mae = mean(abs(errors));

fprintf('Prediction results (tempered biases + weighted+shrunk): RMSE = %.4f, MAE = %.4f\n', rmse, mae);

% Accuracy within tolerance windows
tol1 = 0.5;
tol2 = 1.0;
acc_tol1 = mean(abs(errors) <= tol1);
acc_tol2 = mean(abs(errors) <= tol2);
fprintf('Accuracy within ±0.5 stars: %.2f%%\n', 100*acc_tol1);
fprintf('Accuracy within ±1.0 stars: %.2f%%\n', 100*acc_tol2);

% Top worst predictions for diagnostics (as before)
[~, worstIdx] = sort(abs(errors), 'descend');
nShow = min(10, numel(worstIdx));
fprintf('Top %d worst test predictions (abs error):\n', nShow);
for i = 1:nShow
    t = worstIdx(i);
    fprintf('%2d) user %d, movie %d, true=%.1f, pred=%.3f, absErr=%.3f\n', ...
        i, test_u(t), test_m(t), test_r(t), preds(t), abs(errors(t)));
end

% Print diagnostics for those worst cases
for i = 1:nShow
    t = worstIdx(i);
    u = test_u(t);
    m = test_m(t);

    u_ratings = full(R_train(u, :));  % convert row to full numeric
    m_ratings = full(R_train(:, m));  % convert column to full numeric

    fprintf('User %d: #ratings=%d, mean=%.2f, std=%.2f\n', ...
        u, nnz(u_ratings), mean(nonzeros(u_ratings)), std(nonzeros(u_ratings)));
    fprintf('Movie %d: #ratings=%d, mean=%.2f, std=%.2f\n', ...
        m, nnz(m_ratings), mean(nonzeros(m_ratings)), std(nonzeros(m_ratings)));

    % show the genre-based components used (for debugging)
    movieGenresIdx = find(G(m, :));
    if ~isempty(movieGenresIdx)
        vals = Ugenre_shrunk_full(u, movieGenresIdx);
        counts = userGenreCounts_full(u, movieGenresIdx);
        fprintf('  Genres used (idx): %s\n', mat2str(movieGenresIdx));
        fprintf('  Genre vals: %s\n', mat2str(round(vals,3)));
        fprintf('  Genre counts: %s\n', mat2str(counts));
    else
        fprintf('  No genre tags for movie %d\n', m);
    end
end

fprintf('Step 9 Done.\n');

%% Scaling notes (manual)
% - For very large datasets (32M ratings): consider using datastore/tall arrays
%   or process in chunks to build counts and identify top users/movies.
% - Use svds on sparse matrices (as used above) to avoid converting to full.
% - If memory is still an issue, consider sampling users or movies,
%   or using an implicit-factorization algorithm (e.g., ALS) implemented
%   in specialized libraries.
% - You can also create aggregated user-by-genre profiles (instead of full
%   user-by-movie) if you trust genre signals; this greatly reduces columns.

% End of script