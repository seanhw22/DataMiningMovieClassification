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

%% 8) Classification of users by genre (favorite-only based on training data)
fprintf('Step 8: computing user-genre favorites on training data (favorite-only)...\n');

% compute user x genre average using training data (same as Step 6 process)
Ugenre_train = R_train * double(G);                 % sums
userGenreCounts_train = (R_train ~= 0) * double(G); % counts
Ugenre_train = full(Ugenre_train);
userGenreCounts_train = full(userGenreCounts_train);
userGenreCounts_train = max(userGenreCounts_train, 1); % avoid zeros
Ugenre_train = Ugenre_train ./ userGenreCounts_train; % avg rating per genre (train)

% define a "like" threshold (tuneable)
like_threshold = 3.5;

% For each user, find the genre with the maximum average
[maxVals, maxIdx] = max(Ugenre_train, [], 2);   % maxVals: numUsers x 1, maxIdx: numUsers x 1

% Only keep users whose top genre average >= threshold
validUsersMask = maxVals >= like_threshold;

% favorite genre index for users who pass threshold
favIdx = maxIdx(validUsersMask);   % vector of genre indices (values in 1..numGenres)

% Count favorites per genre (each user contributes at most 1)
usersPerGenre_fav = zeros(1, numGenres);
if ~isempty(favIdx)
    genreCounts = accumarray(favIdx, 1, [numGenres, 1]); % numGenres x 1
    usersPerGenre_fav = genreCounts';
end

% sort and show top genres by favorite counts
[vals_fav, ord_fav] = sort(usersPerGenre_fav, 'descend');
fprintf('Top favorite genres by number of users (threshold = %.1f):\n', like_threshold);
for k = 1:min(10, numel(ord_fav))
    fprintf('%2d) %-15s : %d users\n', k, genreList{ord_fav(k)}, vals_fav(k));
end

% Optional: show how many users didn't have any favorite (below threshold)
nUsers = size(Ugenre_train,1);
nNoFavorite = sum(~validUsersMask);
fprintf('Users counted: %d (out of %d). Users with no favorite (below threshold): %d\n', sum(validUsersMask), nUsers, nNoFavorite);

fprintf('Step 8 Done.\n');

%% 9) Predict ratings for held-out test set (content-based using genre averages) and evaluate
fprintf('Step 9: predicting test ratings and evaluating (RMSE, MAE)...\n');

% user global mean (fallback)
userSum = sum(R_train, 2);
userCount = sum(R_train~=0, 2);
userMean = full(userSum ./ max(userCount,1)); % numUsers x 1
globalMean = sum(train_r) / max(numel(train_r),1);

nTest = numel(test_r);
preds = zeros(nTest,1);

for t = 1:nTest
    u = test_u(t);
    m = test_m(t);
    % movie m corresponds to row m of G (because R columns align with selectedMovieIds/G rows)
    movieGenresIdx = find(G(m, :));
    if isempty(movieGenresIdx)
        % no genre info: fallback to user mean or global mean
        if userCount(u) > 0
            p = userMean(u);
        else
            movieMean = mean(nonzeros(R_train(:, m)));
            if isempty(movieMean)
                p = globalMean;
            else
                p = movieMean;
            end
        end
    else
        % predict as average of user's genre averages for that movie's genres
        p = mean(Ugenre_train(u, movieGenresIdx));
        % if the user has no ratings contributing to these genres, p may be NaN
        if isnan(p)
            if userCount(u) > 0
                p = userMean(u);
            else
                movieMean = mean(nonzeros(R_train(:, m)));
                if isempty(movieMean)
                    p = globalMean;
                else
                    p = movieMean;
                end
            end
        end
    end
    % clip to rating range (assumes 0.5 steps; adjust if needed)
    p = min(max(p, 0.5), 5.0);
    preds(t) = p;
end

% compute errors
errors = preds - double(test_r);
rmse = sqrt(mean(errors.^2));
mae = mean(abs(errors));

fprintf('Prediction results on test set: RMSE = %.4f, MAE = %.4f\n', rmse, mae);

% Accuracy within tolerance windows
tol1 = 0.5;
tol2 = 1.0;
acc_tol1 = mean(abs(errors) <= tol1);
acc_tol2 = mean(abs(errors) <= tol2);
fprintf('Accuracy within ±0.5 stars: %.2f%%\n', 100*acc_tol1);
fprintf('Accuracy within ±1.0 stars: %.2f%%\n', 100*acc_tol2);

% (Optional) a small diagnostic: top bad predictions
[~, worstIdx] = sort(abs(errors), 'descend');
nShow = min(10, numel(worstIdx));
fprintf('Top %d worst test predictions (abs error):\n', nShow);
for i = 1:nShow
    t = worstIdx(i);
    fprintf('%2d) user %d, movie %d, true=%.1f, pred=%.3f, absErr=%.3f\n', i, test_u(t), test_m(t), test_r(t), preds(t), abs(errors(t)));
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