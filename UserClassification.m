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
maxUsers = 5000; % set to Inf to use all users (may not fit memory)
maxMovies = 5000; % set to Inf to use all movies
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
verbose = true;
if verbose
    disp(Rfull(1:15, 1:8));
end
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