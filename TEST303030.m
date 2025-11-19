%% ===============================================================
%  MovieLens ml-32m — Two-Step MANOVA + Classification (MATLAB)
%  STEP 1: MANOVA to test multivariate genre differences
%  STEP 2: LDA classification based on those variables
%  Requirements:
%     - movies.csv, ratings.csv, tags.csv in current folder
%     - Statistics and Machine Learning Toolbox
% ================================================================

clear; clc; close all;

%% 1. LOAD DATA
% Adjust paths if needed
moviesFile  = 'movies.csv';    % movieId,title,genres
ratingsFile = 'ratings.csv';   % userId,movieId,rating,timestamp
tagsFile    = 'tags.csv';      % userId,movieId,tag,timestamp

fprintf('Loading CSV files...\n');
movies  = readtable(moviesFile,  'FileEncoding','UTF-8');
ratings = readtable(ratingsFile, 'FileEncoding','UTF-8');
tags    = readtable(tagsFile,    'FileEncoding','UTF-8');

%% 2. AGGREGATE: MEAN RATING & TAG COUNT PER MOVIE

% ---- Mean rating per movie ----
% groupsummary: group by movieId, take mean rating
fprintf('Computing mean rating per movie...\n');
avgRating = groupsummary(ratings, "movieId", "mean", "rating");
% Rename columns for clarity: movieId, GroupCount, mean_rating
avgRating.Properties.VariableNames = {'movieId','GroupCount','mean_rating'};

% ---- Tag count per movie ----
% Here we just count number of rows per movieId in tags table
fprintf('Computing tag count per movie...\n');
tagCount = groupsummary(tags, "movieId");   % GroupCount = #tags
tagCount = tagCount(:, {'movieId','GroupCount'});
tagCount.Properties.VariableNames = {'movieId','tag_count'};

%% 3. MERGE INTO A SINGLE TABLE (MOVIE-LEVEL DATA)

fprintf('Merging movies, ratings, and tags...\n');
% Merge movies and avgRating
data = outerjoin(movies, avgRating(:,["movieId","mean_rating"]), ...
                 "Keys","movieId", "MergeKeys",true);
% Merge with tagCount
data = outerjoin(data, tagCount, ...
                 "Keys","movieId", "MergeKeys",true);

% Movies without tags → tag_count = 0
if any(isnan(data.tag_count))
    data.tag_count(isnan(data.tag_count)) = 0;
end

%% 4. EXTRACT MAIN GENRE (FIRST GENRE ONLY)

fprintf('Extracting main_genre (first genre before "|")...\n');

% Preallocate main_genre as string array
data.main_genre = strings(height(data),1);

for i = 1:height(data)
    % genres column might be string or char; convert robustly:
    g = string(data.genres(i));
    if contains(g, "|")
        g = extractBefore(g, "|");
    end
    data.main_genre(i) = g;
end

% Optionally remove rows with '(no genres listed)'
maskNoGenre = (data.main_genre == "(no genres listed)");
data(maskNoGenre,:) = [];

%% 5. CLEAN DATA: REMOVE MOVIES WITHOUT RATINGS

fprintf('Removing movies without mean_rating...\n');
data = rmmissing(data, 'DataVariables', {'mean_rating'});

% Optional: filter out genres with very few movies (e.g., < 50)
minPerGenre = 50;
fprintf('Filtering genres with at least %d movies...\n', minPerGenre);
genreCounts = groupsummary(data, "main_genre");
keepGenres = genreCounts.main_genre(genreCounts.GroupCount >= minPerGenre);
data = data(ismember(data.main_genre, keepGenres), :);

fprintf('Final number of movies: %d\n', height(data));
fprintf('Number of genres: %d\n', numel(unique(data.main_genre)));

%% 6. DEFINE VARIABLES FOR MANOVA + CLASSIFICATION

% Dependent variables:
%   - mean_rating : average user rating (1–5)
%   - tag_count   : number of tags per movie (can be very skewed)
Y = [data.mean_rating, data.tag_count];

% If tag_count is extremely skewed, you can instead use:
% Y = [data.mean_rating, log1p(data.tag_count)];

% Grouping variable: main_genre
group = categorical(data.main_genre);

%% 7. STEP 1 — MANOVA

fprintf('\n==================== STEP 1: MANOVA ====================\n');
[d, p, stats] = manova1(Y, group);

fprintf('MANOVA p-value (overall multivariate test): %.4g\n', p);
fprintf('Number of canonical dimensions: %d\n', d);

% Canonical discriminant plot (first two canonical functions)
figure;
gscatter(stats.canon(:,1), stats.canon(:,2), group);
xlabel('Canonical Function 1');
ylabel('Canonical Function 2');
title('Canonical Discriminant Scores by Genre');
legend('Location','bestoutside');
grid on;

%% 8. STEP 2 — CLASSIFICATION (LDA) WITH TRAIN/TEST SPLIT

fprintf('\n==================== STEP 2: CLASSIFICATION ====================\n');

% We do a hold-out validation: 70% train, 30% test
cv = cvpartition(group, 'Holdout', 0.3);
trainIdx = training(cv);
testIdx  = test(cv);

Ytrain  = Y(trainIdx, :);
gTrain  = group(trainIdx);
Ytest   = Y(testIdx, :);
gTest   = group(testIdx);

% Linear Discriminant Analysis (LDA)
% classify(sample, trainingData, trainingGroups)
fprintf('Training LDA classifier (classify)...\n');
gPred = classify(Ytest, Ytrain, gTrain, 'linear');

% Classification accuracy
accuracy = mean(gPred == gTest);
fprintf('Classification Accuracy (test set): %.2f%%\n', accuracy * 100);

% Confusion matrix
confMat = confusionmat(gTest, gPred);
disp('Confusion Matrix (counts):');
disp(confMat);

% Confusion matrix as row-wise percentages
confMatPct = 100 * confMat ./ sum(confMat, 2);
disp('Confusion Matrix (% per true class):');
disp(confMatPct);

% (Optional) show genre labels
genresList = categories(group);
fprintf('\nGenre order in confusion matrix rows/cols:\n');
disp(genresList);

%% 9. OPTIONAL: RESUBSTITUTION ACCURACY (USING ALL DATA FOR TRAIN & TEST)
% (This usually overestimates accuracy, but can be shown for comparison)

fprintf('\nComputing resubstitution accuracy (all data)...\n');
gPredAll = classify(Y, Y, group, 'linear');
accAll   = mean(gPredAll == group);
fprintf('Resubstitution Accuracy (all data): %.2f%%\n', accAll * 100);

fprintf('\nDone.\n');

%% 10. BUILD USER–MOVIE MATRIX (R_train) DARI ratings.csv
fprintf('\n==================== STEP 10: BUILD USER–MOVIE MATRIX ====================\n');

% Map userId dan movieId ke indeks kompakt (1..numUsers, 1..numMovies)
[uniqueUsers, ~, userIdx]  = unique(ratings.userId);
[uniqueMovies, ~, movieIdx] = unique(ratings.movieId);
numUsers  = numel(uniqueUsers);
numMovies = numel(uniqueMovies);

fprintf('Num users: %d, Num movies: %d\n', numUsers, numMovies);

% Bangun matriks sparse user x movie berisi rating
R = sparse(userIdx, movieIdx, ratings.rating, numUsers, numMovies);

% Untuk tugas klasifikasi ini kita anggap seluruh R sebagai "training"
R_train = R;

% Global mean rating (dibutuhkan untuk shrinkage)
allRatings = double(ratings.rating);
globalMean = mean(allRatings);

fprintf('Global mean rating: %.4f\n', globalMean);

%% 11. BUILD MOVIE–GENRE MATRIX (G)
fprintf('\n==================== STEP 11: BUILD MOVIE–GENRE MATRIX ====================\n');

% Kita butuh genres untuk semua movieId yang ada di ratings (uniqueMovies)
% Cari baris di tabel movies yang cocok dengan uniqueMovies
[found, movieRows] = ismember(uniqueMovies, movies.movieId);

% Beberapa movieId mungkin tidak ditemukan di movies.csv (harus dicek)
if any(~found)
    warning('Ada movieId di ratings yang tidak ditemukan di movies.csv. Mereka akan di-skip dalam G.');
end

% Ambil genres untuk movie yang ditemukan
movieGenres = movies.genres(movieRows(found));   % sebagai cell/array of strings
numMoviesG  = sum(found);

% Tokenisasi genre per film (dipisah dengan "|")
allTokens = cellfun(@(s) strsplit(string(s),'|'), movieGenres, 'UniformOutput', false);
genreList = unique([allTokens{:}]);   % daftar semua genre unik
numGenres = numel(genreList);

fprintf('Num movies (dengan genre): %d, Num genres: %d\n', numMoviesG, numGenres);

% Build matriks movie x genre: G(i,g) = 1 jika movie i punya genre g
G = false(numMovies, numGenres);  % pakai seluruh kolom movie (urutannya = uniqueMovies)

for i = 1:numMovies
    if ~found(i)
        continue; % skip yang tidak punya info genre
    end
    toks = allTokens{find(find(found)==i,1)};  % ambil token genre untuk movie ini
    [tf, loc] = ismember(toks, genreList);
    G(i, loc(tf)) = true;
end

fprintf('Matriks G (movie x genre) selesai dibuat. Ukuran: %d x %d\n', size(G,1), size(G,2));

%% 12. HITUNG USER–GENRE PROFILE DENGAN SHRINKAGE (Ugenre_shrunk)
fprintf('\n==================== STEP 12: USER–GENRE PROFILE (SHRINKAGE) ====================\n');

% sum rating per user–genre
sums_genre   = R_train * double(G);           % numUsers x numGenres
counts_genre = (R_train ~= 0) * double(G);    % numUsers x numGenres

sums_genre   = full(sums_genre);
counts_genre = full(counts_genre);

% parameter shrinkage (alpha)
alpha = 10;

% shrunk mean: (sum + alpha*globalMean) / (count + alpha)
Ugenre_shrunk = (sums_genre + alpha * globalMean) ./ (counts_genre + alpha);

fprintf('Ugenre_shrunk dihitung. Ukuran: %d users x %d genres\n', size(Ugenre_shrunk,1), size(Ugenre_shrunk,2));

%% ===============================================================
%  STEP 13–15: KLASIFIKASI
%  "APAKAH RATING INI KARENA GENRE FAVORIT USER ATAU TIDAK?"
%
%  Prasyarat (sudah dihitung di script sebelumnya):
%    - R_train          : matriks user x movie (sparse), hanya data training
%    - Ugenre_shrunk    : matriks user x genre (rata-rata rating per genre, sudah shrink)
%    - G                : matriks movie x genre (boolean/0-1), genre film
% ===============================================================

fprintf('\n==================== STEP 13: FAVORITE GENRE PER USER ====================\n');

% Ugenre_shrunk: numUsers x numGenres
[numUsers, numGenres] = size(Ugenre_shrunk);

% Genre favorit = genre dengan rata-rata rating tertinggi untuk tiap user
[~, favGenreIdx] = max(Ugenre_shrunk, [], 2);   % size: numUsers x 1 (indeks genre favorit)

% Cek sekilas
fprintf('Contoh 5 user pertama dan indeks genre favoritnya:\n');
disp(favGenreIdx(1:min(5,numUsers)));

%% STEP 14: BENTUK DATASET KLASIFIKASI (X, y)
fprintf('\n==================== STEP 14: MEMBENTUK DATASET KLASIFIKASI ====================\n');

% Ambil semua rating dari R_train
% u_idx: indeks user, m_idx: indeks movie, r_vals: nilai rating
[u_idx_all, m_idx_all, r_vals_all] = find(R_train);
nObs = numel(r_vals_all);
fprintf('Jumlah rating di R_{train}: %d\n', nObs);

% Untuk efisiensi, kita sampling sebagian rating saja (misal 200k)
maxSamples = 200000;  % bisa diubah: 100k, 200k, dst
if nObs > maxSamples
    rng(1); % reproducible
    perm = randperm(nObs, maxSamples);
    u_idx = u_idx_all(perm);
    m_idx = m_idx_all(perm);
    r_vals = r_vals_all(perm);
else
    u_idx = u_idx_all;
    m_idx = m_idx_all;
    r_vals = r_vals_all;
end

n = numel(r_vals);
fprintf('Dipakai untuk klasifikasi: %d rating (sample)\n', n);

% Label: 1 jika film termasuk genre favorit user, 0 jika tidak
isFavGenre = false(n, 1);

for i = 1:n
    u = u_idx(i);
    m = m_idx(i);

    gFav = favGenreIdx(u);      % indeks genre favorit user
    if gFav >= 1 && gFav <= numGenres
        % Jika film m punya genre favorit itu (G(m,gFav) == 1)
        if G(m, gFav)
            isFavGenre(i) = true;
        end
    end
end

y = double(isFavGenre);  % label biner: 1 = rating pada genre favorit, 0 = bukan

% Buat beberapa fitur (X):
%  X1: rating aktual
%  X2: rata-rata rating user (semua genre)
%  X3: rata-rata rating movie (dari R_train)
%  X4: |rating - rata-rata user| (seberapa ekstrem rating ini dibanding preferensi user)
fprintf('Menghitung fitur X...\n');

% Rata-rata rating user (bisa pakai mean dari Ugenre_shrunk)
userMeanVec = mean(Ugenre_shrunk, 2);  % numUsers x 1

% Rata-rata rating movie (dari R_train)
movieCount = sum(R_train ~= 0, 1);             % 1 x numMovies
movieSum   = sum(R_train, 1);                  % 1 x numMovies
movieMeanVec = full(movieSum ./ max(movieCount, 1));   % 1 x numMovies
movieMeanVec(movieCount == 0) = globalMean;    % jaga-jaga

% Alokasi matriks fitur
X = zeros(n, 4);

for i = 1:n
    u = u_idx(i);
    m = m_idx(i);
    r = double(r_vals(i));

    uMean = userMeanVec(u);
    mMean = movieMeanVec(m);

    X(i,1) = r;
    X(i,2) = uMean;
    X(i,3) = mMean;
    X(i,4) = abs(r - uMean);   % deviasi rating dari rata-rata user
end

fprintf('Contoh 5 baris pertama X dan y:\n');
disp(table(X(1:5,1), X(1:5,2), X(1:5,3), X(1:5,4), y(1:5), ...
    'VariableNames', {'rating','userMean','movieMean','devUser','isFavGenre'}));

%% STEP 15: TRAIN/TEST SPLIT + LOGISTIC REGRESSION (BINARY CLASSIFICATION)
fprintf('\n==================== STEP 15: TRAIN/TEST SPLIT & LOGISTIC REGRESSION ====================\n');

% Bagi data menjadi train/test (misal 70% train, 30% test)
rng(2);
cvBin = cvpartition(y, 'HoldOut', 0.3);
trainIdx = training(cvBin);
testIdx  = test(cvBin);

Xtrain = X(trainIdx, :);
ytrain = y(trainIdx);
Xtest  = X(testIdx, :);
ytest  = y(testIdx);

fprintf('Ukuran train: %d, test: %d\n', numel(ytrain), numel(ytest));

% Latih model logistic regression
% fitglm -> generalized linear model dengan distribusi binomial
fprintf('Melatih model logistic regression...\n');
mdl = fitglm(Xtrain, ytrain, ...
    'Distribution','binomial', ...
    'Link','logit');

% Prediksi probabilitas rating "genre favorit" di test set
ypred_prob = predict(mdl, Xtest);
ypred = ypred_prob >= 0.5;  % threshold 0.5

% Hitung akurasi
acc = mean(ypred == ytest);
fprintf('Akurasi klasifikasi (apakah rating pada genre favorit): %.2f%%\n', acc*100);

% Confusion matrix
confMat = confusionmat(ytest, double(ypred));
disp('Confusion matrix (baris = label asli, kolom = prediksi):');
disp(confMat);

% Hitung precision, recall, F1 untuk kelas "1" (genre favorit)
TP = confMat(2,2);
FP = confMat(1,2);
FN = confMat(2,1);

precision = TP / max(TP+FP, 1);
recall    = TP / max(TP+FN, 1);
f1        = 2 * precision * recall / max(precision+recall, 1e-9);

fprintf('Precision (genre favorit): %.3f\n', precision);
fprintf('Recall    (genre favorit): %.3f\n', recall);
fprintf('F1-score  (genre favorit): %.3f\n', f1);

fprintf('\nInterpretasi singkat:\n');
fprintf('- Jika akurasi & F1 tinggi -> rating-user sangat konsisten dengan genre favoritnya.\n');
fprintf('- Jika akurasi mendekati 50%% -> genre favorit bukan satu-satunya faktor, ada faktor lain (kualitas film, hype, dsb).\n');

