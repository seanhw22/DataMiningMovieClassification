%% MANOVA Analysis: Do Movie Genres Differ in Rating and Tag Count?
% This script performs a multivariate analysis to test if genres differ
% when considering both mean_rating and tag_count together.

clear; clc; close all;

%% STEP 1: LOAD AND PREPARE THE DATA

fprintf('=== STEP 1: Loading and Preparing Data ===\n');

% Load the CSV files
movies = readtable('movies.csv');
ratings = readtable('ratings.csv');
tags = readtable('tags.csv');

fprintf('Loaded %d movies, %d ratings, %d tags\n', ...
    height(movies), height(ratings), height(tags));

% --- Safer extraction of first genre (main_genre) ---
movies.main_genre = cellfun(@(s) strtok(s, '|'), movies.genres, 'UniformOutput', false);

% Remove movies with no genre or "(no genres listed)"
idx_no_genre = strcmp(movies.main_genre, '(no genres listed)') | cellfun(@isempty, movies.main_genre);
movies = movies(~idx_no_genre, :);

% --- Calculate mean_rating and n_ratings per movie (safe grouping) ---
[rg, movieIds_r] = findgroups(ratings.movieId);
mean_ratings = splitapply(@mean, ratings.rating, rg);
count_ratings = splitapply(@numel, ratings.rating, rg);
rating_stats = table(movieIds_r, mean_ratings, count_ratings, ...
    'VariableNames', {'movieId', 'mean_rating', 'n_ratings'});

% --- Calculate tag_count per movie (safe grouping) ---
[tg, movieIds_t] = findgroups(tags.movieId);
count_tags = splitapply(@numel, tags.tag, tg);
tag_stats = table(movieIds_t, count_tags, 'VariableNames', {'movieId', 'tag_count'});

% --- Merge everything together ---
movie_data = innerjoin(movies(:, {'movieId', 'title', 'main_genre'}), ...
    rating_stats, 'Keys', 'movieId');

% Left join with tags (some movies may not have tags)
movie_data = outerjoin(movie_data, tag_stats, 'Keys', 'movieId', ...
    'MergeKeys', true, 'Type', 'left');

% Replace NaN values in tag_count with 0 (movies with no tags)
movie_data.tag_count(isnan(movie_data.tag_count)) = 0;  

fprintf('After joining ratings and tags: %d movies\n', height(movie_data));

% --- Apply minimum ratings filter ---
MIN_RATINGS = 10;  % Adjust this threshold as needed
movie_data = movie_data(movie_data.n_ratings >= MIN_RATINGS, :);
fprintf('After removing movies with < %d ratings: %d movies\n', ...
    MIN_RATINGS, height(movie_data));

% Check the distribution of n_ratings
fprintf('\nDistribution of number of ratings per movie:\n');
fprintf('  Min: %d, Max: %d, Median: %.0f, Mean: %.1f\n', ...
    min(movie_data.n_ratings), max(movie_data.n_ratings), ...
    median(movie_data.n_ratings), mean(movie_data.n_ratings));

%% STEP 2: EXPLORATORY DATA ANALYSIS

fprintf('\n=== STEP 2: Exploratory Data Analysis ===\n');

% Summary statistics by genre
genre_summary = groupsummary(movie_data, 'main_genre', ...
    {'mean', 'std'}, {'mean_rating', 'tag_count'});
% groupsummary creates: main_genre, GroupCount, mean_mean_rating, std_mean_rating, 
%                        mean_tag_count, std_tag_count
genre_summary = sortrows(genre_summary, 'GroupCount', 'descend');

fprintf('\nSummary by Genre:\n');
disp(genre_summary);

% Keep only genres with sufficient sample size (at least 20 movies)
MIN_GENRE_SIZE = 20;
valid_genres = genre_summary.main_genre(genre_summary.GroupCount >= MIN_GENRE_SIZE);
movie_data = movie_data(ismember(movie_data.main_genre, valid_genres), :);
fprintf('\nKeeping %d genres with >= %d movies\n', ...
    length(valid_genres), MIN_GENRE_SIZE);
fprintf('Final dataset: %d movies\n', height(movie_data));

% Create visualizations
figure('Position', [100, 100, 1400, 900]);

% Overall histograms
subplot(2, 3, 1);
histogram(movie_data.mean_rating, 30, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Mean Rating'); ylabel('Frequency');
title('Distribution of Mean Ratings');
grid on;

subplot(2, 3, 2);
histogram(movie_data.tag_count, 50, 'FaceColor', [0.8 0.4 0.2]);
xlabel('Tag Count'); ylabel('Frequency');
title('Distribution of Tag Counts');
grid on;

% Boxplots by genre (top genres only for clarity)
top_genres = genre_summary.main_genre(1:min(8, height(genre_summary)));
data_subset = movie_data(ismember(movie_data.main_genre, top_genres), :);

subplot(2, 3, 4);
boxplot(data_subset.mean_rating, data_subset.main_genre);
ylabel('Mean Rating');
title('Mean Rating by Genre (Top 8)');
xtickangle(45);
grid on;

subplot(2, 3, 5);
boxplot(data_subset.tag_count, data_subset.main_genre);
ylabel('Tag Count');
title('Tag Count by Genre (Top 8)');
xtickangle(45);
grid on;

% Scatter plot colored by genre
subplot(2, 3, [3, 6]);
genres_list = unique(movie_data.main_genre);
colors = lines(length(genres_list));
hold on;
for i = 1:min(8, length(genres_list))
    idx = strcmp(movie_data.main_genre, genres_list{i});
    scatter(movie_data.mean_rating(idx), movie_data.tag_count(idx), ...
        30, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.5);
end
xlabel('Mean Rating'); ylabel('Tag Count');
title('Mean Rating vs Tag Count by Genre');
legend(genres_list(1:min(8, length(genres_list))), 'Location', 'best');
grid on;
hold off;

sgtitle('Exploratory Data Analysis of Movie Genres');

%% STEP 3: DATA TRANSFORMATION

fprintf('\n=== STEP 3: Data Transformation ===\n');

% Log transform tag_count (highly skewed)
movie_data.tag_count_log = log1p(movie_data.tag_count);

fprintf('Tag count statistics:\n');
fprintf('  Original: mean=%.2f, median=%.2f, max=%.0f\n', ...
    mean(movie_data.tag_count), median(movie_data.tag_count), max(movie_data.tag_count));
fprintf('  Log-transformed: mean=%.2f, median=%.2f, max=%.2f\n', ...
    mean(movie_data.tag_count_log), median(movie_data.tag_count_log), max(movie_data.tag_count_log));

% Standardize variables (optional - helps with interpretation)
movie_data.mean_rating_std = zscore(movie_data.mean_rating);
movie_data.tag_count_log_std = zscore(movie_data.tag_count_log);

% Show transformation effect
figure('Position', [100, 100, 1200, 400]);
subplot(1, 3, 1);
histogram(movie_data.tag_count, 50, 'FaceColor', [0.8 0.4 0.2]);
xlabel('Tag Count (Original)'); ylabel('Frequency');
title('Original Tag Count');

subplot(1, 3, 2);
histogram(movie_data.tag_count_log, 30, 'FaceColor', [0.4 0.7 0.4]);
xlabel('log(1 + Tag Count)'); ylabel('Frequency');
title('Log-Transformed Tag Count');

subplot(1, 3, 3);
scatter(movie_data.mean_rating, movie_data.tag_count_log, 20, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Mean Rating'); ylabel('log(1 + Tag Count)');
title('After Transformation');
grid on;

%% STEP 4: ASSUMPTION CHECKS

fprintf('\n=== STEP 4: Checking Assumptions ===\n');

% Check for independence (should be one row per movie - already satisfied)
fprintf('Independence: Each row is a unique movie ✓\n');

% Check normality per genre (Q-Q plots for top genres)
figure('Position', [100, 100, 1400, 800]);
top_genres_check = genre_summary.main_genre(1:min(6, height(genre_summary)));

for i = 1:length(top_genres_check)
    genre_data = movie_data(strcmp(movie_data.main_genre, top_genres_check{i}), :);
    
    subplot(2, 6, i);
    qqplot(genre_data.mean_rating);
    title([top_genres_check{i} ' - Rating']);
    
    subplot(2, 6, i+6);
    qqplot(genre_data.tag_count_log);
    title([top_genres_check{i} ' - Log Tags']);
end
sgtitle('Q-Q Plots: Normality Check by Genre');

% Check homogeneity of variance
fprintf('\nChecking spread (variance) across genres:\n');
for i = 1:min(6, length(top_genres_check))
    genre_data = movie_data(strcmp(movie_data.main_genre, top_genres_check{i}), :);
    fprintf('  %s: rating_std=%.3f, tag_log_std=%.3f, n=%d\n', ...
        top_genres_check{i}, std(genre_data.mean_rating), ...
        std(genre_data.tag_count_log), height(genre_data));
end

% Sample size per genre
fprintf('\nSample sizes per genre:\n');
genre_counts = groupsummary(movie_data, 'main_genre', @numel, 'movieId');
genre_counts = sortrows(genre_counts, 'GroupCount', 'descend');
disp(genre_counts);

%% STEP 5: RUN MANOVA

fprintf('\n=== STEP 5: Running MANOVA ===\n');

% First check what genres we have
fprintf('Genres in dataset before MANOVA cleaning: %d\n', length(unique(movie_data.main_genre)));
fprintf('Total movies: %d\n', height(movie_data));

% Additional cleaning: Remove genres with insufficient variance
fprintf('\nChecking for genres with insufficient variance...\n');
genre_variance = groupsummary(movie_data, 'main_genre', 'std', {'mean_rating', 'tag_count_log'});

% Debug: show variance stats
fprintf('Variance check for top genres:\n');
disp(genre_variance(1:min(5, height(genre_variance)), :));

% Use more lenient threshold
low_var_genres = genre_variance.main_genre(genre_variance.std_mean_rating < 0.001 | ...
                                            genre_variance.std_tag_count_log < 0.001 | ...
                                            isnan(genre_variance.std_mean_rating) | ...
                                            isnan(genre_variance.std_tag_count_log));
if ~isempty(low_var_genres)
    fprintf('Removing %d genres with near-zero variance:\n', length(low_var_genres));
    disp(low_var_genres);
    movie_data = movie_data(~ismember(movie_data.main_genre, low_var_genres), :);
else
    fprintf('No genres with near-zero variance found.\n');
end

fprintf('\nFinal MANOVA dataset: %d movies across %d genres\n', ...
    height(movie_data), length(unique(movie_data.main_genre)));

% Prepare data for MANOVA
Y = [movie_data.mean_rating, movie_data.tag_count_log];
genre_cat = categorical(movie_data.main_genre);

% Run MANOVA with error handling
try
    [d, p, stats] = manova1(Y, genre_cat);
    
    fprintf('\n--- MANOVA Results ---\n');
    fprintf('Multivariate Test Statistics:\n');
    % Check which fields exist in stats structure
    if isfield(stats, 'lambda')
        fprintf('  Wilks'' Lambda: %.4f\n', stats.lambda);
    end
    if isfield(stats, 'chisq')
        fprintf('  Chi-squared: %.2f\n', stats.chisq);
    end
    if isfield(stats, 'df')
        fprintf('  df: %d\n', stats.df);
    end
    fprintf('  p-value: %.4e\n', p);
    
    if p < 0.05
        fprintf('\n✓ SIGNIFICANT: Genres differ in the combination of mean_rating and tag_count (p < 0.05)\n');
        fprintf('  At least one genre is multivariatly different from others.\n');
    else
        fprintf('\n✗ NOT SIGNIFICANT: No clear multivariate difference between genres (p >= 0.05)\n');
    end
    
    manova_success = true;
    
catch ME
    fprintf('\n⚠ MANOVA failed with error: %s\n', ME.message);
    fprintf('This typically means the data has multicollinearity or singularity issues.\n');
    fprintf('We will proceed with univariate ANOVAs instead.\n');
    p = NaN;
    manova_success = false;
end

%% STEP 6: CANONICAL DISCRIMINANT ANALYSIS

fprintf('\n=== STEP 6: Canonical View (Discriminant Analysis) ===\n');

% Perform discriminant analysis to find canonical axes
try
    % Use Linear Discriminant Analysis
    MdlLinear = fitcdiscr(Y, genre_cat);
    
    % Get canonical coefficients (eigenvectors)
    [coeffs, scores] = predict(MdlLinear, Y);
    
    % For visualization, use the first two discriminant dimensions
    figure('Position', [100, 100, 1200, 500]);
    
    subplot(1, 2, 1);
    gscatter(Y(:,1), Y(:,2), genre_cat, [], [], 20);
    xlabel('Mean Rating');
    ylabel('log(1 + Tag Count)');
    title('Original Space');
    legend('Location', 'best');
    grid on;
    
    subplot(1, 2, 2);
    % Plot in canonical space (not directly available in fitcdiscr)
    % Instead, show separation by showing class predictions
    gscatter(Y(:,1), Y(:,2), coeffs, [], [], 20);
    xlabel('Mean Rating');
    ylabel('log(1 + Tag Count)');
    title('Predicted Classes (Canonical View)');
    legend('Location', 'best');
    grid on;
    
    sgtitle('Genre Separation in Original vs Canonical Space');
    
catch ME
    fprintf('Note: Full canonical analysis requires Statistics Toolbox\n');
    fprintf('Error: %s\n', ME.message);
end

%% STEP 7: UNIVARIATE FOLLOW-UP TESTS

fprintf('\n=== STEP 7: Univariate ANOVA Follow-up ===\n');

% Test mean_rating across genres
fprintf('\n--- Testing mean_rating across genres ---\n');
[p_rating, tbl_rating, stats_rating] = anova1(movie_data.mean_rating, ...
    movie_data.main_genre, 'off');
fprintf('ANOVA for mean_rating: F=%.2f, p=%.4e\n', tbl_rating{2,5}, p_rating);

if p_rating < 0.05
    fprintf('✓ SIGNIFICANT: mean_rating differs across genres\n');
    % Post-hoc pairwise comparisons
    figure('Position', [100, 100, 800, 600]);
    [c_rating, m_rating, h_rating] = multcompare(stats_rating, 'Display', 'on');
    title('Pairwise Comparisons: Mean Rating by Genre');
end

% Test tag_count_log across genres
fprintf('\n--- Testing log(tag_count) across genres ---\n');
[p_tags, tbl_tags, stats_tags] = anova1(movie_data.tag_count_log, ...
    movie_data.main_genre, 'off');
fprintf('ANOVA for log(tag_count): F=%.2f, p=%.4e\n', tbl_tags{2,5}, p_tags);

if p_tags < 0.05
    fprintf('✓ SIGNIFICANT: tag_count differs across genres\n');
    % Post-hoc pairwise comparisons
    figure('Position', [100, 100, 800, 600]);
    [c_tags, m_tags, h_tags] = multcompare(stats_tags, 'Display', 'on');
    title('Pairwise Comparisons: Log(Tag Count) by Genre');
end

%% STEP 8: EFFECT SIZES

fprintf('\n=== STEP 8: Effect Sizes ===\n');

% Calculate practical effect sizes
genre_effects = groupsummary(movie_data, 'main_genre', ...
    {'mean', 'std'}, {'mean_rating', 'tag_count', 'tag_count_log'});

fprintf('\nMean values by genre:\n');
fprintf('%-15s %10s %15s %15s\n', 'Genre', 'Rating', 'Tags (raw)', 'Tags (log)');
fprintf('%s\n', repmat('-', 1, 60));
for i = 1:height(genre_effects)
    fprintf('%-15s %10.3f %15.1f %15.3f\n', ...
        genre_effects.main_genre{i}, ...
        genre_effects.mean_mean_rating(i), ...
        genre_effects.mean_tag_count(i), ...
        genre_effects.mean_tag_count_log(i));
end

% Calculate range of differences
rating_range = max(genre_effects.mean_mean_rating) - min(genre_effects.mean_mean_rating);
tags_range_raw = max(genre_effects.mean_tag_count) - min(genre_effects.mean_tag_count);

fprintf('\n--- Effect Size Interpretation ---\n');
fprintf('Mean Rating range across genres: %.2f stars\n', rating_range);
if rating_range > 0.3
    fprintf('  → This is a NOTICEABLE difference (> 0.3 stars)\n');
else
    fprintf('  → This is a SMALL difference (< 0.3 stars)\n');
end

fprintf('Tag Count range across genres: %.1f tags (on average)\n', tags_range_raw);
fprintf('  → Genres differ by about %.0f%% in average tag count\n', ...
    100 * tags_range_raw / mean(genre_effects.mean_tag_count));

%% FINAL SUMMARY

fprintf('\n');
fprintf('==========================================================\n');
fprintf('                    FINAL SUMMARY                         \n');
fprintf('==========================================================\n');
fprintf('Research Question: Do movie genres differ in mean_rating\n');
fprintf('                   and tag_count when considered together?\n');
fprintf('\n');
fprintf('MANOVA Result: p = %.4e\n', p);
if p_rating < 0.05 && p_tags < 0.05
    fprintf('CONCLUSION: Both mean_rating AND tag_count differ significantly\n');
    fprintf('            across genres. The effect sizes are noticeable.\n');
elseif p_rating < 0.05
    fprintf('CONCLUSION: Only mean_rating differs significantly across genres.\n');
elseif p_tags < 0.05
    fprintf('CONCLUSION: Only tag_count differs significantly across genres.\n');
else
    fprintf('CONCLUSION: No significant differences found.\n');
end
fprintf('\n');
fprintf('Univariate Results:\n');
fprintf('  mean_rating: p = %.4e %s\n', p_rating, ...
    ternary(p_rating < 0.05, '(significant)', '(not significant)'));
fprintf('  tag_count:   p = %.4e %s\n', p_tags, ...
    ternary(p_tags < 0.05, '(significant)', '(not significant)'));
fprintf('\n');
fprintf('Practical Significance:\n');
fprintf('  Rating difference: %.2f stars\n', rating_range);
fprintf('  Tag difference: %.1f tags on average\n', tags_range_raw);
fprintf('==========================================================\n');

% Helper function for ternary operator
function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
