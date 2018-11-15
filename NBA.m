%% NBA.m
%
% Tool for identifying principle components of NBA player stats
% for 2D visualization purposes and anomaly detection to identify
% players with anomalous stats.
%
% This is still very much a work in progress! The 2D PCA visualization
% appears to do a nice job of separating star players from the rest,
% with the x-axis capturing offensive skill and the y-axis capturing defense.
%
% Anomaly detection identifies both above and below average players.
%


%% Initialization
clear; close all; clc


% ======= load data =======
X = load('NBA_stats.csv');

% get player info
nplayers = size(X,1);
playerList = loadPlayerList(nplayers);
players_train = playerList(X(:,end) == 1,:);
players_cv = playerList(X(:,end) == 2,:);
players_test = playerList(X(:,end) == 3,:);

% import variable names
fid = fopen('NBA_vars.txt');
C = textscan(fid, '%s', 'CollectOutput', true);
fclose(fid);
varList = C{1};

%% ======= PCA to reduce data for visualization purposes =======

fprintf('\nRunning PCA to reduce dimensionality of data for visualization...\n')

% normalize features
[X_norm, mu, sigma] = featureNormalize(X);

% run PCA
[U, S] = pca(X_norm);

% project data onto K dimensions
K = 2;
Z = projectData(X_norm, U, K);
Z = -Z; % changes sign so that better = positive

% find players with extreme values
elite = find(Z(:,1)>=8);
Z_mu = mean(Z);
Z_norm = bsxfun(@minus, Z, Z_mu);
Z_sigma = std(Z_norm);
elite_x = find(Z_norm(:,1) > (2 * Z_sigma(:,1)));
elite_y = find(Z_norm(:,2) > (2 * Z_sigma(:,2)));
elite_both = find(Z_norm(:,1) > (1.5 * Z_sigma(:,1)) & Z_norm(:,2) > (1.5 * Z_sigma(:,2)));

% generate list of 'elite' players
fprintf('\n--- Elite on Both Dimensions ---\n\n');
for i = 1:length(elite_both)
    if elite_both(i) > 0
        idx_temp = elite_both(i);
        fprintf('%s: %s\n', playerList{idx_temp,1}, ...
              playerList{idx_temp,2});
    end
end

fprintf('\n--- Elite on Dimension 1 (~ Offense) ---\n\n');
for i = 1:length(elite_x)
    if elite_x(i) > 0
        idx_temp = elite_x(i);
        fprintf('%s: %s\n', playerList{idx_temp,1}, ...
              playerList{idx_temp,2});
    end
end

fprintf('\n--- Elite on Dimension 2 (~ Defense) ---\n\n');
for i = 1:length(elite_y)
    if elite_y(i) > 0
        idx_temp = elite_y(i);
        fprintf('%s: %s\n', playerList{idx_temp,1}, ...
              playerList{idx_temp,2});
    end
end

fprintf('\n\nPlotting player data projected onto %d dimensions...\n', K)
% plot reduced data
plot(Z(:,1),Z(:,2), 'b.', 'markersize', 5)
title('PCA: Player Stats Reduced to 2 Dimensions')
xlabel("Dimension 1: Offense?")
ylabel("Dimension 2: Defense?")

% plot identified "elite" players in different colors
hold on
p1 = plot(Z(elite_x,1), Z(elite_x,2), 'g.');
p2 = plot(Z(elite_y,1), Z(elite_y,2), 'c.');
p3 = plot(Z(elite_both,1), Z(elite_both,2),'m.');

% circle all-stars
allstar = [playerList{:,3}]'; %'
allstar = str2double(allstar);
allstars = find(allstar==1);
p4 = plot(Z(allstars,1), Z(allstars,2),'ro', 'markersize', 5);
h = legend('Non-elite','D1 elite','D2 elite','D1/D2 elite','All-star');
set(h,'Interpreter','none')

xlimits = xlim();
ylimits = ylim();
p5 = plot([xlimits(1), xlimits(2)], [0, 0]);
p6 = plot([0, 0], [ylimits(1), ylimits(2)]);

fprintf('Program paused. Press any key to continue.\n');
pause;

% find specific player on plot by name
search = 1;
while search == 1
  fprintf('\nTo see where a player falls on this plot, enter his name (Firstname Lastname) below.\nCareful, spelling and capitalization matter!\n\n');
  prompt = 'Which player would you like to see? ';
  player_name = input(prompt, 's');
  player_idx = find(strcmp({playerList{:,1}}, player_name));
  if isempty(player_idx)
    search_again = input('Sorry, could not find this player. Try again? (y/n) ', 's');
  else
    p_temp = plot(Z(player_idx,1), Z(player_idx,2),'r*', 'markersize', 10, 'LineWidth', 1);
    search_again = input('Do you want to see another player? (y/n) ', 's');
    delete(p_temp);
  end
  if strcmp(search_again, 'n')
    search = 0;
  end
end
hold off


%% ======= Anomaly Detection =======
fprintf('\n\n\nProceeding to anomaly detection... \n')

% choose variables that might indicate anomalies
X_AD = [X(:,4) X(:,7:8) X(:,10) X(:,16:17) X(:,19:20) X(:,25) X(:,26:28) X(:,38) X(:,40:42)];
varList_AD = {};
varList_AD(1,1) = varList(4);
varList_AD(2,1) = varList(7);
varList_AD(3,1) = varList(8);
varList_AD(4,1) = varList(10);
varList_AD(5,1) = varList(16);
varList_AD(6,1) = varList(17);
varList_AD(7,1) = varList(19);
varList_AD(8,1) = varList(20);
varList_AD(9,1) = varList(25);
varList_AD(10,1) = varList(26);
varList_AD(11,1) = varList(27);
varList_AD(12,1) = varList(28);
varList_AD(13,1) = varList(38);
varList_AD(14,1) = varList(40);


% transform variables for normality
X_AD(:,1) = (X_AD(:,1).^(1/3));
X_AD(:,3) = (sqrt(X_AD(:, 3)));
X_AD(:,4) = (sqrt(X_AD(:,4)));
X_AD(:,5) = (X_AD(:,5).^(1/3));
X_AD(:,6) = (X_AD(:,6).^(1/3));
X_AD(:,7) = (X_AD(:,7).^(1/3));
X_AD(:,8) = (X_AD(:,8).^(1/3));
X_AD(:,10) = (log(X_AD(:,10)));

% divide data into training, CV, and test sets
X_train = X_AD(X_AD(:,end) == 1,:);
X_cv = X_AD(X_AD(:,end) == 2,:);
X_test= X_AD(X_AD(:,end) == 3,:);

% remove grouping data
X_AD = X_AD(:,1:end-1);
X_train = X_train(:,1:end-1);
X_cv = X_cv(:,1:end-1);
X_test = X_test(:,1:end-1);

% move last column (all-star status) from X to y
y = X_AD(:,end);
y_train = X_train(:,end);
y_cv = X_cv(:,end);
y_test = X_test(:,end);
X_AD = X_AD(:,1:end-1);
X_train = X_train(:,1:end-1);
X_cv = X_cv(:,1:end-1);
X_test = X_test(:,1:end-1);

% create subsets with traditional and advanced stats
X_traditional = X_AD(:, 1:9);
X_advanced = [X_AD(:, 1) X_AD(:, 10:14)];


% estimate parameters of Gaussian distribution
[mu sigma2] = estimateGaussian(X_train);

% Density estimation (training)
p = multivariateGaussian(X_train, mu, sigma2);

% Density estimation (validation)
p_cv = multivariateGaussian(X_cv, mu, sigma2);

% choose threshold
[epsilon F1] = selectThreshold(y_cv, p_cv);

% run on test set
p_test = multivariateGaussian(X_test, mu, sigma2);

% find outliers
outliers_train = find(p < epsilon);
outliers_cv = find(p_cv < epsilon);
outliers_test = find(p_test < epsilon);

% display lists of players from CV and test sets IDed as outliers
fprintf('\n\n--- Validation Set Outliers:\n\n');
for i = 1:length(outliers_cv)
    if outliers_cv(i) > 0
        idx_temp = outliers_cv(i);
        fprintf('%s: %s\n', players_cv{idx_temp,1}, ...
              players_cv{idx_temp,2});
    end
end

fprintf('\n\n--- Test Set Outliers:\n\n');
for i = 1:length(outliers_test)
    if outliers_test(i) > 0
        idx_temp = outliers_test(i);
        fprintf('%s: %s\n', players_test{idx_temp,1}, ...
              players_test{idx_temp,2});
    end
end


fprintf('\n\nProgram paused. Press any key to continue.\n');
pause;

% plot 2 variables to see correlation and anomalies
fprintf('\n\n\n--- Choose two variables to plot in relation to ''anomaly'' status:\n\n')
% display variable list for users reference
var_num = size(varList_AD,1);
for j = 1:var_num
  fprintf('%d: %s\n', j, varList_AD{j});
end
fprintf('\n');

plot_more = 1;
while plot_more == 1
  x_prompt = sprintf('Enter index of first variable to plot on x-axis (1-%d): ',var_num);
  y_prompt = sprintf('Enter index of second variable to plot on y-axis (1-%d): ',var_num);
  xvar = input(x_prompt);
  yvar = input(y_prompt);
  if xvar > var_num | yvar > var_num
    search_again = input('Sorry, variable index out of range. Try again? (y/n) ', 's');
  else
    p5 = plot(X_test(:, xvar), X_test(:, yvar),'b.');
    hold on
    p6 = plot(X_test(outliers_test, xvar), X_test(outliers_test, yvar),'ro');
    title('Relation of selected variables to ''anomaly'' status');
    xlabel(varList_AD{xvar});
    ylabel(varList_AD{yvar});
    plot_again = input('Do you want to create another plot? (y/n) ', 's');
    hold off
  end
  if strcmp(plot_again, 'y')
  else
    plot_more = 0;
  end
end

close all;


% ======== Logistic Regression =========
fprintf('\n\n\nProceeding to logistic regression... \n')

X_LR = X(:,1:end-2);
% randomly assign examples into training, CV, test sets
sets = X(:,end)(randperm(size(X(:,end),1)),:);
X_LR_train = X_LR(sets == 1,:);
y_LR_train = X(sets == 1,end-1);
X_LR_cv = X_LR(sets == 2,:);
y_LR_cv = X(sets == 2,end-1);
X_LR_test= X_LR(sets == 3,:);
y_LR_test = X(sets == 3,end-1);


% Initialize fitting parameters
initial_theta = zeros(size(X_LR_train, 2), 1);

% Set value for regularization term
lambda = 1;

% Set Options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X_LR_train, y_LR_train, lambda)), initial_theta, options);

fprintf('\nPredictions of all-star status: \n\n')

% Compute accuracy on training set
p = predict(theta, X_LR_train);
fprintf('Train Accuracy: %f\n', mean(double(p == y_LR_train)) * 100);

p_cv = predict(theta, X_LR_cv);
fprintf('CV Accuracy: %f\n', mean(double(p_cv == y_LR_cv)) * 100);

p_test = predict(theta, X_LR_test);
fprintf('Test Accuracy: %f\n', mean(double(p_test == y_LR_test)) * 100);

%
