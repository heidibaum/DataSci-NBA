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
clear ; close all; clc


% ======= load data =======
X = load('NBA_stats.csv');

% get player info
nplayers = size(X,1);
playerList = loadPlayerList(nplayers);
players_train = playerList(X(:,end) == 1,:);
players_cv = playerList(X(:,end) == 2,:);
players_test = playerList(X(:,end) == 3,:);



%% ======= PCA to reduce data for visualization purposes =======

% normalize features
[X_norm, mu, sigma] = featureNormalize(X);

% run PCA
[U, S] = pca(X_norm);

% project data onto K dimensions
K = 2;
Z = projectData(X_norm, U, K);
Z = -Z; % changes sign so that better = positive

% plot reduced data
plot(Z(:,1),Z(:,2),'bo')

% find players with extreme values
elite = find(Z(:,1)>=8);
Z_mu = mean(Z);
Z_norm = bsxfun(@minus, Z, Z_mu);
Z_sigma = std(Z_norm);
elite = find(Z_norm(:,1) >= (max(Z_norm(:,1))-(2 * Z_sigma(:,1))));
%elite = find(Z_norm(:,2) >= (max(Z_norm(:,2))-(Z_sigma(:,2))));



fprintf('\n\nElite Players:\n\n');
for i = 1:length(elite)
    if elite(i) > 0 
        idx_temp = elite(i);
        fprintf('%s: %s\n', playerList{idx_temp,1}, ...
              playerList{idx_temp,2});
    end
end

% plot identified "elite" players in different color
hold on
plot(Z(elite,1), Z(elite,2),'ro');
hold off


fprintf('Program paused. Press enter to continue.\n');
pause;



%% ======= Anomaly Detection =======

% choose variables that might indicate anomalies
X = [X(:,4) X(:,7:8) X(:,10) X(:,14:15) X(:,17) X(:,19:20) X(:,25) X(:,26:28) X(:,38) X(:,40:42)];

% transform variables for normality
X(:,1) = (X(:,1).^(1/3));
X(:,3) = (sqrt(X(:, 3)));
X(:,4) = (sqrt(X(:,4)));
X(:,5) = (X(:,5).^(1/3));
X(:,6) = (sqrt(X(:,6)));
X(:,7) = (X(:,7).^(1/3));
X(:,8) = (X(:,8).^(1/3));
X(:,9) = (X(:,9).^(1/3));


% divide data into training, CV, and test sets
X_train = X(X(:,end) == 1,:);
X_cv = X(X(:,end) == 2,:);
X_test= X(X(:,end) == 3,:);

% remove grouping data
X = X(:,1:end-1);
X_train = X_train(:,1:end-1);
X_cv = X_cv(:,1:end-1);
X_test = X_test(:,1:end-1);

% move last column (all-star status) from X to y
y = X(:,end);
y_train = X_train(:,end);
y_cv = X_cv(:,end);
y_test = X_test(:,end);
X = X(:,1:end-1);
X_train = X_train(:,1:end-1);
X_cv = X_cv(:,1:end-1);
X_test = X_test(:,1:end-1);

% create subsets with traditional and advanced stats
X_traditional = X(:, 1:9);
X_advanced = [X(:, 1) X(:, 10:15)];



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

% display lists of players from CV and test sets ID'ed as outliers
fprintf('\n\nValidation Set Outliers:\n\n');
for i = 1:length(outliers_cv)
    if outliers_cv(i) > 0 
        idx_temp = outliers_cv(i);
        fprintf('%s: %s\n', players_cv{idx_temp,1}, ...
              players_cv{idx_temp,2});
    end
end

fprintf('\n\nTest Set Outliers:\n\n');
for i = 1:length(outliers_test)
    if outliers_test(i) > 0 
        idx_temp = outliers_test(i);
        fprintf('%s: %s\n', players_test{idx_temp,1}, ...
              players_test{idx_temp,2});
    end
end


% plot 2 variables to see correlation and outliers
plot(X_test(:, 1), X_test(:, 15),'bo')
hold on
plot(X_test(outliers_test, 1), X_test(outliers_test, 15),'ro')
hold off