function [TrainFFACE, TotalMeanFACE, PCA_project, PCA_result, LDA_project, LDA_result]=PCALDA_Train(num_train_sample, people)

TrainFFACE = [];
PCA_dim_num = 50; % dim from 1024 down to 50 
LDA_dim_num = 20;
train_start = 1;
TrainFFACE = LoadData(people, train_start);

% ---------------------- start of PCA ----------------------
TotalMeanFACE = mean(TrainFFACE);
zeromeanTotalFACE = TrainFFACE;

% zero mean
zeromeanTotalFACE = zeromeanTotalFACE - TotalMeanFACE; % 正規化

covariance = zeromeanTotalFACE' * zeromeanTotalFACE;
[PCA_vec, PCA_val] = eig(covariance);
PCA_val = diag(PCA_val);

[junk, index] = sort(PCA_val, 'descend'); % 排序Eigenvalue
PCA_vec = PCA_vec(:, index);
PCA_project = PCA_vec(:, 1:PCA_dim_num); % get 1-50 dim

PCA_result = zeromeanTotalFACE * PCA_project;

% --------------------- start of LDA -----------------------
all_class_mean = [];
within = 0;
for i = 1 : people
    class = PCA_result(1+(i-1)*num_train_sample : i*num_train_sample, :);
    class_mean = mean(class);
    all_class_mean = [all_class_mean; class_mean];
    class_pri = class - class_mean;
    
    within = within + class_pri' * class_pri;
end

mean_global = mean(PCA_result);
mean_class_pri = all_class_mean - mean_global;

between = mean_class_pri' * mean_class_pri;

LDA_co = inv(within) * between;
[LDA_vec, LDA_val] = eig(LDA_co);
LDA_val = diag(LDA_val);
[junk, index] = sort(LDA_val, 'descend'); % sort Eigenvalue
LDA_vec = LDA_vec(:, index);
LDA_project = LDA_vec(:, 1:LDA_dim_num);% get 1-20 dim

LDA_result = PCA_result * LDA_project;
end
