function [TrainFFACE, TotalMeanFACE, PCA_project, PCA_result, LDA_project, LDA_result, test_result, num_pred_correct, error_predict, accuracy]=Main

people = 40; % total 40 classes
num_train_sample = 5;
num_test_sample = 5;

% ------------------------- Train -------------------------
[TrainFFACE, TotalMeanFACE, PCA_project, PCA_result, LDA_project, LDA_result]=PCALDA_Train(num_train_sample, people);

% ------------------------- Test -------------------------
[test_result, num_pred_correct, total_test_sample, error_predict]=PCALDA_Test(num_test_sample, people, PCA_project, LDA_project, LDA_result, TotalMeanFACE);

% ------------------ Calculate accuracy -------------------
accuracy = num_pred_correct / total_test_sample;
