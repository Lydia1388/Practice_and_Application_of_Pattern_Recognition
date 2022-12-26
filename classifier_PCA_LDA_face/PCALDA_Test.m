function [test_result, num_pred_correct, total_test_sample, error_predict]=PCALDA_Test(num_test_sample, people, PCA_project, LDA_project, LDA_result, TotalMeanFACE)

TestFFACE = [];
test_start = 2;
TestFFACE = LoadData(people, test_start);
total_train_sample = people * num_test_sample;
total_test_sample = people * num_test_sample;

num_pred_correct = 0;
zeromeanTestFFACE = TestFFACE - TotalMeanFACE;
error_predict = ["sample", "predict", "label"];
test_result = [];
for i = 1 : total_test_sample
    euclid_dis = zeros(total_train_sample, 1);  
    result = zeromeanTestFFACE(i, :) * PCA_project * LDA_project;
    test_result = [test_result; result];
    min_dis = Inf;
    for j = 1 : total_train_sample
        tmp = LDA_result(j, :) - result;
        euclid_dis = tmp * tmp';
        if euclid_dis < min_dis
            min_dis = euclid_dis;
            train_index = j;
        end
    end

    mod_test = mod(i, num_test_sample);
    mod_predict = mod(train_index, num_test_sample);
    label_predict = floor(train_index/num_test_sample);
    label_test = floor(i/num_test_sample);

    if mod_test == 0
        if mod_predict == 0
            label_predict;
        else
            label_predict = label_predict + 1;
        end
        label_test;
    else
        if mod_predict == 0
            label_predict;
        else
            label_predict = label_predict + 1;
        end
        label_test = label_test + 1;
    end

    if label_predict == label_test
        num_pred_correct = num_pred_correct + 1;
    else
        error_predict = [error_predict; i, label_predict, label_test];
    end 
end
