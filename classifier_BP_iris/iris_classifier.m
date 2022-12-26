function [accuracy, false_pred, num_pred_correct, RMSE, sigma_out, weight_hid, weight_out]=iris_classifier

%----------------- Load data -----------------
load iris_in.csv;
load iris_out.csv;
input = iris_in;
output = iris_out;
%------------------- Train ------------------
num_hidden = 12;
weight_hid = rand(num_hidden, 4);
weight_out = rand(1, num_hidden);
epoch = 1000;
RMSE = zeros(epoch, 1);
learning_rate = 0.4;
for e = 1 : epoch
    error_square = [];
    for i = 1 : 75
        % 前傳
        train_data = input(i,:);
        target = output(i);
        sigma_hid = train_data * weight_hid';
        hidden_net = logsig(sigma_hid);
        sigma_out = hidden_net * weight_out';
        output_net = purelin(sigma_out);
        % 倒傳
        error = target - output_net;
        delta_out = error * dpurelin(output_net);
        error_square = [error_square; error.^2];
        % update weight of hidden layer
        for j = 1 : num_hidden
            delta_hid = delta_out * weight_out(j) * dlogsig(sigma_hid(j), hidden_net(j));
            weight_hid(j,:) = weight_hid(j,:) + (delta_hid * train_data) * learning_rate;
        end
        % update weight of output layer
        weight_out = weight_out + (delta_out * hidden_net) * learning_rate;
    end
    RMSE(e) = sqrt(sum(error_square) / 75);
    fprintf('Epoch: %.0f, RMSE: %.3f\n', e, RMSE(e))
end

plot(1:epoch, RMSE(1:epoch));
legend('Training');
xlabel('Epoch');
ylabel('RMSE');

%------------------- Test ------------------
num_pred_correct = 0;
false_pred = ["data", "target", "predict class", "output"];
for i = 76 : 150
    test_data = input(i,:);
    target = output(i);
    sigma_out = logsig(test_data * weight_hid') * weight_out';

    if sigma_out >= 0.5 && sigma_out < 1.5
        pred_class = 1;
    elseif sigma_out >= 1.5 && sigma_out < 2.5
        pred_class = 2;
    elseif sigma_out >= 2.5 && sigma_out < 3.5
        pred_class = 3;
    end
    % compute TP for accuracy
    if pred_class == target
        num_pred_correct = num_pred_correct + 1;
    else
        false_pred = [false_pred; i, target, pred_class, sigma_out];
    end
end
%--------------------------------------------
accuracy = num_pred_correct / 75;