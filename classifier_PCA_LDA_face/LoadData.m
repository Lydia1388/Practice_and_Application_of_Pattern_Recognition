function [FFACE] = LoadData(people, data_start)
FFACE=[];
for k=1:1:people
    % get 1,3,5,7,9 sample from 10 samples for train 
    % or get 2,4,6,8,10 sample for test      
    for m = data_start:2:10
        matchstring = ['ORL3232' '\' num2str(k) '\' num2str(m) '.bmp'];
        matchX = imread(matchstring);
        matchX = double(matchX); % from int to float
        if (k == 1 && m == data_start)
            [row,col] = size(matchX);
        end
        matchtempF = [];
        %--arrange the 32X32 image into a 1X1024 vector
        for n = 1:row
            matchtempF = [matchtempF ,matchX(n,:)];
        end
        FFACE = [FFACE; matchtempF]; 
    end
end % end of k=1:1:people