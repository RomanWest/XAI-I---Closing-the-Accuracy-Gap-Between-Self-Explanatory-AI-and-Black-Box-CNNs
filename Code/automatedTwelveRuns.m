cd D:\Theo\XAInI\Scripts\AllNetworkTraining
max_Runs = 12; %max_Runs = 1;
runs = 1;
set = 1;

for x=1:max_Runs
    validation_Sets = {[1,2,3] [4,5,6] [7,8,9] [10,11,12] [1,4,7] [5,8,10] [2,9,11] [3,6,12] [1,6,9] [2,7,10] [3,8,11] [4,5,12]};
    for i=validation_Sets
        training_Tokens = [1,2,3,4,5,6,7,8,9,10,11,12];
        validation_Tokens = i{:};
        for j=validation_Tokens
            if ismember(j, training_Tokens)
                training_Tokens(training_Tokens == j) = [];   
            end
        end
        disp("Set " + set);
        disp("Run number " + runs + " of " + max_Runs);
        disp("Using validation image numbers " + strjoin(string(validation_Tokens)));
        cd D:\Theo\XAInI\Scripts\AllNetworkTraining
        networksAll
        clearvars -except runs max_Runs set;
        runs = runs + 1;
        if runs > max_Runs
            runs = 1;
            set = set+1;
        else
        end
    end
    if set > max_Runs
        exit()
    else
    end
end
