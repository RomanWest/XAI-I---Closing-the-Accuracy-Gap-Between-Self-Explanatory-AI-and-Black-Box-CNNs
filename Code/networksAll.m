%% This script returns Shallow Network, C2 and C4 Networks

% Shallow Network = classification using our expert ratings
% C2 = expert features with neural network, 
% C4 = a neural network using the categories

% This script does not adjust the error of the networks prediction if it is
% equal to -1. This is done in cases where there is no appropriate value for
% a dimension.
%%
cd D:\Theo\XAInI\Scripts\AllNetworkTraining %("D:\Theo\XAInI\Scripts\Rule based classification")%C:\Users\Admin\Desktop\XAInI\Scripts\Rule based classification
categoriesToUse = [5,6,7,9,13,15,20,22,24,29];%Perhaps would be good to not hard code which indexes to use for training and test.
%%
%training_Tokens = [1,2,3,4,5,6,7,8,9];%[1,2,3,6,7,8,9,10,11];%still need to pass this to get tables
%validation_Tokens = [10,11,12];%[4,5,12];

% Validation sets to use:
%([1,2,3] ; [4,5, 6] ; [7,8, 9] ; [10,11,12]) ;
%([1,4,7] ; [5,8,10] ; [2,9,11] ; [ 3, 6,12]) ;
%([1,6,9] ; [2,7,10] ; [3,8,11] ; [ 4, 5,12]) ;
%%
% pool = parpool;
% pool.NumWorkers
%%
% if acts_needed
[X_train_expert,X_test_expert,X_train_additional] = loadExpertFeatures(training_Tokens);
[totalTrain,totalTrainLabels,valFeaturePatches,totalValLabels,nr_of_train_images_in_orignal,valTableLabels] = getResnetFeatures(X_train_expert,X_test_expert,X_train_additional,training_Tokens,validation_Tokens,categoriesToUse);%TODO: extract resnet 2048 features for nosofsky original and nosofsky additional images.
% end
%net = trainUnconstrained(categoriesToUse,totalTrain);%Temporary
[dlnet,networkExpertPredictions,validationPredictions] = trainDeepNetwork(totalTrain,totalTrainLabels,valFeaturePatches);%Image -> expert features

networkExpertPrediciton256Weights = dlnet.Layers(3).Weights;
networkExpertPrediciton256Bias = dlnet.Layers(3).Bias;
networkExpertPrediciton13Weights = dlnet.Layers(5).Weights;
networkExpertPrediciton13Bias = dlnet.Layers(5).Bias;

saveas(gcf, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_DeepNetwork_"+datestr(now,'mm-dd-yyyy_HH-MM')+".fig");
delete(findall(0));

nr_imgs_val=30;

dims_predicted = [];
true_labels = repelem(categoriesToUse,3);

rockNamesTen = ["Granite","Obsidian","Pegmatite","Pumice","Gneiss","Marble","Slate","Breccia","Conglomerate","Sandstone"];

%%
%% Shallow network only (using expert features):
% netShallow = trainShallowExpert(categoriesToUse,totalTrainLabels,13,totalValLabels,valTableLabels); % For shallow network only network predictions - replace number for quantity of features e.g. 9 or 13
% 
% currentfig = findall(groot, 'Tag', 'NNET_CNN_TRAININGPLOT_UIFIGURE');
% savefig(currentfig, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_ShallowFig_"+datestr(now,'mm-dd-yyyy_HH-MM')+".fig");
% exportapp(currentfig,"D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_ShallowFig_"+datestr(now,'mm-dd-yyyy_HH-MM')+".jpg");
% delete(findall(0));
% 
% predicted_labels_shallow = [];
% confidenceShallow = [];
% for i = 1:nr_imgs_val
% p = totalValLabels(:,(i-1)*320+1+nr_imgs_val:i*320+nr_imgs_val);% Expert rated features for the patches
% p = gather(sum(p,2)/320);
% p = predict(netShallow,p(1:13)');%use feature-> category network to predict the category of a validation image
% confidenceShallow = [confidenceShallow; "Rock Number "+i,"","","","","","","","",""; rockNamesTen; p];
% [~,p] = max(p);
% predicted_labels_shallow = [predicted_labels_shallow,categoriesToUse(p)];
% end
% disp("Using shallow network classifier on expert features")
% disp(sum(predicted_labels_shallow==true_labels))
% accuracy_Shallow = sum(predicted_labels_shallow==true_labels)/30*100;
% netShallowMatrix = confusionmat(true_labels, predicted_labels_shallow);
% netShallowWeights = netShallow.Layers(2).Weights;
% netShallowBias = netShallow.Layers(2).Bias;
% writematrix(confidenceShallow, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(Shallow Network - Confidence of Prediciton)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");

%% C2:
% Find best rule via a neural network only on expert features.
% Use the first 9/10 features which are also used by experts.

% 10 Features:
% [netC2,info] = trainShallowNetwork(categoriesToUse,networkExpertPredictions,10,validationPredictions,valTableLabels);

% 9 Features:
% netC2 = trainShallowNetwork(categoriesToUse,networkExpertPredictions,9,validationPredictions,valTableLabels);

% 13 Features:
[netC2,info] = trainShallowNetwork(categoriesToUse,networkExpertPredictions,13,validationPredictions,valTableLabels);

% 12 Features:
%[netC2,info] = trainShallowNetwork(categoriesToUse,networkExpertPredictions,12,validationPredictions,valTableLabels);

currentfig = findall(groot, 'Tag', 'NNET_CNN_TRAININGPLOT_UIFIGURE');
savefig(currentfig, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_C2Fig_"+datestr(now,'mm-dd-yyyy_HH-MM')+".fig");
exportapp(currentfig,"D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_C2Fig_"+datestr(now,'mm-dd-yyyy_HH-MM')+".jpg");
delete(findall(0));

predicted_labels = [];
predicted_features = [];
expert_val_features = X_test_expert';
confidenceC2 = [];
for i = 1:nr_imgs_val
imageActs = valFeaturePatches(:,(i-1)*320+1+nr_imgs_val:i*320+nr_imgs_val);%gets resnet features for 320 validation images
p = predict(dlnet,dlarray(imageActs,"CB"));%use feature prediction network to predict image patch features for a validation image // "C" (channel), "B" (batch)
p = gather(extractdata(sum(p,2)/320));%average 320 patch predictions.
predicted_features = [predicted_features, p];
%%%%%%%% Change to number of features %%%%%%%%
p2 = p(1:13)';
p3 = predict(netC2,p2);% use feature-> category network to predict the category of a validation image % options for number of features ..... p = predict(netC2,p(1:10)'); p = predict(netC2,p(1:13)'
confidenceC2 = [confidenceC2, p3']; % added ' to p3 to transpose when saving %[confidenceC2; "Rock Number "+i,"","","","","","","","",""; rockNamesTen; p]
% explainerShapley = shapley(netC2,p2); %explainerShapley = shapley(blackboxFcn,predictorData);
[~,p4] = max(p3);
predicted_labels = [predicted_labels,categoriesToUse(p4)];%categoriesToUse(p)
end
disp("Using C2 network classifier on expert features")
disp(sum(predicted_labels==true_labels))
accuracy_C2 = sum(predicted_labels==true_labels)/30*100;
netC2Matrix = confusionmat(true_labels, predicted_labels);
netC2Weights = netC2.Layers(2).Weights;
netC2Bias = netC2.Layers(2).Bias;
writematrix(netC2Weights, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(C2-Weights)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");
writematrix(netC2Bias, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(C2-Bias)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");
writematrix(confidenceC2, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(C2-ConfidencePred)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");
writematrix(predicted_features', "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(C2-PredFeatures)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");
%% Save the validation rock with predicated feature, expert feature and
% calculated RMSE, and total RMSE of all features - export as a csv

% 13 features
feature_Names = ["Average Grainsize"; "Roughness"; "Presence of Foliation"; "Presence of Banding"; "Heterogeneity of Grainsize C";...
                "Lightness of Color"; "Heterogeneity of Hue"; "Heterogeneity of Brightness"; "Volume of Vesicles"; "Glasslike texture";...
                "Angular Clasts"; "Rounded Clasts"; "Presence of Crystals"];
% 12 features - no brightness
%feature_Names = ["Average Grainsize"; "Roughness"; "Presence of Foliation"; "Presence of Banding"; "Heterogeneity of Grainsize C";...
%                "Lightness of Color"; "Heterogeneity of Hue"; "Volume of Vesicles"; "Glasslike texture";...
%                "Angular Clasts"; "Rounded Clasts"; "Presence of Crystals"];


% 9 baseline features
% feature_Names = ["Grainsize C"; "Presence of Foliation"; "Presence of Banding"; "Heterogeneity of Grainsize C";...
%                 "Presence of Crystals"; "Heterogeneity of colour C";  "Volume of Vesicles"; "Glasslike texture";...
%                  "Rounded/Angular Clasts"; ];
errorSquare_features = [];
rock_feature_comparison = [];
RMSE = [];
total_Error = [];

for i = 1:width(predicted_features) % From 1:30
    single_rock_pred = predicted_features(:, i); 
    single_rock_expert = expert_val_features(:, i);
    
%     if a rock has an expert rating of -1 replace the predicted and expert
%     rating with a 0 so that the number is not summed
    for j = 1:height(single_rock_pred)
        if single_rock_expert(j) == -1
            single_rock_expert(j) = 0;
            single_rock_pred(j) = single_rock_pred(j);
        else 
            single_rock_expert(j) = single_rock_expert(j);
            single_rock_pred(j) = single_rock_pred(j);
        end
        errorSquare_feature = (single_rock_pred(j)-single_rock_expert(j))^2;
        errorSquare_features = [errorSquare_features; errorSquare_feature];   
    end
    rock_feature_comparison = [rock_feature_comparison;"Rock Number "+i,"","","";"Feature","Prediction","Expert Rating","Error Squared";feature_Names, single_rock_pred, single_rock_expert, errorSquare_features];
    total_Error = [total_Error, errorSquare_features];
    errorSquare_features = [];
end
total_Error = sum(total_Error,2);

% Rounded/Angular Clasts is only relevant to 6 rock images
% Adjust if feature order changes
for k = 1:height(total_Error)
%      if k == 9 % Rounded/Angular Clasts is the 9th feature
%         
%            RMSE = [RMSE; realsqrt(total_Error(k)/6);];
%     else
           RMSE = [RMSE; realsqrt(total_Error(k)/30);];  
%     end
end

RMSE = ["Feature", "RMSE"; feature_Names, RMSE];

% Write the results to a CSV
writematrix(RMSE, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(Total_RMSE_of_Features)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");

writematrix(rock_feature_comparison, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(1-30_FeaturePred_ExpertRating_ErrorSquared)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");

%% Get the performance using an unconstrained classifier
% disp("Training unconstrained classifier")
% netC4 = trainUnconstrained(categoriesToUse,totalTrain);
% predicted_labels = [];
% confidenceC4 = [];
% for i = 1:nr_imgs_val
% imageActs = valFeaturePatches(:,(i-1)*320+1+nr_imgs_val:i*320+nr_imgs_val);
% p = predict(netC4,imageActs')';
% % confidenceC4 = [confidenceC4; "Rock Number "+i,"","","","","","","","",""; rockNamesTen; p];
% [~,p] = max(sum(p,2));
% predicted_labels = [predicted_labels,categoriesToUse(p)];
% end
% disp("Using the unconstrained learning approach: ")
% disp(sum(predicted_labels==true_labels))
% accuracy_C4 = sum(predicted_labels==true_labels)/30*100;
% netC4Matrix = confusionmat(true_labels, predicted_labels);
% netC4Weights = netC4.Layers(5).Weights;
% netC4Bias = netC4.Layers(5).Bias;
% writematrix(confidenceC4, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(C4 Network - Confidence of Prediciton)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");

%% Train hybrid network with pre-set weights and biases from network expert networks
disp("Training hybrid network")
netHybrid = trainUnconstrainedHybrid(categoriesToUse,totalTrain, networkExpertPrediciton256Weights, networkExpertPrediciton256Bias, networkExpertPrediciton13Weights, networkExpertPrediciton13Bias, netC2Weights, netC2Bias, networkExpertPredictions);

currentfig = findall(groot, 'Tag', 'NNET_CNN_TRAININGPLOT_UIFIGURE');
savefig(currentfig, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_netHybrid_"+datestr(now,'mm-dd-yyyy_HH-MM')+".fig");
exportapp(currentfig,"D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_netHybrid_"+datestr(now,'mm-dd-yyyy_HH-MM')+".jpg");
delete(findall(0));

predicted_labels = [];
confidenceNetHybrid = [];
averageActivation13Hybrid = [];
empty13x1 = strings(13,1);
for i = 1:nr_imgs_val
imageActs = valFeaturePatches(:,(i-1)*320+1+nr_imgs_val:i*320+nr_imgs_val);
p = predict(netHybrid,imageActs')';
predAllRocks = sum(p,2)/320;
act13Hybrid = activations(netHybrid,imageActs',5);
average_act13Hybrid = gather(sum(act13Hybrid,2)/320);
averageActivation13Hybrid = [averageActivation13Hybrid, average_act13Hybrid]; %[averageActivation13Hybrid; "Rock Number "+i; average_act13Hybrid];
confidenceNetHybrid = [confidenceNetHybrid, predAllRocks]; %[confidenceNetHybrid; "Rock Number "+i,"","","","","","","","",""; rockNamesTen; p];
[~,p] = max(sum(p,2));
predicted_labels = [predicted_labels,categoriesToUse(p)];
end
disp("Using the hybrid learning approach: ")
disp(sum(predicted_labels==true_labels))
accuracy_netHybrid = sum(predicted_labels==true_labels)/30*100;
netHybridMatrix = confusionmat(true_labels, predicted_labels);
netHybridWeights = netHybrid.Layers(6).Weights;
netHybridBias = netHybrid.Layers(6).Bias;
writematrix(netHybridWeights, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(netHybrid-Weights)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");
writematrix(netHybridBias, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(netHybrid-Bias)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");
writematrix(confidenceNetHybrid, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(netHybrid-ConfidencePred)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");
writematrix(averageActivation13Hybrid', "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(netHybrid-Av13NodeActs)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");

%% Get the performance using an unconstrained classifier
disp("Training unconstrained 13 classifier")
netUnconstrained_13 = trainUnconstrained_13(categoriesToUse,totalTrain, networkExpertPredictions);
predicted_labels = [];
confidence_unconstrained_13 = [];
averageActivation13Unconstrained = [];
for i = 1:nr_imgs_val
imageActs = valFeaturePatches(:,(i-1)*320+1+nr_imgs_val:i*320+nr_imgs_val);
p = predict(netUnconstrained_13,imageActs')';
act13Unconstrained = activations(netUnconstrained_13,imageActs',5);
average_act13Unconstrained = gather(sum(act13Unconstrained,2)/320);
averageActivation13Unconstrained = [averageActivation13Unconstrained, average_act13Unconstrained]; %[averageActivation13Unconstrained; "Rock Number "+i; average_act13Unconstrained];
predAllRocks = sum(p,2)/320;
confidence_unconstrained_13 = [confidence_unconstrained_13, predAllRocks];
[~,p] = max(sum(p,2));
predicted_labels = [predicted_labels,categoriesToUse(p)];
end
disp("Using the unconstrained 13 learning approach: ")
disp(sum(predicted_labels==true_labels))
accuracy_Unconstrained_13 = sum(predicted_labels==true_labels)/30*100;
Unconstrained_13Matrix = confusionmat(true_labels, predicted_labels);
Unconstrained_13Weights = netUnconstrained_13.Layers(6).Weights;
Unconstrained_13Bias = netUnconstrained_13.Layers(6).Bias;
writematrix(averageActivation13Unconstrained', "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(netUnconstrained_13 - Average 13 Node Activations of 13x256 matrix)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");
writematrix(confidence_unconstrained_13, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(Unconstrained 13 - Confidence of Prediciton)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");
writematrix(Unconstrained_13Weights, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(netHybrid-Weights)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");
writematrix(Unconstrained_13Bias, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(netHybrid-Bias)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");

%% Data Visualisation Confusion Matrix
% The 10 rock names/classes that we are using. Stored in correct order
rockNamesTen = ["Granite","Obsidian","Pegmatite","Pumice","Gneiss","Marble","Slate","Breccia","Conglomerate","Sandstone"];
rockNamesTen = categorical(rockNamesTen,'Ordinal',true);
% Build confusion matrix 
% figure
% c2MatChart = confusionchart(netC2Matrix, rockNamesTen, 'Title', 'C2', 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
% c4MatChart = confusionchart(c4, rockNamesTen,'Title', 'C4 with Validation Images 1:3','RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');

%% Save all confusion matrix as csv

rockNamesTen = ["Granite","Obsidian","Pegmatite","Pumice","Gneiss","Marble","Slate","Breccia","Conglomerate","Sandstone"];
shallow_name = ["Shallow Expert Network","Predicted","","","","","","","",""];
unconstrained_name = ["Unconstrained Deep Network","Predicted","","","","","","","",""];
constrained_name = ["Constrained Expert Network","Predicted","","","","","","","",""];
hybrid_name = ["Hybrid Network","Predicted","","","","","","","",""];
Unconstrained_13_name = ["Unconstrained_13","Predicted","","","","","","","",""];

%Accuracy_Shallow_StrAr = ["Accuracy:", accuracy_Shallow,"","","","","","","",""];
%Accuracy_C4_StrAr = ["Accuracy:", accuracy_C4,"","","","","","","",""];
Accuracy_C2_StrAr = ["Accuracy:", accuracy_C2,"","","","","","","",""];
Accuracy_netHybrid_StrAr = ["Accuracy:", accuracy_netHybrid,"","","","","","","",""];
Accuracy_Unconstrained_13_StrAr = ["Accuracy:", accuracy_Unconstrained_13,"","","","","","","",""];

%names_ShallowMat = [shallow_name; rockNamesTen; netShallowMatrix; Accuracy_Shallow_StrAr];
%names_C4Mat = [unconstrained_name; rockNamesTen; netC4Matrix; Accuracy_C4_StrAr];
names_C2Mat = [constrained_name; rockNamesTen; netC2Matrix; Accuracy_C2_StrAr];
names_netHybrid = [hybrid_name; rockNamesTen; netHybridMatrix; Accuracy_netHybrid_StrAr];
names_Unconstrained_13 = [Unconstrained_13_name; rockNamesTen; Unconstrained_13Matrix; Accuracy_Unconstrained_13_StrAr];

%all_Mat = [names_ShallowMat, names_C4Mat, names_C2Mat, names_netHybrid, names_Unconstrained_13];
% all_Mat = [names_C2Mat, names_netHybrid];
all_Mat = [names_C2Mat, names_netHybrid,names_Unconstrained_13];
writematrix(all_Mat, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(Confusion_Matrix_Shallow_Unconstrained_Constrained_Hybrid_Unconstrained 13)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");
% writematrix(names_Unconstrained_13, "D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_(Confusion_Matrix_Unconstrained_13)_"+datestr(now,'mm-dd-yyyy_HH-MM')+".csv");


%% Get the ResNet50 activations for rock features
function [totalTrain,totalTrainLabels,valFeaturePatches,totalValLabels, nr_of_train_images_in_orignal,valTableLabels] = getResnetFeatures(X_train,X_test,X_train_additional,training_Tokens,validation_Tokens,categoriesToUse)

%this function assumes a 9/3 split
i=1;
net = resnet50();
featureLayer = "avg_pool";
excludeCategories = setdiff(1:30,categoriesToUse);%[5,6,7,9,13,15,20,22,24,29]
cd D:\Theo\XAInI\Scripts\AllNetworkTraining %D:\Theo\XAInI\Scripts\TrainNetwork..;%\TrainNetwork
parentDir = pwd;%C:\Users\Admin\Desktop\XAInI\Scripts\TrainNetwork

%Get training/test/external test set activations
cd(parentDir)
rng(i);
disp("Building trainTable");
[trainTable,valTable,testTable,externalTestTable] = getTables([length(training_Tokens),length(validation_Tokens),0],excludeCategories,"D:\Theo\XAInI\Images\All_Processed","D:\Theo\XAInI\Images\testImages","D:\Theo\XAInI\Images\ExternalTest",false,training_Tokens,validation_Tokens);

cd(parentDir)
nr_of_train_images_in_orignal = [1:size(trainTable, 1)]; % 1 : 28890
disp("Building trainTableExternal");
[trainTableExternal,~,~,~] = getTablesExternal([4,0,0],excludeCategories,"D:\Theo\XAInI\Images\ExtTestSetTrain","D:\Theo\XAInI\Images\testImages","D:\Theo\XAInI\Images\ExternalTest");

cd(parentDir)
rng(i);
disp("Building valTablePatches");
[valTablePatches,~,~,~] = getTables([length(training_Tokens),length(validation_Tokens),0],excludeCategories,"D:\Theo\XAInI\Images\All_Processed","D:\Theo\XAInI\Images\testImages","D:\Theo\XAInI\Images\ExternalTest",true,training_Tokens,validation_Tokens);
valTableLabels = valTablePatches.labels';

disp("Getting ResNet50 activations with trainTable");
trainFeatures = activations(net, trainTable, featureLayer,'MiniBatchSize', 256, 'OutputAs', 'columns','ExecutionEnvironment','gpu');%,'Acceleration','mex'
trainLabels = trainTable.labels';

disp("Getting ResNet50 activations with trainTableExternal");
trainFeaturesExternal = activations(net, trainTableExternal, featureLayer,'MiniBatchSize', 256, 'OutputAs', 'columns','ExecutionEnvironment','gpu');
trainLabelsExternal = trainTableExternal.labels';

disp("Getting ResNet50 activations with valTablePatches");
valFeaturePatches = activations(net, valTablePatches, featureLayer, 'MiniBatchSize', 256, 'OutputAs', 'columns','ExecutionEnvironment','gpu');

totalTrain = [trainFeatures,trainFeaturesExternal]; %trainFeatures;  ;% 


%set the labels to expert labels
%TODO APPEND THE ADDITIONAL TRAINING LABELS TO totalTrainLabels!
%totalTrainLabels = [X_train;X_train_additional];
s = size(X_train,2);%number of expert features
totalTrainLabels = [X_train;imresize(X_train, [28800,s], 'nearest');X_train_additional;imresize(X_train_additional, [28800/9*4,s], 'nearest')]';%[totalTrainLabels;imresize(X_train, [28800,s], 'nearest')]';% 
totalValLabels = X_test;
totalValLabels = [totalValLabels;imresize(totalValLabels, [28800/3,s], 'nearest') ]';%28890

end
%% Train Deep Network
function [dlnet, networkExpertPredictions, validationPredictions] = trainDeepNetwork(totalTrain,totalTrainLabels,valFeaturePatches) % totalValLabels
disp("Training deep network...");

layers = [ ...
featureInputLayer(2048,'Name',"inputLayer")
dropoutLayer(0.5,'Name',"dropout")
fullyConnectedLayer(256,'Name',"fc1")
reluLayer('Name',"relu1")
fullyConnectedLayer(size(totalTrainLabels,1),'Name',"fc2")
];

XTrain = totalTrain;
YTrain = single(totalTrainLabels)';
numClasses = size(totalTrainLabels,2);
lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);

miniBatchSize = 2048;%2048
numEpochs = 200;%200
numObservations = size(YTrain,1);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);

executionEnvironment = "auto";

plots = "training-progress";

if plots == "training-progress"
    figure
    subplot(2,1,1)
    lineAccuracyTrain = animatedline('Color',[0 0.4470 0.7410]);
    ylim([0 inf])
    ylabel("Accuracy")
    grid on
    
    subplot(2,1,2)
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
    
end

averageGrad = [];
averageSqGrad = [];
learningRate = 10e-3;
gradDecay = 0.99;
squaredGradDecay = 0.99;
epsilon=10e-8;

iteration = 1;
start = tic;


for epoch = 1:numEpochs
    % Shuffle data.
    s=size(totalTrainLabels,2);
    idx = randperm(s);%1:s;
    XTrain = XTrain(:,idx);
    YTrain = YTrain(idx,:);
    
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % Read mini-batch of data and convert the labels to dummy
        % variables.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        X = XTrain(:,idx);
        Y = YTrain(idx,:)';
        
        % Convert mini-batch of data to a dlarray.
        dlX = dlarray(single(X),'CB');
        
        % If training on a GPU, then convert data to a gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients helper function.
        [gradients,loss] = dlfeval(@modelGradients,dlnet,dlX,Y);%[gradients,loss,accuracy] = dlfeval(@modelGradients,dlnet,dlX,Y);
        
        % Update the network parameters using the adam optimizer.
        [dlnet,vel] = adamupdate(dlnet,gradients,averageGrad,averageSqGrad,iteration, ...
        learningRate,gradDecay,epsilon);
    
   
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            %addpoints(lineAccuracyTrain,iteration,double(gather(extractdata(accuracy))))
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
            
        end
    end
    %disp(loss);
end

networkExpertPredictions = gather(extractdata(predict(dlnet,dlarray(totalTrain,"CB"))));
validationPredictions = gather(extractdata(predict(dlnet,dlarray(valFeaturePatches,"CB"))));
%exportapp(networkExpertPredictions,"D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_Expert_train_Ratings_Used_From_WC_14_02_22"+datestr(now,'mm-dd-yyyy_HH-MM')+".jpg")
%exportapp(validationPredictions,"D:\Theo\XAInI\Results\Val_"+strjoin(string(validation_Tokens))+"_C2_train_Ratings_Used_From_WC_14_02_22"+datestr(now,'mm-dd-yyyy_HH-MM')+".jpg")

end
%% C2 Shallow Network - trained with network expert predictions
function [net,info] = trainShallowNetwork(categoriesToUse,networkExpertPredictions,nrExpertFeatures,validationPredictions,valTableLabels)% net = trainShallowNetwork(categoriesToUse,totalTrainLabels,nrExpertFeatures,totalValLabels,valTableLabels)% net = trainShallowNetwork(categoriesToUse,networkExpertPredictions,nrExpertFeatures,validationPredictions,valTableLabels)%Theo - Changed to totalTrain from trainFeatures.. - Henrijs - nr_of_train_images_in_orignal
disp("Training C2 shallow network...");
 if ~exist('nrExpertFeatures','var')
      nrExpertFeatures = size(networkExpertPredictions,1);% nrExpertFeatures = size(totalTrainLabels,1);%  nrExpertFeatures = size(networkExpertPredictions,1); %use all ratings by default.
  end 
    
    
layers = [
    featureInputLayer(nrExpertFeatures),
    fullyConnectedLayer(10),
    softmaxLayer,
    classificationLayer()];

lbls = repelem(categoriesToUse,9); % The 10 categories (using 9 out of the 12 images), labels repeated 9 times per category, return 90 labels
lbls = [lbls,repelem(categoriesToUse,320*9)]; % Adding the 320 augmented image labels (from 1-9, as per previous full sized images)
lbls = [lbls,repelem(categoriesToUse,4)]; % As per above, appending the additonal set of Nosofsky image labels to the array
lbls = [lbls,repelem(categoriesToUse,320*4)]; % Then add the augmented images of the additional set

valTableLabelsLocal = double(valTableLabels);
valTableLabelsCopy = zeros(size(valTableLabelsLocal));
for i=10:-1:1
    valTableLabelsCopy(valTableLabelsLocal==i)=categoriesToUse(i);
end

options = trainingOptions('adam', ...
    'MaxEpochs',200,...%200 % was temporarily 175 to prevent divergence when using binary crystals
    'MiniBatchSize', 1024,...%256 %1024 %1440%'Plots','training-progress',, ...'ExecutionEnvironment','cpu'
    'InitialLearnRate',10^-3,...%10^-3
    'Verbose',false,...
    'Plots','training-progress',...
    'ValidationData',{validationPredictions',categorical(valTableLabelsCopy)});

[net,info] = trainNetwork(networkExpertPredictions(1:nrExpertFeatures,:)',categorical(lbls),layers,options);% net = trainNetwork(totalTrainLabels(1:nrExpertFeatures,:)',categorical(lbls),layers,options);% net = trainNetwork(networkExpertPredictions(1:nrExpertFeatures,:)',categorical(lbls),layers,options);
end
%% Shallow Network - trained with expert ratings
function net = trainShallowExpert(categoriesToUse,totalTrainLabels,nrExpertFeatures,totalValLabels,valTableLabels)%net = trainShallowExpert(categoriesToUse,networkExpertPredictions,nrExpertFeatures,validationPredictions,valTableLabels)%  net = trainShallowNetwork(categoriesToUse,networkExpertPredictions,nrExpertFeatures,validationPredictions,valTableLabels)%Theo - Changed to totalTrain from trainFeatures.. - Henrijs - nr_of_train_images_in_orignal
disp("Training expert shallow only network...");
 if ~exist('nrExpertFeatures','var')
      nrExpertFeatures = size(totalTrainLabels,1);%   nrExpertFeatures = size(networkExpertPredictions,1); %use all ratings by default.
  end 
    
    
layers = [
    featureInputLayer(nrExpertFeatures),
    fullyConnectedLayer(10),
    softmaxLayer,
    classificationLayer()];

lbls = repelem(categoriesToUse,9); % The 10 categories (using 9 out of the 12 images), labels repeated 9 times per category, return 90 labels
lbls = [lbls,repelem(categoriesToUse,320*9)]; % Adding the 320 augmented image labels (from 1-9, as per previous full sized images)
lbls = [lbls,repelem(categoriesToUse,4)]; % As per above, appending the additonal set of Nosofsky image labels to the array
lbls = [lbls,repelem(categoriesToUse,320*4)]; % Then add the augmented images of the additional set

valTableLabelsLocal = double(valTableLabels);
valTableLabelsCopy = zeros(size(valTableLabelsLocal));
for i=10:-1:1
    valTableLabelsCopy(valTableLabelsLocal==i)=categoriesToUse(i);
end

options = trainingOptions('adam', ...
    'MaxEpochs',200,...%200
    'MiniBatchSize', 1024,...%1024 %1440%'Plots','training-progress',, ...'ExecutionEnvironment','cpu'
    'InitialLearnRate',10^-3,...
    'Verbose',false,...
    'Plots','training-progress',...
    'ValidationData',{totalValLabels',categorical(valTableLabelsCopy)});%'ValidationData',{totalValLabels',categorical(valTableLabelsCopy)}); % 'ValidationData',{validationPredictions',categorical(valTableLabelsCopy)});

net = trainNetwork(totalTrainLabels(1:nrExpertFeatures,:)',categorical(lbls),layers,options);% net = trainNetwork(totalTrainLabels(1:nrExpertFeatures,:)',categorical(lbls),layers,options);% net = trainNetwork(networkExpertPredictions(1:nrExpertFeatures,:)',categorical(lbls),layers,options);
end
%% Unconstrained Network
function net = trainUnconstrained(categoriesToUse,totalTrain) % net = trainUnconstrained(categoriesToUse,trainFeatures)
%take resnet features and predict a category (e.g. pumice) no expert/human
%ratings needed.
layers = [featureInputLayer(2048),
    dropoutLayer()
    fullyConnectedLayer(256),
    reluLayer(),
    %fullyConnectedLayer(8),
    fullyConnectedLayer(10),
    softmaxLayer,
    classificationLayer()];

lbls = repelem(categoriesToUse,9); % The 10 categories (using 9 out of the 12 categories), labels repeated 9 times per category = 90 labels
lbls = [lbls,repelem(categoriesToUse,320*9)]; % Adding the 320 augmented image labels (from 1-9, as per previous full sized images) = 28820 + 90 = 28890
lbls = [lbls,repelem(categoriesToUse,4)]; % As per above, appending the additonal set of Nosofsky image labels to the array = 40 + 28890 = 28930
lbls = [lbls,repelem(categoriesToUse,320*4)]; % Then add the augmented images of the additional set = 12800 + 28930 = 41730

options = trainingOptions('adam', ...
    'MaxEpochs',200,...%200
    'MiniBatchSize', 1024,...%1440%'Plots','training-progress',
    'InitialLearnRate',10^-3,...
    'VerboseFrequency',1000);%, ...'ExecutionEnvironment','cpu'
net = trainNetwork(totalTrain',categorical(lbls),layers,options); %net = trainNetwork(trainFeatures',categorical(lbls),layers,options);
end

%% Unconstrained hybrid with preset weights
function net = trainUnconstrainedHybrid(categoriesToUse,totalTrain, networkExpertPrediciton256Weights, networkExpertPrediciton256Bias, networkExpertPrediciton13Weights, networkExpertPrediciton13Bias, netC2Weights, netC2Bias, networkExpertPredictions)
disp("Training hybrid network...");
 if ~exist('nrExpertFeatures','var')
      nrExpertFeatures = size(networkExpertPredictions,1);
  end 

layers = [featureInputLayer(2048),
    dropoutLayer()
    fullyConnectedLayer(256),
    reluLayer(),
    fullyConnectedLayer(nrExpertFeatures),
    fullyConnectedLayer(10),
    softmaxLayer,
    classificationLayer()];

% The 10 categories (using 9 out of the 12 original images), labels repeated 9 times per category = 90 labels
lbls = repelem(categoriesToUse,9);

% Adding the 320 augmented image labels (from 1-9, as per previous full sized images) = 28820 + 90 = 28890
lbls = [lbls,repelem(categoriesToUse,320*9)]; 

% As per above, appending the additonal set of Nosofsky image labels to the array = 40 + 28890 = 28930
lbls = [lbls,repelem(categoriesToUse,4)];

% Then add the augmented images of the additional set = 12800 + 28930 = 41730
lbls = [lbls,repelem(categoriesToUse,320*4)]; 

options = trainingOptions('adam', ...
    'MaxEpochs',50,...
    'MiniBatchSize', 1024,...
    'InitialLearnRate',10^-3,...
    'Verbose',false,...
    'Plots','training-progress');

layers(3).Weights = networkExpertPrediciton256Weights;
layers(3).Bias = networkExpertPrediciton256Bias;
layers(5).Weights = networkExpertPrediciton13Weights;
layers(5).Bias = networkExpertPrediciton13Bias;
layers(6).Weights = netC2Weights;
layers(6).Bias = netC2Bias;

layers(3).WeightLearnRateFactor = 0;
layers(3).BiasLearnRateFactor = 0;

net = trainNetwork(totalTrain',categorical(lbls),layers,options);
end

%% Unconstrained "13" node Network
function net = trainUnconstrained_13(categoriesToUse,totalTrain, networkExpertPredictions) % net = trainUnconstrained(categoriesToUse,trainFeatures)
%take resnet features and predict a category (e.g. pumice) no expert/human
%ratings needed.
 if ~exist('nrExpertFeatures','var')
      nrExpertFeatures = size(networkExpertPredictions,1);% nrExpertFeatures = size(totalTrainLabels,1);%  nrExpertFeatures = size(networkExpertPredictions,1); %use all ratings by default.
  end 

layers = [featureInputLayer(2048),
    dropoutLayer()
    fullyConnectedLayer(256),
    reluLayer(),
    fullyConnectedLayer(nrExpertFeatures), % 12 % 13 % change to number of features 
    fullyConnectedLayer(10),
    softmaxLayer,
    classificationLayer()];

lbls = repelem(categoriesToUse,9); % The 10 categories (using 9 out of the 12 categories), labels repeated 9 times per category = 90 labels
lbls = [lbls,repelem(categoriesToUse,320*9)]; % Adding the 320 augmented image labels (from 1-9, as per previous full sized images) = 28820 + 90 = 28890
lbls = [lbls,repelem(categoriesToUse,4)]; % As per above, appending the additonal set of Nosofsky image labels to the array = 40 + 28890 = 28930
lbls = [lbls,repelem(categoriesToUse,320*4)]; % Then add the augmented images of the additional set = 12800 + 28930 = 41730

options = trainingOptions('adam', ...
    'MaxEpochs',200,... % Was tested at 175 to prevent overfitting % 200 has been standard for most tests
    'MiniBatchSize', 1024,...%1440%'Plots','training-progress',
    'InitialLearnRate',10^-3,...
    'Verbose',false,...
    'Plots','training-progress');
%     'VerboseFrequency',1000);%, ...'ExecutionEnvironment','cpu'
net = trainNetwork(totalTrain',categorical(lbls),layers,options); %net = trainNetwork(trainFeatures',categorical(lbls),layers,options);
end
%% Helper modelGradients function used when training deep network
function [gradients,loss] = modelGradients(dlnet,dlX,Y)

    dlYPred = forward(dlnet,dlX);
    
    dlYPred(Y==-1)=-1;
    
    loss = mse(dlYPred,Y);
    
    gradients = dlgradient(loss,dlnet.Learnables);

end