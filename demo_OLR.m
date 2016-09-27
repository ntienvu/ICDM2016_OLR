clear all;
warning off;
addpath(genpath(pwd));

%% demo DNA dataset

[yyTrain, xxTrain]=libsvmread('data\dna.scale');
[yyTest, xxTest]=libsvmread('data\dna.scale.t');
   
if isempty(yyTrain) || isempty(yyTest)
    disp('cannot load data');
    return;
end

yyTrain=uint8(categorical(yyTrain));
yyTest=uint8(categorical(yyTest));
xxTrain=full(xxTrain);
xxTest=full(xxTest);

disp('=================================================================');
disp('DNA dataset');
disp('=================Onepass Logistic Regression================');
OutputOLR = OnepassLogisticRegression(yyTrain,xxTrain,yyTest,xxTest)

fprintf('Onepass Logistic Regression: \tTrain Accuracy=%.2f \tTest Accuracy=%.2f \tElapse=%.2f\n',...
    OutputOLR.AccuracyTrain,OutputOLR.AccuracyTest,OutputOLR.elapse);

%% demo KDD dataset
clear all;

load('data\kdd1999.mat');
NN=length(GroundTruth);

rng(6789);
idx=randperm(NN);
idxTrain=idx(1:ceil(0.8*NN));
idxTest=idx(ceil(0.8*NN)+1:end);

xxTrain=Feature(idxTrain,:);
yyTrain=GroundTruth(idxTrain);

xxTest=Feature(idxTest,:);
yyTest=GroundTruth(idxTest,:);

disp('=================================================================');
disp('KDD dataset');
disp('=================Onepass Logistic Regression================');
OutputOLR = OnepassLogisticRegression(yyTrain,xxTrain,yyTest,xxTest)

fprintf('Onepass Logistic Regression: \tTrain Accuracy=%.2f \tTest Accuracy=%.2f \tElapse=%.2f\n',...
    OutputOLR.AccuracyTrain,OutputOLR.AccuracyTest,OutputOLR.elapse);