function [ Output ] = ExperimentLibLinearSVM( labTrain, instTrain, labTest, instTest, varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

labTrain=double(labTrain);
labTest=double(labTest);
if ~issparse(instTrain)
    instTrain=sparse(instTrain);
end

if ~issparse(instTest)
    instTest=sparse(instTest);
end

if isrow(labTrain)
    labTrain=labTrain';
end

if isrow(labTest)
    labTest=labTest';
end

tic

%skew data
%[IsDefault, IsPlot, command] = process_options(varargin,'IsDefault',1,'IsPlot',0,'command','');

%cmd = ['-t 0 -q '];
cmd = ['-q'];
%model=ovrtrain_liblinear(labTrain,instTrain,cmd);
model=train(labTrain,instTrain,cmd);

trainTime=toc;
tic;


labelSet = unique(labTrain);
[predict_labelTrain, TrainAccuracy, decision_values] = predict(labTrain,instTrain,model);
[predict_labelTest, TestAccuracy, decision_values] = predict(labTest,instTest,model);


testTime=toc;

Output.AccuracyTrain=TrainAccuracy(1);
Output.AccuracyTest=TestAccuracy(1);
Output.predLabelTest=predict_labelTest;
Output.elapse=trainTime+testTime;
%fprintf('LibLinear Accuracy=%.3f%%\n',TestAccuracy*100);
end

