function Out = LogisticRegressionUsingMLE(yTrain,XTrain,yTest,XTest)
%   Summary of this function goes here
%   Detailed explanation goes here
tic;
%% Initialize parameters
%fprintf('Initializing parameters');
lambda = 0.1; % regularization parameter
numLabels = size(unique(yTrain),1); % number of labels
%fprintf('...done\n');

%% Training Logistic Regression classifier
%fprintf('Training One-vs-All Logistic Regression');

theta = LRCourseraClassifier(XTrain, yTrain, numLabels, lambda);
%fprintf('...done\n');
trainTime=toc;
tic;
%% Predict numbers 
predTrain = predictLRCoursera(theta, XTrain);
predTest = predictLRCoursera(theta, XTest);

testTime=toc;
%% Calculate Accuracy over the training data
AccTrain=mean(double(predTrain==yTrain))*100;
AccTest=mean(double(predTest==yTest))*100;

Out.AccuracyTrain=AccTrain;
Out.AccuracyTest=AccTest;
Out.predTest=predTest;
Out.elapse=trainTime+testTime;
end

