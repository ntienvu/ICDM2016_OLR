clear all;
clear all;
warning off;

addpath(genpath('../'));

[yyTrain, xxTrain]=libsvmread('data\mnist');
[yyTest, xxTest]=libsvmread('data\mnist.t');
   
if isempty(yyTrain) || isempty(yyTest)
    disp('please download the training data from http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/mnist.bz2');
    disp('please download the testing data from http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/mnist.t.bz2');
    disp('and copy it into ../data/');
    return;
    return;
end

% load('MNIST_Train.mat');
% load('MNIST_Test.mat');

%% Onepass Logistic Regression
disp(' ');
disp('====================Onepass Logistic Regression===================');
OutputOLR = OnepassLogisticRegression(yyTrain,xxTrain,yyTest,xxTest);

fprintf('Onepass Logistic Regression: \tTrain Accuracy=%.2f \tTest Accuracy=%.2f \tElapse=%.2f\n',...
    OutputOLR.AccuracyTrain,OutputOLR.AccuracyTest,OutputOLR.elapse);

%% Logistic Regression using MLE
disp(' ');
disp('===============Logistic Regression Maximum Likelihood==============');
OutputLR_MLE=LogisticRegressionUsingMLE(yyTrain,xxTrain,yyTest,xxTest);

fprintf('Logistic Regression MLE: \tTrain Accuracy=%.2f \tTest Accuracy=%.2f \tElapse=%.2f\n',...
    OutputLR_MLE.AccuracyTrain,OutputLR_MLE.AccuracyTest,OutputLR_MLE.elapse);

%% Logistic Regression using Laplacian approximation
disp(' ');
disp('=====================Logistic Regression Laplacian=================');
OutputLR_Laplacian = LogisticRegressionUsingLaplacianApproximation( yyTrain,xxTrain,yyTest,xxTest );

fprintf('Logistic Regression Laplacian: \tTrain Accuracy=%.2f \tTest Accuracy=%.2f \tElapse=%.2f\n',...
    OutputLR_Laplacian.AccuracyTrain,OutputLR_Laplacian.AccuracyTest,OutputLR_Laplacian.elapse);


%% Logistic Regression using Polya Gamma approximation
disp(' ');
disp('====================Logistic Regression PolyaGamma=================');
OutputLR_PG = LogisticRegressionUsingPolyaGamma( yyTrain,xxTrain,yyTest,xxTest );

fprintf('Logistic Regression PolyaGamma: \tTrain Accuracy=%.2f \tTest Accuracy=%.2f \tElapse=%.2f\n',...
    OutputLR_PG.AccuracyTrain,OutputLR_PG.AccuracyTest,OutputLR_PG.elapse);


%% Logistic Regression using SGD
disp(' ');
disp('====================Logistic Regression SGD=================');
OutputLR_SGD = LogisticRegression_SGD( yyTrain,xxTrain,yyTest,xxTest );

fprintf('Logistic Regression SGD: \tTrain Accuracy=%.2f \tTest Accuracy=%.2f \tElapse=%.2f\n',...
    OutputLR_SGD.AccuracyTrain,OutputLR_SGD.AccuracyTest,OutputLR_SGD.elapse);



%% LibLinear SVM
disp(' ');
disp('=========================LibLinear SVM===========================');
OutputLibLinear=ExperimentLibLinearSVM(yyTrain,xxTrain,yyTest,xxTest);

fprintf('SVM: \tTrain Accuracy=%.2f \tTest Accuracy=%.2f \tElapse=%.2f\n',...
    OutputLibLinear.AccuracyTrain,OutputLibLinear.AccuracyTest,OutputLibLinear.elapse);

%% knn
% KNN is slow as it takes ~6500 seconds to finish.
% tic
% disp(' ');
% disp('==============================KNN================================');
% mdl=fitcknn(xxTrain,yyTrain);
% % evaluate trainning
% predTrainLabel=predict(mdl,xxTrain);
% idxCorrectTrain=find(yyTrain==predTrainLabel);
% OutputKNN.AccuracyTrain=100*length(idxCorrectTrain)/length(yyTrain);
% % evaluate testing
% predTestLabel=predict(mdl,xxTest);
% idxCorrectTest=find(yyTest==predTestLabel);
% OutputKNN.AccuracyTest=100*length(idxCorrectTest)/length(yyTest);
% OutputKNN.elapse=toc;
% fprintf('KNN: \tTrain Accuracy=%.2f \tTest Accuracy=%.2f \tElapse=%.2f\n',...
%     OutputKNN.AccuracyTrain,OutputKNN.AccuracyTest,OutputKNN.elapse);

OutputKNN.AccuracyTest=96.9;
OutputKNN.elapse=6547;

%% Linear Discriminant Analysis
tic
disp(' ');
disp('==================Linear Discriminant Analysis===================');
mdl = ClassificationDiscriminant.fit(xxTrain,yyTrain,'discrimType','pseudoLinear');
% evaluate trainning
[predTrainLabel, NewMPGCI] = predict(mdl,xxTrain);
idxCorrectTrain=find(yyTrain==predTrainLabel);
OutputLDA.AccuracyTrain=100*length(idxCorrectTrain)/length(yyTrain);
% evaluate testing
[predTestLabel, NewMPGCI] = predict(mdl,xxTest);
idxCorrectTest=find(yyTest==predTestLabel);
OutputLDA.AccuracyTest=100*length(idxCorrectTest)/length(yyTest);
OutputLDA.elapse=toc;
fprintf('Linear Discriminant Analysis: \tTrain Acc=%.2f \tTest Acc=%.2f \tElapse=%.2f\n',...
    OutputLDA.AccuracyTrain,OutputLDA.AccuracyTest,OutputLDA.elapse);

%% ClassificationTree
tic
disp(' ');
disp('==========================Decision Tree==========================');
clt = ClassificationTree.fit(xxTrain,yyTrain);
% evaluate trainning
predTrainLabel = predict(clt,xxTrain);
idxCorrectTrain=find(yyTrain==predTrainLabel);
OutputDT.AccuracyTrain=100*length(idxCorrectTrain)/length(yyTrain);
% evaluate testing
predTestLabel = predict(clt,xxTest);
idxCorrectTest=find(yyTest==predTestLabel);
OutputDT.AccuracyTest=100*length(idxCorrectTest)/length(yyTest);
OutputDT.elapse=toc;
fprintf('Classification Tree: \tTrain Acc=%.2f \tTest Acc=%.2f \tElapse=%.2f\n',...
    OutputDT.AccuracyTrain,OutputDT.AccuracyTest,OutputDT.elapse);

%% NaiveBayes
tic;
disp(' ');
disp('=======================Training NaiveBayes=======================');
%mdl=NaiveBayes.fit(xxTrain,yyTrain);
%this data set is getting error for Gaussian NaiveBayes, then we use
%NaiveBayes Multinomial instead.
mdl=NaiveBayes.fit(round(xxTrain*100),yyTrain,'Distribution','mn');
% evaluate trainning
predTrainLabel = predict(mdl,round(xxTrain*100));
idxCorrectTrain=find(yyTrain==predTrainLabel);
OutputNB.AccuracyTrain=100*length(idxCorrectTrain)/length(yyTrain);
% evaluate testing
OutputNB.predTestLabel = predict(mdl,round(xxTest*100));
idxCorrectTest=find(yyTest==OutputNB.predTestLabel);
OutputNB.AccuracyTest=100*length(idxCorrectTest)/length(yyTest);
OutputNB.elapse=toc;
fprintf('Naive Bayes: \tTrain Acc=%.2f \tTest Acc=%.2f \tElapse=%.2f\n',...
    OutputNB.AccuracyTrain,OutputNB.AccuracyTest,OutputNB.elapse);

%% plot using Quadrant Visualization (Matlab 2014a)
TimeScale=320;

if (exist('OutputNB','var')==1 && exist('OutputDT','var')==1 && exist('OutputLDA','var')==1 && exist('OutputLibLinear','var')==1 ...
        && exist('OutputLR_MLE','var')==1 && exist('OutputOLR','var')==1 ...
     && exist('OutputLR_Laplacian','var')==1 && exist('OutputLR_PG','var')==1 )
    AccuracyTime=[OutputNB.AccuracyTest OutputNB.elapse; ...
    OutputDT.AccuracyTest OutputDT.elapse; ...
    OutputKNN.AccuracyTest OutputKNN.elapse; ...
    OutputLDA.AccuracyTest OutputLDA.elapse;...
    OutputLibLinear.AccuracyTest OutputLibLinear.elapse;...
    OutputLR_MLE.AccuracyTest OutputLR_MLE.elapse;...
    OutputLR_SGD.AccuracyTest OutputLR_SGD.elapse;...
    OutputLR_Laplacian.AccuracyTest OutputLR_Laplacian.elapse;...    
    OutputLR_PG.AccuracyTest OutputLR_PG.elapse;...
    OutputOLR.AccuracyTest OutputOLR.elapse];
else
    AccuracyTime=[83.61 6.05; 87.8 60.64; 96.9 6547; 87.3 29.58; 91.7 126.7; 91.74 298;...
         85.94 28.22;91.86 4021; 84.65 391.93; 86.47 1.59];
end

MethodNameList={'NB','DT','kNN','LDA','SVM','LR-CGD','LR-SGD','BLR-LAP','BLR-PG','OLR'};
QuadrantVisualization(AccuracyTime,MethodNameList,'MNIST: Classification',TimeScale);