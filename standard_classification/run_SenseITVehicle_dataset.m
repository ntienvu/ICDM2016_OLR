clear all;
clear all;
warning off;

addpath(genpath('../'));

%load data
%training data obtained from http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined_scale.bz2
%testing data obtained from http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined_scale.t.bz2
[yyTrain, xxTrain]=libsvmread('data\combined_scale');
[yyTest, xxTest]=libsvmread('data\combined_scale.t');
   
if isempty(yyTrain) || isempty(yyTest)
    disp('please download the training data from http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined_scale.bz2');
    disp('please download the testing data from http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined_scale.t.bz2');
    disp('and copy it into ../data/');
    return;
    return;
end

yyTrain=uint8(categorical(yyTrain));
yyTest=uint8(categorical(yyTest));
xxTrain=full(xxTrain);
xxTest=full(xxTest);

%scale data to 0-1
% minus_min = bsxfun(@minus,xxTrain,min(xxTrain));
% xxTrain = bsxfun(@rdivide,minus_min,max(xxTrain)-min(xxTrain));
% xxTrain(isnan(xxTrain))=0;
% 
% minus_min = bsxfun(@minus,xxTest,min(xxTest));
% xxTest = bsxfun(@rdivide,minus_min,max(xxTest)-min(xxTest));
% xxTest(isnan(xxTest))=0;


%% Onepass Logistic Regression
disp(' ');
disp('=================Onepass Logistic Regression================');
OutputOLR = OnepassLogisticRegression(yyTrain,xxTrain,yyTest,xxTest);

%OutputOLR_Fourier = OnepassLogisticRegression_FourierFeatures(yyTrain,xxTrain,yyTest,xxTest);

fprintf('Onepass Logistic Regression: \tTrain Accuracy=%.2f \tTest Accuracy=%.2f \tElapse=%.2f\n',...
    OutputOLR.AccuracyTrain,OutputOLR.AccuracyTest,OutputOLR.elapse);

% fprintf('Onepass Logistic Regression Fourier: \tTrain Accuracy=%.2f \tTest Accuracy=%.2f \tElapse=%.2f\n',...
%     OutputOLR_Fourier.AccuracyTrain,OutputOLR_Fourier.AccuracyTest,OutputOLR_Fourier.elapse);


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
tic
disp(' ');
disp('==============================KNN================================');
mdl=fitcknn(xxTrain,yyTrain);
% evaluate trainning
predTrainLabel=predict(mdl,xxTrain);
idxCorrectTrain=find(yyTrain==predTrainLabel);
OutputKNN.AccuracyTrain=100*length(idxCorrectTrain)/length(yyTrain);
% evaluate testing
predTestLabel=predict(mdl,xxTest);
idxCorrectTest=find(yyTest==predTestLabel);
OutputKNN.AccuracyTest=100*length(idxCorrectTest)/length(yyTest);
OutputKNN.elapse=toc;
fprintf('KNN: \tTrain Accuracy=%.2f \tTest Accuracy=%.2f \tElapse=%.2f\n',...
    OutputKNN.AccuracyTrain,OutputKNN.AccuracyTest,OutputKNN.elapse);

%% Linear Discriminant Analysis
tic
disp(' ');
disp('==================Linear Discriminant Analysis===================');
mdl = ClassificationDiscriminant.fit(xxTrain,yyTrain,'discrimType','linear');
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
mdl=NaiveBayes.fit(xxTrain,yyTrain);
% evaluate trainning
predTrainLabel = predict(mdl,xxTrain);
idxCorrectTrain=find(yyTrain==predTrainLabel);
OutputNB.AccuracyTrain=100*length(idxCorrectTrain)/length(yyTrain);
% evaluate testing
OutputNB.predTestLabel = predict(mdl,xxTest);
idxCorrectTest=find(yyTest==OutputNB.predTestLabel);
OutputNB.AccuracyTest=100*length(idxCorrectTest)/length(yyTest);
OutputNB.elapse=toc;
fprintf('Naive Bayes: \tTrain Acc=%.2f \tTest Acc=%.2f \tElapse=%.2f\n',...
    OutputNB.AccuracyTrain,OutputNB.AccuracyTest,OutputNB.elapse);

%% plot using Quadrant Visualization (Matlab 2014a)
TimeScale=70;

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
    AccuracyTime=[75.93 0.82;76.51 44.9; 76.15 780; 80.78 2; 80.79 97; 79.89 18.4;...
       79.62 11.45; 80.88 21.00; 80.38 391.77; 80.42 0.11];
end


MethodNameList={'NB','DT','kNN','LDA','SVM','LR-CGD','LR-SGD','BLR-LAP','BLR-PG','OLR'};
QuadrantVisualization(AccuracyTime,MethodNameList,'SenseIT Vehicle: Classification',TimeScale);

