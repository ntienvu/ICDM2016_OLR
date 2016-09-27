clear all;
clear all;
warning off;

addpath(genpath('../'));

%the data is obtained from http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/covtype.scale01.bz2

[GroundTruth, Feature]=libsvmread('data\covtype.scale01');
nClass=length(unique(GroundTruth));

if isempty(GroundTruth) || isempty(Feature)
    disp('please download the data from http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/covtype.scale01.bz2');
    disp('and copy it into ../data/');
    return;
end

Feature=full(Feature);

%divide data into 90% trainning and 10% testing
trainPercent=0.9;

yyTrain=[];
yyTest=[];
xxTrain=[];
xxTest=[];
for ii=1:nClass
    idx=find(GroundTruth==ii);
    truncate=floor(trainPercent*length(idx));
    idxTrain=idx(1:truncate);
    idxTest=idx(truncate+1:end);
    yyTrain=[yyTrain; GroundTruth(idxTrain)];
    yyTest=[yyTest; GroundTruth(idxTest)];
    xxTrain=[xxTrain; Feature(idxTrain,:)];
    xxTest=[xxTest; Feature(idxTest,:)];
end
    

%% Onepass Logistic Regression
disp(' ');
disp('====================Onepass Logistic Regression==================');
OutputOLR = OnepassLogisticRegression(yyTrain,xxTrain,yyTest,xxTest);

% OutputOLR_Fourier = OnepassLogisticRegression_FourierFeatures(yyTrain,xxTrain,yyTest,xxTest);

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
disp('===================Logistic Regression PolyaGamma=================');
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

%% knn
% KNN is slow and cannot finish within 25,000 secs.
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
%mdl=NaiveBayes.fit(xxTrain,yyTrain);
% trainning Gaussian NaiveBayes for this data set is error, 
% we use Multinomial NaiveBayes instead.
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
TimeScale=90;

if (exist('OutputNB','var')==1 && exist('OutputDT','var')==1 && exist('OutputLDA','var')==1 && exist('OutputLibLinear','var')==1 ...
        && exist('OutputLR_MLE','var')==1 && exist('OutputOLR','var')==1 ...
     && exist('OutputLR_Laplacian','var')==1 && exist('OutputLR_PG','var')==1)
    AccuracyTime=[OutputNB.AccuracyTest OutputNB.elapse; ...
    OutputDT.AccuracyTest OutputDT.elapse; ...
    OutputLDA.AccuracyTest OutputLDA.elapse;...
    OutputLibLinear.AccuracyTest OutputLibLinear.elapse;...
    OutputLR_MLE.AccuracyTest OutputLR_MLE.elapse;...
    OutputLR_SGD.AccuracyTest OutputLR_SGD.elapse;...
    OutputLR_Laplacian.AccuracyTest OutputLR_Laplacian.elapse;...    
    OutputLR_PG.AccuracyTest OutputLR_PG.elapse;...
    OutputOLR.AccuracyTest OutputOLR.elapse];
else  
    AccuracyTime=[43.52 2.32; 61.6 20.96; 54.53 5.31; 55.8 78.3; 56.54 171; 52.04 115.37; 56.67 197.37...
        ;56.23 2603.52 ; 56.42 0.55 ];
end

MethodNameList={'NB','DT','LDA','SVM','LR-CGD','LR-SGD','BLR-LAP','BLR-PG','OLR'};
QuadrantVisualization(AccuracyTime,MethodNameList,'CoverType: Classification',TimeScale);
