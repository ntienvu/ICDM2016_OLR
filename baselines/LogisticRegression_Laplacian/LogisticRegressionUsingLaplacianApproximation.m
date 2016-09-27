function Output = LogisticRegressionUsingLaplacianApproximation( yyTrain,xxTrain,yyTest,xxTest )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

tic 


KK=length(unique(yyTrain));
NN=length(yyTrain);
NNTest=length(yyTest);
%% training
for kk=1:KK
    tempyyTrain=zeros(NN,1);
    tempyyTrain(yyTrain==kk)=1;
    model{kk}=LogisticRegrLaplacianApproxBinary(tempyyTrain,xxTrain);
    
end

%% prediction on testset
outcome=[];
for kk=1:KK
    outcome=[outcome SigmoidFunction(xxTest*model{kk}.ww)];
end
[~, predicted_yyTest]=max(outcome');

AccuracyTest=100*length(find(predicted_yyTest==yyTest'))/length(yyTest);

%% prediction on trainset
outcome=[];
for kk=1:KK
    outcome=[outcome SigmoidFunction(xxTrain*model{kk}.ww)];
end
[~, predicted_yyTrain]=max(outcome');

AccuracyTrain=100*length(find(predicted_yyTrain==yyTrain'))/length(yyTrain);

elapse=toc;
Output.AccuracyTest=AccuracyTest;
Output.AccuracyTrain=AccuracyTrain;

Output.elapse=elapse;
end

