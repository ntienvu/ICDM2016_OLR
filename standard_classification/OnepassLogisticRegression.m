function Output = OnepassLogisticRegression(yyTrain,xxTrain,yyTest,xxTest)
%  input ===================================================================
%   yyTrain: label of trainning data [NTrain x 1]
%   xxTrain: feature of training data [NTrain x dd]
%   yyTest: label of testing data [NTest x 1]
%   xxTest: feature of testing data [NTest x dd]
%   =========================================================================
%   output ==================================================================
%   Output.AccuracyTrain = classification accuracy on training set
%   Output.AccuracyTest  = classification accuracy on testing set
%   Output.predLabelTest = predicted label : [NTest x 1]

tic;
NTrain=length(yyTrain); % number of training instance
NTest=length(yyTest); % number of testing instance
KK=length(unique(yyTrain)); % number of classes
dd=size(xxTrain,2); % feature dimension

ww=ones(dd,KK); % hyperplanes
QQ=zeros(dd,KK); % feature over classes

%% computing lambda and PP are processed within each block buffer to make the memory is efficient.
PP=speye(dd);
bufferMax=100000;
if NTrain>bufferMax
    nPatch=ceil(NTrain/bufferMax);
    for ii=1:nPatch-1
        to=ii*bufferMax;
        from=(ii-1)*bufferMax+1;
        xx_patch=xxTrain(from:to,:);
        lambda_block=sample_PolyaGamma_approximation(sum(xx_patch,2));
        temp = bsxfun(@times, xx_patch, lambda_block);
        PP = PP+temp'*xx_patch;% accummulate PP
    end
    from=(nPatch-1)*bufferMax+1;
    xx_patch=xxTrain(from:end,:);
    lambda_block=sample_PolyaGamma_approximation(sum(xx_patch,2));
    temp = bsxfun(@times, xx_patch, lambda_block);
    PP = PP+temp'*xx_patch;% accummulate PP
else
    % described in Algorithm 2
    % lambda ~ PolyaGamma (X)
    lambda=sample_PolyaGamma_approximation(sum(xxTrain,2));
    % P = (X'diag(lambda)X) + I
    temp = bsxfun(@times, xxTrain, lambda);
    PP = PP+temp'*xxTrain;
end

%% compute QQ
% Qk=X*Lk
for kk=1:KK
    Lk=-1*ones(NTrain,1);
    Lk(yyTrain==kk)=1;
    QQ(:,kk)=xxTrain'*Lk;
end

%% sampling beta k
for kk=1:KK
    if dd<10000
        ww(:,kk)=PP\QQ(:,kk);
    else
        ww(:,kk)=pcg(PP,QQ(:,kk),1e-3,10);
    end
end

%% calculate loss on Train set
outcome=SigmoidFunction(xxTrain*ww);
[~, predlabelTrain]=max(outcome,[],2);
AccuracyTrain=length(find(predlabelTrain==yyTrain))*100/NTrain;
%% calculate loss on Test set
outcome=SigmoidFunction(xxTest*ww);
[~, predLabelTest]=max(outcome,[],2);
if isempty(yyTest)
    AccuracyTest=[];
else
    AccuracyTest=100*length(find(predLabelTest==yyTest))/NTest;
end
elapse=toc;

%fprintf('#Class = %d \t Accuracy on Train set = %.3f \t Test set = %.3f%% \t ElapseTime=%.3f sec\n',KK,AccuracyTrain,AccuracyTest,elapse);

Output.AccuracyTrain=AccuracyTrain;
Output.AccuracyTest=AccuracyTest;
Output.elapse=elapse;
Output.SufficientStatistic.ww=ww;
Output.SufficientStatistic.PP=PP;
Output.SufficientStatistic.KK=KK;
Output.SufficientStatistic.QQ=QQ;
Output.SufficientStatistic.predLabelTest=predLabelTest';
end

function out = SigmoidFunction( in )
%SIGMOIDFUNCTION Summary of this function goes here
    out=1./(1+exp(-in));
end

function lambda = sample_PolyaGamma_approximation(cc)
% sampling PolyaGamma distribution approximation
lambda=tanh(cc./2)./(2*cc);
end
