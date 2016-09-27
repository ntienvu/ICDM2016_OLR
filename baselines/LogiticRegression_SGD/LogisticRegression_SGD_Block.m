function Output = LogisticRegression_SGD_Block(model,yyTrain,xxTrain,yyTest,xxTest)
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
%   =======================================================================
%   Problem to solve: max {LCL - 1/2m*||w||^2} LCL: log conditional
%   likelihood

tic;
dd=size(xxTrain,2); % feature dimension

if isempty(model)
    
    nTrain=0;
    KK=length(unique(yyTrain));
 
    model.TrainTime=0;
    ww=zeros(dd,KK-1); % hyperplanes dd x (KK-1)

else
    ww=model.ww;
    nTrain=model.nTrain+length(yyTrain);
    KK=model.KK;
 
    maxClass=max(yyTrain);
    if maxClass>KK
        moreClass=maxClass-KK;
        KK=maxClass;
        for kk=1:moreClass
            ww(:,end+1)=zeros(dd,1);
        end
    end
end

NTrain=length(yyTrain); % number of training instance
NTest=length(yyTest); % number of testing instance
KK=length(unique(yyTrain)); % number of classes
NEpoch =10; % number of rounds traversing the entire dataset
r0=1; %initial learning rate
delta = 1; %annealing rate
sigArr = ones(1,dd); %Gauss prior is used 1xdd
%sigArr=std(xxTrain);
sigSqr = -1./(sigArr.*sigArr); %1xdd
lnew =0;
eps = 0.01;

for i=1:NEpoch %traverse entire dataset NEpoch times
    rt = r0./(1+i./delta);
    lold = lnew;
    for n=1:NTrain %traverse the training set
        xn = xxTrain(n,:);  %the current sample 1xdd
        yn = yyTrain(n);  %the current label 1x1
        expArr= xn*ww; %1 x (KK-1)
        expArr =exp(expArr-max(expArr)); %1x(KK-1)
        
        Z = sum(expArr); %nomarlized factor
        for k=1:KK-1
           p = expArr ./ Z; %1x(KK-1)
           errl = (iif(yn,k) - p(k)).*xn'; %likelihood error
           errp = bsxfun(@times,ww(:,k)',sigSqr); %[1xdd],[1xdd]-> [1xdd]
           ww(:,k) = ww(:,k) + rt *(errl + errp'./NTrain);
        end
    end
    logErrl=0;
    for n=1:NTrain
        xn=xxTrain(n,:);
        yn=yyTrain(n);
        expArr= xn*ww; %1x(KK-1)
        expArr =exp(expArr); %1x(KK-1)
        Z = 1 + sum(expArr);
        if yn<KK
          logErrl=logErrl-log(expArr(yn)./Z);
        else
          logErrl=logErrl-log(1./Z);
        end
    end
    lnew = logErrl+logPriorMatrix(ww,sigArr);
    relDiff = @(x,y) abs(x-y)./(abs(x)+abs(y));

    if relDiff(lnew,lold)<eps 
        break;
    end
end

ww(:,KK) = zeros(dd,1);
%% calculate loss on Train set
%outcome=SigmoidFunction(xxTrain*ww);
outcome=xxTrain*ww;
[~, predlabelTrain]=max(outcome,[],2);
AccuracyTrain=length(find(predlabelTrain==yyTrain))*100/NTrain;
%% calculate loss on Test set
%outcome=SigmoidFunction(xxTest*ww);
outcome=xxTest*ww;
[~, predLabelTest]=max(outcome,[],2);
if isempty(yyTest)
    AccuracyTest=[];
else
    AccuracyTest=100*length(find(predLabelTest==yyTest))/NTest;
end

elapse=toc;

Output.TrainTime=model.TrainTime+elapse;
Output.KK=KK;
Output.nTrain=nTrain;
Output.AccuracyTrain=AccuracyTrain;
Output.AccuracyTest=AccuracyTest;
Output.elapse=elapse;
Output.ww=ww;

end


function lp= logPriorVec(w,sigArr)    %for all dimensions of w
 logPriorOneDim = @(beta,sig) 0.5.*log(2.*pi)+log(sig) +beta.^2./(2.*sig.^2);  %for one dimension

 logArr = bsxfun(logPriorOneDim,w,sigArr);
 lp = sum(logArr); 
end

function lp= logPriorMatrix(ww,sigArr)    %for all vectors in a matrix
 lp =0;
 d= size(ww,2);
 for i=1:d
     lp = lp+logPriorVec(ww(:,i),sigArr);
 end
end

function b=iif(y,c)
   if(y==c)
       b=1;
   else
       b=0;
   end
end
