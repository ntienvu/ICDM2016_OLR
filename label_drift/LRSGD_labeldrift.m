function model = LRSGD_labeldrift(model,yyTrain,xxTrain,yyTest,xxTest)
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
[NTrain, dd]=size(xxTrain); % feature dimension

if isempty(model.Elapse)    
    model.nTrain=0;
    model.KK=0;
    model.label_name = [];
    model.label_map = [];
    model.TrainTime=0;
    model.ww=zeros(dd, model.KK); % hyperplanes dd x (KK-1)
end

model.nTrain = model.nTrain + NTrain;
new_label = setdiff(unique(yyTrain), model.label_name);
for i=new_label
    model.KK = model.KK+1;
    model.label_name = [model.label_name; i];
    model.label_map(i) = model.KK;
    model.ww(:, end+1) = zeros(dd, 1);
end
yyTrain = model.label_map(yyTrain);

NTest=length(yyTest); % number of testing instance
NEpoch =1; % number of rounds traversing the entire dataset
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
        expArr= xn*model.ww; %1 x (KK-1)
        expArr =exp(expArr-max(expArr)); %1x(KK-1)
        
        Z = sum(expArr); %nomarlized factor
        for k=1:model.KK-1
           p = expArr ./ Z; %1x(KK-1)
           errl = (iif(yn,k) - p(k)).*xn'; %likelihood error
           errp = bsxfun(@times,model.ww(:,k)',sigSqr); %[1xdd],[1xdd]-> [1xdd]
           model.ww(:,k) = model.ww(:,k) + rt *(errl + errp'./NTrain);
        end
    end
    logErrl=0;
    for n=1:NTrain
        xn=xxTrain(n,:);
        yn=yyTrain(n);
        expArr= xn*model.ww; %1x(KK-1)
        expArr =exp(expArr); %1x(KK-1)
        Z = 1 + sum(expArr);
        if yn<model.KK
          logErrl=logErrl-log(expArr(yn)./Z);
        else
          logErrl=logErrl-log(1./Z);
        end
    end
    lnew = logErrl+logPriorMatrix(model.ww,sigArr);
    relDiff = @(x,y) abs(x-y)./(abs(x)+abs(y));

    if relDiff(lnew,lold)<eps 
        break;
    end
end

elapse=toc;

%% calculate loss on Train set
%outcome=SigmoidFunction(xxTrain*ww);
% outcome=xxTrain*model.ww;
% [~, predlabelTrain]=max(outcome,[],2);
% AccuracyTrain=length(find(predlabelTrain==yyTrain))*100/NTrain;
%% calculate loss on Test set
%outcome=SigmoidFunction(xxTest*ww);
outcome=xxTest*model.ww;
[~, predLabelTest]=max(outcome,[],2);
if isempty(yyTest)
    model.AccuracyTest=[];
else
    AccuracyTest=100*length(find(model.label_name(predLabelTest)==yyTest))/NTest;
end

model.TrainTime=model.TrainTime+elapse;
model.AccuracyTest=[model.AccuracyTest, AccuracyTest];
model.elapse=elapse;

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
