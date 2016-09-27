function [ model ] = NaiveBayesClassification_labeldrift( model,yyTrain,xxTrain,yyTest,xxTest)
% NaiveBayes parameter estimation using block
% model: model parameter of previous step
% xxTrain:[NTrain x dd] 
% yyTrain:[NTrain x 1] 
% xxTest: [NTest x dd]
% yyTest: [NTest x 1]

tic;

dim=size(xxTrain,2);

if isempty(model)
    model.nTrain = 0;
    model.KK = 0;
    model.label_name = [];
    model.label_map = [];
    model.sumData=zeros(model.KK,dim);
    model.squaredSumData=zeros(model.KK,dim);
    model.nDataOverK=zeros(1, model.KK);
    model.TrainTime=0;
end

model.nTrain = model.nTrain + length(yyTrain);
new_label = setdiff(unique(yyTrain), model.label_name);
for i=new_label
    model.KK = model.KK+1;
    model.label_name = [model.label_name; i];
    model.label_map(i) = model.KK;
    model.sumData(end+1, :) = zeros(1, dim);
    model.squaredSumData(end+1, :) = zeros(1, dim);
    model.nDataOverK = [model.nDataOverK, 0];
    
end
yyTrain = model.label_map(yyTrain);
model.nDataOverK = model.nDataOverK + histc(yyTrain, 1:model.KK);

% prior
prior=model.nDataOverK./sum(model.nDataOverK);
% prior=prior';

distr='normal';

switch distr
    case 'normal'
        % normal distribution
        % parameters from training set
        for kk=1:model.KK
            idx=find(yyTrain==kk);
            xi=xxTrain(idx,:);
            model.sumData(kk,:)=model.sumData(kk,:)+sum(xi);
            mymu(kk,:)=model.sumData(kk,:)/model.nDataOverK(kk);
            model.squaredSumData(kk,:)=model.squaredSumData(kk,:)+sum(xi.*xi);
            mysigma(kk,:)=sqrt(model.nDataOverK(kk)*model.squaredSumData(kk,:)-model.sumData(kk,:).^2)/model.nDataOverK(kk);            
        end
        tttTrain=toc;       
end

if nargin>3
    % probability for test set
    nTest=length(yyTest);
    P=zeros(nTest,model.KK);%prior
    for kk=1:model.KK
        for dd=1:dim
            if  (mysigma(kk,dd)==0) || (isnan(mysigma(kk,dd))==1) || (mysigma(kk,dd)<0.00001)
            else
                temp=normpdf(xxTest(:,dd),mymu(kk,dd),mysigma(kk,dd));%likelihood
                temp(temp==0)=0.000001;% smooth
                P(:,kk)=P(:,kk)+log(temp);
            end
        end
    end
    
    P=P+ones(nTest,1)*log(prior);
    
    % get predicted output for test set
    [val,predLabel]=max(P,[],2);
    
    accTest=100*sum(model.label_name(predLabel)==yyTest)/length(yyTest);
    model.AccuracyTest=accTest;

end
model.predLabelTest=predLabel;
model.TrainTime=model.TrainTime+tttTrain;
end
