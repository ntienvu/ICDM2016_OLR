function model = OnepassLogisticRegression_labeldrift(model,yyTrain,xxTrain,yyTest,xxTest)
%  input ===================================================================
%   modelSuffStats: contain model sufficient statistic of P, Q
%   yyTrain: label of trainning data in block b[NbTrain x 1]
%   xxTrain: feature of training data in block b[NbTrain x dd]
%   yyTest: label of testing data [NTest x 1]
%   xxTest: feature of testing data [NTest x dd]
%   =========================================================================
%   output ==================================================================
%   Output.AccuracyTrain = classification accuracy on training set
%   Output.AccuracyTest  = classification accuracy on testing set
%   Output.predLabelTest = predicted label : [NTest x 1]

tic;

NbTrain = length(yyTrain);

dd=size(xxTrain,2);

if isempty(model.Elapse)
    model.PP = speye(dd);        
    model.nTrain = 0;
    model.KK = 0;
    model.QQ = zeros(dd, model.KK);
    model.Q0 = zeros(dd, 1);
    model.label_name = [];
    model.label_map = [];
    model.trainTime=0;
end

model.nTrain = model.nTrain + NbTrain;
new_label = setdiff(unique(yyTrain), model.label_name);
for i=new_label
    model.KK = model.KK+1;
    model.label_name = [model.label_name; i];
    model.label_map(i) = model.KK;
    model.QQ(:, end+1) = model.Q0;
end
yyTrain = model.label_map(yyTrain);

Lk = -0.5*ones(NbTrain, 1);
model.Q0 = model.Q0 + xxTrain'*Lk;

for kk=1:model.KK
    Lk = -0.5*ones(NbTrain, 1);
    Lk(yyTrain==kk) = 0.5;
    model.QQ(:, kk) = model.QQ(:, kk) + xxTrain'*Lk;
end

model.ww = ones(dd, model.KK);

bufferMax=100001;
if NbTrain>bufferMax
    nPatch=ceil(NbTrain/bufferMax);
    for ii=1:nPatch-1
        to=ii*bufferMax;
        afrom=(ii-1)*bufferMax+1;
        xx_patch=xxTrain(afrom:to,:);
        lambda_block = sample_PolyaGamma_approximation(sum(xx_patch,2));
        temp = bsxfun(@times, xx_patch, lambda_block);
        model.PP = model.PP+temp'*xx_patch;
    end
    afrom=(nPatch-1)*bufferMax+1;
    xx_patch=xxTrain(afrom:end,:);
    
    lambda_block=sample_PolyaGamma_approximation(sum(xx_patch,2));
    
    temp = bsxfun(@times, xx_patch, lambda_block);
    model.PP = model.PP+temp'*xx_patch;
    
else
    lambda=sample_PolyaGamma_approximation(sum(xxTrain,2));
    temp = bsxfun(@times, xxTrain, lambda);
    model.PP = model.PP+temp'*xxTrain;
end

trainTime=toc;

if nargin>3
    nTest=length(yyTest);
    for kk=1:model.KK
        model.ww(:,kk)=model.PP\model.QQ(:,kk);
    end
    
    %% calculate loss on Test set
    outcome=xxTest*model.ww;
    [~, predicted_yyTest]=max(outcome,[],2);
    AccuracyTest=100*length(find(model.label_name(predicted_yyTest)==yyTest))/nTest;
    model.AccuracyTest=AccuracyTest;
    model.predyyTest=model.label_name(predicted_yyTest);
end
model.NumFeature=dd;
model.trainTime=model.trainTime+trainTime;
end

function lambda = sample_PolyaGamma_approximation(cc)
% sampling PolyaGamma distribution approximation
lambda=tanh(cc./2)./(2*cc);

end
