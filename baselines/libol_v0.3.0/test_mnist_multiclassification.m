load('data\mnist.mat');
addpath('algorithms');
xxTest=test_data.mat;
yyTest=test_data.label+1;%make sure the label is from 1 -> 10

xxTrain=train_data.mat;
yyTrain=train_data.label+1;%make sure the label is from 1 -> 10

%% init default parameter
KK=10;
dim=784;
model.W = rand(KK,dim);
model.Sigma = 1*eye(dim);
model.eta=0.7;
model.C=1;
model.t=1;
model.r=1;
model.loss_type=1;
model.phi   = 1;
model.alpha=0.9;
model.k_AL=1;
model.p=2;

model.task_type='mc';

for tt=1:100
    for ii=1:length(yyTrain)
        
        %model =  M_CW(yyTrain(ii),full(xxTrain(ii,:) ),model);
        
        model =  M_PA(yyTrain(ii),full(xxTrain(ii,:) ),model);
        %model =  M_PerceptronM(yyTrain(ii)',full(xxTrain(ii,:) ),model);
        %model =  M_AROW(yyTrain(ii),full(xxTrain(ii,:) ),model);
        %model =  M_OGD(yyTrain(ii),full(xxTrain(ii,:) ),model);
        %model =  M_ROMMA(yyTrain(ii),full(xxTrain(ii,:) ),model);
        
        if mod(ii,10000)==0 %evaluating after passing 5000 points
            [pred_Y, model.AccuracyTest] = predict(yyTest,full(xxTest),model);
        end
        
    end
end



