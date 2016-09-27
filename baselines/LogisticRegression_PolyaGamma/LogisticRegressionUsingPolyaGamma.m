function Output = LogisticRegressionUsingPolyaGamma(yyTrain,xxTrain,yyTest,xxTest, varargin)
%  input ===================================================================
%   labelTrain: [size of Training Data x 1]
%   featureTrain: [size of Training Data x size of Feature]
%   labelTest: [size of Testing of Data x 1]
%   featureTest: [size of Testing of Data x size of Feature]
%   =========================================================================
%   output ==================================================================
%   Output.AccuracyTrain = classification accuracy on training set 
%   Output.AccuracyTest  = classification accuracy on testing set
%   Output.predLabelTest = predicted label : [size of Testing of Data x 1]

tic

nTrain=size(xxTrain,1);
nTest=size(xxTest,1);

%% checking consistency
nClassTrain=length(unique(yyTrain));
nClassTest=length(unique(yyTest));
% if nClassTrain~=nClassTest
%     disp('number of class in training set must be consistency with the one in testing set');
%     return;
% end

KK=nClassTrain;
dim=size(xxTrain,2);


%% iteratively sampling
omega=ones(1,nTrain);

ww=cell(1,KK);
for kk=1:KK
    ww{kk}=ones(dim,1);
end


inv_V0=1*eye(dim);

m0=zeros(dim,1);

nIteration=30;

bufferMax=100001;
if nTrain>bufferMax
    nPatch=ceil(nTrain/bufferMax);
    xxpatch=cell(1,nPatch+1); %#ok<NASGU>
    for ii=1:nPatch-1
        to=ii*bufferMax;
        from=(ii-1)*bufferMax+1;
        xx_patch{ii}=xxTrain(from:to,:);
    end
    from=(nPatch-1)*bufferMax+1;
    xx_patch{nPatch}=xxTrain(from:end,:);
end

%loop
for tt=1:nIteration
  
    %% sampling omega ik
    for ii=1:nTrain
        kk=yyTrain(ii);
        omega(ii)=sample_PolyaGamma_truncate(1,xxTrain(ii,:)*ww{kk});
    end
   
    inv_Vk=speye(dim);

    if nTrain>bufferMax
        for ii=1:nPatch-1
            to=ii*bufferMax;
            from=(ii-1)*bufferMax+1;
            temp = bsxfun(@times, xx_patch{ii}, omega(from:to)');
            inv_Vk = inv_Vk+temp'*xx_patch{ii};% accummulate PP
        end
        from=(nPatch-1)*bufferMax+1;
        temp = bsxfun(@times, xx_patch{nPatch}, omega(from:end)');
        inv_Vk = inv_Vk+temp'*xx_patch{nPatch};% accummulate PP
    else
        
        temp = bsxfun(@times, xxTrain, omega');
        inv_Vk = speye(dim)+temp'*xxTrain;
    end
    
    

    %inv_Vk=xxTrain'*diag(omega)*xxTrain'+inv_V0;
    Vk=pinv(inv_Vk);


    %% sampling beta k
    for kk=1:KK  
        idx=find(yyTrain==kk);
        kik=-1/2*ones(nTrain,1);
        kik(idx)=1/2;

        %mk=inv_Vk\(xxTrain'*kik+inv_V0*m0);
        mk=Vk*(xxTrain'*kik+inv_V0*m0);
        
        try
           temp = mvnrnd(mk,Vk);
        catch
            temp=mk';
        end
        ww{kk}=temp';
    end
end

%% calculate loss on Train set
outcome=[];
for kk=1:KK
    outcome=[outcome SigmoidFunction(xxTest*ww{kk})];
end
[~, predicted_yyTest]=max(outcome');

AccuracyTest=100*length(find(predicted_yyTest==yyTest'))/length(yyTest);

outcome=[];
for kk=1:KK
    outcome=[outcome SigmoidFunction(xxTrain*ww{kk})];
end
[~, pred_label]=max(outcome');

AccuracyTrain=length(find(pred_label==yyTrain'))*100/length(yyTrain);
    
elapse=toc;
Output.AccuracyTest=AccuracyTest;
Output.AccuracyTrain=AccuracyTrain;
Output.predLabelTest=predicted_yyTest';
Output.elapse=elapse;
end
