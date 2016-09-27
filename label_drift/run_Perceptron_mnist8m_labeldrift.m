%% load stored files and run
addpath('baselines/libsvm');
addpath('baselines/libol_v0.3.0');
addpath('label_drift');

% load test data
[yyTest, xxTest] = libsvmread('data/mnist8m/mnist.scale.t');
xxTest = [xxTest, zeros(size(xxTest, 1), 784-size(xxTest, 2))];
xxTest = full(xxTest);
yyTest = yyTest + 1;

path = 'data/mnist8m/mnist8m_block';

nBlock=10;

%init
KK=0;
dd=784;
model.KK=KK;
model.nTrain=0;
model.TrainTime=0;
model.W = zeros(KK,dd);
model.Sigma = 1*eye(dd);
model.eta=0.7;
model.C=0.9999999999;
model.t=1;
model.r=1;
model.loss_type=1;
model.phi   = 0.1;
model.alpha=0.9;
model.k_AL=1;
model.p=2;
model.task_type='mc';

OutputTime=[];
OutputAccuracy=[];

for tt=2:2
    rng(6789);
    model.label_name = [];
    model.label_map = zeros(1, 10);
    model.W = zeros(KK,dd);
    model.W0 = zeros(1,dd);
    model.w = zeros(KK,dd);
    model.count=0;
    Elapse=[];
    Accuracy=[];
    for ii=1:nBlock
        mytime=tic;
        str=sprintf('%s/block%d',path,ii-1);
        disp(str);
%         load(str);
        [yyblock, xxblock] = libsvmread(str);
        xxblock = [xxblock, zeros(size(xxblock, 1), 784-size(xxblock, 2))];
        xxblock = full(xxblock);
        yyblock = yyblock + 1;
        idx = randperm(length(yyblock));
        xxblock = xxblock(idx, :);
        yyblock = yyblock(idx);
        
        new_label = setdiff(unique(yyblock), model.label_name);
        for i=new_label
            model.KK = model.KK+1;
            model.label_name = [model.label_name, i];
            model.label_map(i) = model.KK;
            model.W(end+1,:) = zeros(1, dd);
        end
        yyblock = model.label_map(yyblock);
    
        for uu=1:length(yyblock)
            model=Perceptron_labeldrift(yyblock(uu)',full(xxblock(uu,:) ),model);
        end
        [pred_Y, AccuracyTest] = libol_predict(yyTest,full(xxTest),model);
        ttt=toc(mytime);
        Elapse=[Elapse ttt];
        Accuracy=[Accuracy AccuracyTest*100];
        fprintf('Perceptron:\tBlock %d \tTotal Elapse=%.2f (secs)\t Accuracy=%.2f (%%)\n',ii,sum(Elapse),Accuracy(end));
    end
    OutputModel{tt}.Elapse=Elapse;
    OutputModel{tt}.AccuracyTest=Accuracy;
end

save('stored_model/Perceptron_mnist8m_labeldrift.mat', 'OutputModel');
