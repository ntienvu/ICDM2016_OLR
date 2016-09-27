%% load stored files and run
addpath('baselines/libsvm');
addpath('label_drift');

% load test data
[yyTest, xxTest] = libsvmread('data/mnist8m/mnist.scale.t');
xxTest = [xxTest, zeros(size(xxTest, 1), 784-size(xxTest, 2))];
xxTest = full(xxTest);
yyTest = yyTest + 1;

path = 'data/mnist8m/mnist8m_block';

nBlock=10;

%% RUN OLR
OutputNB=[];
NB.Elapse=[];
NB.AccuracyTest=[];

for ii=1:nBlock
    mytime=tic;
    
    % load data
    str=sprintf('%s/block%d',path,ii-1);
    disp(str);
    [yyblock, xxblock] = libsvmread(str);
    xxblock = [xxblock, zeros(size(xxblock, 1), 784-size(xxblock, 2))];
    xxblock = full(xxblock);
    yyblock = yyblock + 1;
    idx = randperm(length(yyblock));
    xxblock = xxblock(idx, :);
    yyblock = yyblock(idx);
        
    OutputNB = NaiveBayesClassification_labeldrift(OutputNB,yyblock,xxblock,yyTest,xxTest);
    ttt=toc(mytime);
    NB.Elapse=[NB.Elapse ttt];
    NB.AccuracyTest=[NB.AccuracyTest OutputNB.AccuracyTest];
    fprintf('NaiveBayes \t Block %d \t Total Elapse=%.2f (secs) \t Accuracy =%.2f (%%)\n',ii,...
        sum(NB.Elapse),NB.AccuracyTest(end));
end
save('stored_model\NB_mnist8m_labeldrift.mat', 'OutputNB', 'NB');
