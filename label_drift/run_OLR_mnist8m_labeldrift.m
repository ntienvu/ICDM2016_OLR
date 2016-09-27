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
OLR=[];
OLR.Elapse=[];
OLR.AccuracyTest=[];

for ii=1:nBlock
    mytime=tic;
    
    % load data
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
        
    OLR = OnepassLogisticRegression_labeldrift(OLR,yyblock,xxblock,yyTest,xxTest);
    ttt=toc(mytime);
    OLR.Elapse=[OLR.Elapse ttt];
    fprintf('OLR \t Block %d \t Total Elapse=%.2f (secs) \t Accuracy =%.2f (%%)\n',ii,...
        sum(OLR.Elapse),OLR.AccuracyTest(end));
end
save('stored_model\OLR_mnist8m_labeldrift.mat', 'OLR');
