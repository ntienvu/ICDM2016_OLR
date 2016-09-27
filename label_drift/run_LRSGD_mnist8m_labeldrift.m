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

%% Logistic Regression using Stochastic Gradient Descent
LRSGD = [];
LRSGD.Elapse=[];
LRSGD.AccuracyTest=[];

%pass through each block, then update the model parameter
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

    LRSGD = LRSGD_labeldrift(LRSGD,yyblock,full(xxblock),yyTest,full(xxTest));    
    ttt=toc(mytime);
    
    LRSGD.Elapse=[LRSGD.Elapse ttt];
    LRSGD.AccuracyTest=[LRSGD.AccuracyTest LRSGD.AccuracyTest];
    fprintf('LRSGD \tBlock %d \tTotal Elapse=%.2f (sec) \tAccuracy=%.2f (%%)\n',ii,sum(LRSGD.Elapse),LRSGD.AccuracyTest(end));
end
save('stored_model/LRSGD_mnist8M_labeldrift.mat', 'LRSGD');
