function [ output_args ] = QuadrantVisualization(AccuracyTime,MethodNameList,Title,TimeLimit)
% AccuracyTime: N methods x 2: [Accuracy1 Time1; Accuracy2 Time2;...;AccuracyN  TimeN];
% MethodNameList: name of N methods, eg, MethodNameList={'Method1','Methods2',...,'MethodsN'};
% Title: title of this plot. eg, Title='DNA: Classification'
% TimeLimit: to scale and balance between accuracy and time. eg, TimeLimit=10
% Example:
% 		AccuracyTime=[91.23 0.09; 90.64 0.47; 76.56 1.45; 94.1 0.3; 94.43 0.08; 94.77 0.81; 93.76 0.01];
%		MethodNameList={'NB','DT','kNN','LDA','SVM','LR','SLR'};
%		Title='DNA: Classification'
%		TimeLimit=0.85;
%		QuadrantVisualization(AccuracyTime,MethodNameList,'DNA: Classification',TimeSLimit);

% This plot works well in Matlab 2014a and 2014b

nMethods=size(AccuracyTime,1);
if nMethods~=length(MethodNameList)
    fprintf('Number of Methods in AccuracyTime (%d) is not consistent with MethodNameList (%d)\n',nMethods,length(MethodNameList));
    return;
end


%polar([1,2,3],[4,5,6]);
myfig=figure;
h=polar(100);
ax = gca;

%colormap(h, jet)
set(groot, 'DefaultFigureColormap', jet)

%ax.LineWidth=2;
%ax.Color='k';


%polarticks(8,myfig);
% 
% 
xl = get(gca,'XLim'); yl = get(gca,'YLim');
set(gca,'XLim', [0 xl(2)], 'YLim', [0 yl(2)]);


ph=findall(gca,'type','text');
ps=get(ph,'string');

if verLessThan('matlab', '8.4')
    ps(1:end)={''};
    ps([1,2,3,4,5,10,11,12])={
        'Accuracy'
        'Time (sec)'
        Title
        num2str(TimeLimit)
        ''
        ''
        '0'
        'Worst'
        };    
else
    ps(1:end)={''};
ps([1,8])={
    num2str(TimeLimit)
    '0'
    };
end


set(ph,{'string'},ps);
ps=get(ph,'fontweight');
ps(1:3)={'bold'};
%set(ph,{'fontweight'},ps);
set(ph,'fontsize',14);
ps=get(ph,'position');


if verLessThan('matlab', '8.4')
    ps{1}(1)=-8;
end
set(ph,{'position'},ps);
hold on;


set(gca,'fontsize',14);
hold on;

text(2,-5,'0','fontsize',14);
hold on;
text(-15,3,'100','fontsize',14);
TimeUnitLabel='sec';
strXlabel=sprintf('Time (%s)',TimeUnitLabel);

if verLessThan('matlab', '8.4')
    xlabel(strXlabel,'fontsize',14);
    ylabel('Accuracy (%)','fontsize',14);
else
    text(40,-7,strXlabel,'fontsize',14);
    text(-7,35,'Accuracy (%)','fontsize',14,'Rotation',90);
    %t=xlabel(strXlabel,'fontsize',14);
    %get(t,'Position')
    %set(t,'Position',get(t,'Position')+[0 10 0]);  % move up slightly
    %ylabel('Accuracy (%)','fontsize',14);
end



t=title(Title,'fontsize',14);

if verLessThan('matlab', '8.4')==0
    get(t,'Position');
    set(t,'Position',get(t,'Position')+[50 0 0]);  % move up slightly
end

%% annotate scores into the plot
Accuracy=AccuracyTime(:,1);
Time=AccuracyTime(:,2);

plottype={'or';'sk';'xb';'+b';'vm';'dk';'*b';'ob';'sr';'xr'};
mysize=150;

QuadrantRadius=zeros(1,nMethods);
myhandle=cell(1,nMethods);
MethodNames=cell(1,nMethods);

temp=[];
for ii=1:nMethods
    uu=mod(ii-1,length(plottype))+1;
    myhandle{ii}=scatter(100*Time(ii)/TimeLimit,100-Accuracy(ii),mysize,plottype{uu});
    hold on;
    %QuadrantRadius(ii)=sqrt( (Time(ii))^2 + (100-Accuracy(ii))^2 );
    
    QuadrantRadius(ii)=sqrt( (100*Time(ii)/TimeLimit)^2 + (100-Accuracy(ii))^2 );
    
    MethodNames{ii}=sprintf('%s (%.2f)',MethodNameList{ii},QuadrantRadius(ii));
    %legend(myhandle{ii},MethodNames{ii});
    temp=[temp myhandle{ii}];
end

legend(temp,MethodNames);
hold on

co = [0,0,1; 
      0,0.5,0;
      1,0,0;
      0,0.75,0.75;
      0.75,0,0.75;
      0.75,0.75,0;
      0.25,0.25,0.25];
set(groot,'defaultAxesColorOrder',co);

end

