function model = LogisticRegrLaplacianApproxBinary(yyTrain,xxTrain)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
fun=@Laplacian_mean_function;

XX=xxTrain'*xxTrain;
dd=size(xxTrain,2);
mu_0=zeros(dd,1);
Sig_0=1*eye(dd);
Sig_1_Sig=1;

options = optimset('Display', 'off','MaxIter',10);
model.mu_n = fsolve(fun,mu_0,options,mu_0,Sig_0,xxTrain,yyTrain);

% for ii=1:length(yyTrain)
%     txx = xxTrain(ii,:);
%     temp=txx* model.mu_n;
%     temp1=SigmoidFunction(temp);
%     Sig_1_Sig= Sig_1_Sig+temp1*(1-temp1);
% end

temp=SigmoidFunction(xxTrain*model.mu_n);
Sig_1_Sig=sum(temp.*(1-temp));

model.invSig_n= Sig_0+ XX*Sig_1_Sig;

try
    model.ww=mvnrnd(model.mu_n,pinv(model.invSig_n));
catch
    model.ww=model.mu_n;
end


end

function out = Laplacian_mean_function( w,m0,S0,xx,yy )
%LAPLACIAN_MEAN_FUNCTION Summary of this function goes here
%   Detailed explanation goes here

dim=length(w);

out=(m0'-w')*(S0\eye(dim));

temp=xx'*(yy-SigmoidFunction(xx*w));

out=out+temp';
end
