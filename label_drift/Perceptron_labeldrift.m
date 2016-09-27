function [model, hat_y_t, l_t] = Perceptron_labeldrift(y_t, x_t, model)
% Perceptron_labeldrift: Multi-class Perceptron Algorithms with Max-score update for label-drift classification.
%--------------------------------------------------------------------------
%Reference:
%Koby Crammer and Yoram Singer. Ultraconservative online algorithms for multiclass problems.
%Journal of Machine Learning Research, 3:951¨C991, 2003.
%--------------------------------------------------------------------------
% INPUT:
%      y_t:     class label of t-th instance;
%      x_t:     t-th training data instance, e.g., X(t,:);
%    model:     classifier
%
% OUTPUT:   
%    model:     a struct of the weight vector (w) and the SV indexes
%  hat_y_t:     predicted class label
%      l_t:     suffered loss
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Initialization
%--------------------------------------------------------------------------

% maxClass=max(y_t);
% W     = model.W;
% W0    = model.W0;
% [KK,dim]=size(W);
% if maxClass>KK
%     moreClass=maxClass-KK;
%     KK=maxClass;
%     for kk=1:moreClass
%         W(end+1,:)=zeros(1,dim);
% %         W(end+1,:)=W0/model.count;
%         %FeatAccummulate=FeatAccummulate+FeatAccummulate;
%     end
% end

W = model.W;
    
%--------------------------------------------------------------------------
% Prediction
%--------------------------------------------------------------------------
F_t = W*x_t';
[F_max,hat_y_t]=max(F_t);
%% compute the hingh loss and support vector
Fs=F_t;
Fs(y_t)=-inf;
[Fs_max, s_t]=max(Fs);
%--------------------------------------------------------------------------
% Making Update
%--------------------------------------------------------------------------
l_t = (hat_y_t ~= y_t); % 0 - correct prediction, 1 - incorrect
if (l_t > 0),
    model.W(y_t,:) = W(y_t,:) + x_t;
    model.W(s_t,:) = W(s_t,:) - x_t;
%     model.count=model.count+1;
%     model.W0=model.W0-x_t;
end

% THE END


