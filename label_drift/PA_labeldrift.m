function [model, hat_y_t, l_t] = PA_labeldrift(y_t, x_t, model)
% PA_labeldrift: Multiclass Passive-Aggressive (M-PA) learning algorithms for label-drift classification.
%--------------------------------------------------------------------------
% Reference:
% - Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz, and Yoram
% Singer. Online passive-aggressive algorithms. JMLR, 7:551?85, 2006.
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
% [KK,dim]=size(W);
% if maxClass>KK
%     moreClass=maxClass-KK;
%     KK=maxClass;
%     for kk=1:moreClass
%         W(end+1,:)=zeros(1,dim);
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
%l_t = max(0, 1 - (F_t(y_t) - F_t(s_t))); 
l_t = max(0, 1 - (F_t(y_t) - F_t(s_t))); 
%l_t=1;
%--------------------------------------------------------------------------
% Making Update
%--------------------------------------------------------------------------
if (l_t > 0),
%if (y_t ~=s_t),
    eta_t = min(model.C,l_t/(2*norm(x_t)^2));
    model.W(y_t,:) = W(y_t,:) + eta_t*x_t;
    %model.W(s_t,:) = W(s_t,:) - eta_t*x_t;
end
% THE END
