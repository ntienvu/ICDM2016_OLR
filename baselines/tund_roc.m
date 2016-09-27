function [F1, rec, spec, prec] = tund_roc(true_label, pred_label, varargin)
%TUND_ROC Summary of this function goes here
%   Detailed explanation goes here

% %% parse input arguments
% parser = inputParser;
% % parser.addRequired('true_label', @isnumeric || @islogical);
% % parser.addRequired('predicted_label', @isnumeric || @islogical);
% 
% parser.parse(true_label, predicted_label, varargin{:});

%%
TP = full(sum(true_label & pred_label));
FN = full(sum(true_label & (~pred_label)));
FP = full(sum((~true_label) & pred_label));
TN = full(sum((~true_label) & (~pred_label)));

rec = TP / (TP+FN+realmin);
prec = TP / (TP+FP+realmin);
F1 = 2*(rec*prec) / (rec+prec+realmin);
spec = TN / (FP+TN+realmin);

end

