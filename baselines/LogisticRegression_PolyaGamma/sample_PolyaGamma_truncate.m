function X = sample_PolyaGamma_truncate( aa, cc )
%SAMPLE_POLYAGAMMA Summary of this function goes here
%   a>0
%   c in R

pi=3.14;
KK=1000;
kk=1:KK;

temp=sum(gamrnd(aa,1,[1 KK])./((kk-1/2).*(kk-1/2)+cc*cc/(4*pi*pi)));

X=temp/(2*pi*pi);
end

