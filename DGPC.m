clc;clear;close all
Subj =textread('D:\taskswitch100\data\taskHCSCH.txt','%s');
N_sub=length(Subj);N=100;S=[];P=[];
for sub=1:N_sub
    DataDir=strcat('D:\taskswitch100\data\FC\',Subj(sub),'_Schaefer2018_100_sFC.mat');
    load(char(DataDir));
    fc=FC;
    fc(fc<0)=0;
    fc=(fc+fc')/2;
    [M,Q]=modularity_und(fc,1);
    p=participation_coef(fc,M,0)';
    P=[P;p];
    [deg] = degrees_und(fc);
    S=[S;deg];
end
S1=S/100;
S2=mean(S1,2);
pc=normalize(P);
deg=normalize(S);
save('deg.mat','deg')
save('pc.mat','pc')