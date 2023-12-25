clc;clear;close all
IN = xlsread('D:\taskswitch100\result', 'Hin');
IM= xlsread('D:\taskswitch100\result', 'Hse');
Diff_Hin = mean(IN(1:111,:))-mean(IN(112:161,:));
Diff_Hse = mean(IM(1:111,:))-mean(IM(112:161,:));
P=[];
for i = 1:100
[h,p] = ttest2(IN(1:111,i),IN(112:161,i));
[h1,p1] = ttest2(IM(1:111,i),IM(112:161,i));
P(1,i) = p;
P(2,i) = p1;
end
% Diff_Hin(find(P(1,:)>0.05))=0;
% Diff_Hse(find(P(2,:)>0.05))=0;
brain=xlsread('D:\taskswitch100\Thomas_roi100.xlsx','d1:d100');
A=[];
for s=1:7
a=sum(Diff_Hin(find(brain==s)))/length(find(Diff_Hin(find(brain==s))~=0));
a1=sum(Diff_Hse(find(brain==s)))/length(find(Diff_Hse(find(brain==s))~=0));
A(1,s) = a;
A(2,s) = a1;
end
% % 
% hdr = spm_vol('D:\taskswitch100\code\Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii');%%save node degree atlas
% vol = spm_read_vols(hdr);
% new_vol = zeros(size(vol));
% for i=1:100
%     new_vol(vol == i) = Diff_Hse(i);
% end
% new_hdr = hdr;
% revised_aal=char(strcat('D:\taskswitch100\code\Diff_Hse.nii'));
% new_hdr.fname = revised_aal;
% spm_write_vol(new_hdr, new_vol);

sys=xlsread('D:\taskswitch100\Thomas_roi100.xlsx','sheet1','d1:d100');
G=[];
T=[];
for i=1:7
    n=find(sys==i);
    IN1=mean(IN(:,n)');
    IM1=mean(IM(:,n)');
    [h, p, ci, stats]=ttest2(IN1(:,1:111),IN1(:,112:161));
    tvalue = stats.tstat;
    [h1, p1, ci1, stats1]=ttest2(IM1(:,1:111),IM1(:,112:161));
    tvalue1 = stats1.tstat;
    G=[G;p,p1];
    T=[T;tvalue,tvalue1];
end

