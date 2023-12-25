clc; clear; close all

Subj = textread('D:\taskswitch100\data\taskswitch.txt', '%s');
N = 100; N_sub = 252;

for sub = 1:length(Subj)
    BOLD_path = char(strcat('D:\taskswitch100\data\taskswitch\', Subj(sub), '_Schaefer2018_100_taskswitch.mat'));
    
    % Add try-catch block to handle potential loading errors
    try
        load(BOLD_path, 'BOLD');
        
        % Check the size or content of BOLD
        disp(size(BOLD)); % Or disp(BOLD) depending on the data type
        
        FC = corr(BOLD);
        
        [Clus_num, Clus_size] = Functional_HP(FC, N);
        [Hin, Hse, R_Hin, R_Hse, Hin_inter, Hse_inter, HF] = Seg_Int_component(FC, N, Clus_size, Clus_num);
        
        % Save variables in separate .mat files
        save(char(strcat('D:\taskswitch100\data\process result\', Subj(sub), '_cluster_size.mat')), 'Clus_num', 'Clus_size');
        save(char(strcat('D:\taskswitch100\data\process result\', Subj(sub), '_HF.mat')), 'Hin', 'Hse', 'R_Hin', 'R_Hse', 'Hin_inter', 'Hse_inter', 'HF');
        
    catch
        disp(['Error loading or processing data for subject: ' char(Subj(sub))]);
    end
end


Subj = textread('D:\taskswitch100\data\taskswitch.txt', '%s');
Hin = [];
Hse = [];
for i = 1:length(Subj)
    HF_path = fullfile('D:\taskswitch100\data\process result', [char(Subj(i)), '_HF.mat']);
    if exist(HF_path, 'file') == 2
        HF = load(HF_path);
        Hin = [Hin; HF.R_Hin];
        Hse = [Hse; HF.R_Hse];
    else
        disp(['File not found for subject: ' char(Subj(i))]);
    end
end
HB = Hse - Hin;

HIN=mean(Hin,2);
HSE=mean(Hse,2);
IHB=HIN-HSE;

sys=xlsread('D:\taskswitch100\Thomas_roi100.xlsx','sheet1','d1:d100');
R=[];H=[];T=[];
for s=1:7
 n=find(sys==s);
 IN1=mean(Hin(:,n)')';
 IM1=mean(Hse(:,n)')';
 H=[H;IN1];
 R=[R;IM1];
end
H=reshape(H,250,7);
R=reshape(R,250,7);
T=[H,R];
B=H-R;
AIB=abs(IHB);
AB=abs(B);
