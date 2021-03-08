%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Weakly-Supervised Online Hashing
%
% This demo is based on MIRFlickr. For NUS-WIDE, one can easily adapts our code.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all; clear; clc;
nbits_set=[8 16 32 64 96];

%% load dataset
fprintf('loading dataset...\n')
set = 'MIRFlickr';
load('MIRFlickr-data.mat');

anchor=XTrain(randsample(2000,1000),:);


%% initialization
fprintf('initializing...\n')
param.mu = 10;
param.alpha = 300;
param.beta = 0.1;
param.theta = 0.1;
param.iter = 7;
param.datasets = set;
param.chunk = 2000;

for bit=1:length(nbits_set)
    nbits=nbits_set(bit);
    param.nbits=nbits;

    %% model training
        
    [MAP] = train(XTrain,YTrain, ZTrain, param,LTrain,XTest,LTest,anchor);
   
    
end
