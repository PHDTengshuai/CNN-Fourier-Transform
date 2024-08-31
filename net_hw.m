%% ׼�������ռ�
clc;clear all;close all
%% ��������
digitDatasetPath = fullfile('./', '/HandWrittenDataset/');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');% �����ļ���������Ϊ���ݱ��
%,'ReadFcn',@mineRF
% ���ݼ�ͼƬ����
countEachLabel(imds)
numTrainFiles = 500;        % ÿһ��������22��������ȡ17��������Ϊѵ������
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
% �鿴ͼƬ�Ĵ�С
img=readimage(imds,1);
size(img)
%% ������������Ľṹ
% layers = [
% % �����
% imageInputLayer([128 128 1])%��Ƭ�ĳߴ�
% % �����
% convolution2dLayer(3,4,'stride',2)
% batchNormalizationLayer          %��һ��
% dropoutLayer
% reluLayer                        %�����
% maxPooling2dLayer(2,'stride',2)
% convolution2dLayer(3, 8)
% batchNormalizationLayer          %��һ��
% reluLayer                        %�����
% maxPooling2dLayer(2,'stride',2)
% convolution2dLayer(3, 16)
% batchNormalizationLayer
% reluLayer
% % ���ղ�
% fullyConnectedLayer(5)           %ȫ���Ӳ�
% softmaxLayer
% classificationLayer];            %�����
%% ѵ��������
% ����ѵ������
options = trainingOptions('sgdm',...
    'maxEpochs', 100, ...
    'Executionenvironment','cpu',...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency',5,...
    'Verbose',false,...
    'Plots','training-progress');% ��ʾѵ������
% ѵ�������磬��������
net = trainNetwork(imdsTrain, lgraph ,options);
save 'CSNet.mat' net
%% ������ݣ��ļ����Ʒ�ʽ�����й��죩
mineSet = imageDatastore('./ceshi/',  'FileExtensions', '.jpg',...
    'IncludeSubfolders', false);%%,'ReadFcn',@mineRF
mLabels=cell(size(mineSet.Files,1),1);
for i =1:size(mineSet.Files,1)
[filepath,name,ext] = fileparts(char(mineSet.Files{i}));
mLabels{i,1} =char(name);
end
mLabels2=categorical(mLabels);
mineSet.Labels = mLabels2;
%% ʹ��������з��ಢ����׼ȷ��
% % ��д����
% YPred = classify(net,mineSet);%������predict�滻classify
% gailv = predict(net,mineSet)
% YValidation =mineSet.Labels;
% % ������ȷ��
% accuracy = sum(YPred ==YValidation)/numel(YValidation);
% % ����Ԥ����
% figure;
% D = dir(['ceshi/*.jpg'])
% nSample=length(D);%��Ƭ����
% ind = randperm(size(YPred,1),nSample);
% for i = 1:nSample
t=4; %t��ʾ��ͼ����  
% subplot(t,ceil(nSample/t),i)
% imshow(char(mineSet.Files(ind(i))))
% title(['Ԥ��:' char(gailv(ind(i)))],'color','r')  %char(gailv(ind(i)))
% if char(YPred(ind(i))) ==char(YValidation(ind(i)))
%     xlabel(['��Ƭ:' char(YValidation(ind(i)))])
% else
%     xlabel(['��Ƭ:' char(YValidation(ind(i)))])
% end
% end
YPred = classify(net,mineSet);%������predict�滻classify
YValidation =mineSet.Labels;
% ������ȷ��
accuracy = sum(YPred ==YValidation)/numel(YValidation);
% ����Ԥ����
figure;
D = dir(['ceshi/*.jpg'])
nSample=length(D);%��Ƭ����
ind = randperm(size(YPred,1),nSample);
for i = 1:nSample
  subplot(t,ceil(nSample/t),i)
% ceil(4,(nSample/4),i)
imshow(char(mineSet.Files(ind(i))))
% title(['Ԥ�⣺' char(YPred(ind(i)))])
% 
%     xlabel(['��Ƭ:' char(YValidation(ind(i)))],'color','b')
end