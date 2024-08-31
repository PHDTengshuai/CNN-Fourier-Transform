%% 准备工作空间
clc;clear all;close all
%% 导入数据
digitDatasetPath = fullfile('./', '/HandWrittenDataset/');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');% 采用文件夹名称作为数据标记
%,'ReadFcn',@mineRF
% 数据集图片个数
countEachLabel(imds)
numTrainFiles = 500;        % 每一个数字有22个样本，取17个样本作为训练数据
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
% 查看图片的大小
img=readimage(imds,1);
size(img)
%% 定义卷积神经网络的结构
% layers = [
% % 输入层
% imageInputLayer([128 128 1])%照片的尺寸
% % 卷积层
% convolution2dLayer(3,4,'stride',2)
% batchNormalizationLayer          %归一化
% dropoutLayer
% reluLayer                        %激活层
% maxPooling2dLayer(2,'stride',2)
% convolution2dLayer(3, 8)
% batchNormalizationLayer          %归一化
% reluLayer                        %激活层
% maxPooling2dLayer(2,'stride',2)
% convolution2dLayer(3, 16)
% batchNormalizationLayer
% reluLayer
% % 最终层
% fullyConnectedLayer(5)           %全连接层
% softmaxLayer
% classificationLayer];            %分类层
%% 训练神经网络
% 设置训练参数
options = trainingOptions('sgdm',...
    'maxEpochs', 100, ...
    'Executionenvironment','cpu',...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency',5,...
    'Verbose',false,...
    'Plots','training-progress');% 显示训练进度
% 训练神经网络，保存网络
net = trainNetwork(imdsTrain, lgraph ,options);
save 'CSNet.mat' net
%% 标记数据（文件名称方式，自行构造）
mineSet = imageDatastore('./ceshi/',  'FileExtensions', '.jpg',...
    'IncludeSubfolders', false);%%,'ReadFcn',@mineRF
mLabels=cell(size(mineSet.Files,1),1);
for i =1:size(mineSet.Files,1)
[filepath,name,ext] = fileparts(char(mineSet.Files{i}));
mLabels{i,1} =char(name);
end
mLabels2=categorical(mLabels);
mineSet.Labels = mLabels2;
%% 使用网络进行分类并计算准确性
% % 手写数据
% YPred = classify(net,mineSet);%概率用predict替换classify
% gailv = predict(net,mineSet)
% YValidation =mineSet.Labels;
% % 计算正确率
% accuracy = sum(YPred ==YValidation)/numel(YValidation);
% % 绘制预测结果
% figure;
% D = dir(['ceshi/*.jpg'])
% nSample=length(D);%照片数量
% ind = randperm(size(YPred,1),nSample);
% for i = 1:nSample
t=4; %t表示绘图行数  
% subplot(t,ceil(nSample/t),i)
% imshow(char(mineSet.Files(ind(i))))
% title(['预测:' char(gailv(ind(i)))],'color','r')  %char(gailv(ind(i)))
% if char(YPred(ind(i))) ==char(YValidation(ind(i)))
%     xlabel(['照片:' char(YValidation(ind(i)))])
% else
%     xlabel(['照片:' char(YValidation(ind(i)))])
% end
% end
YPred = classify(net,mineSet);%概率用predict替换classify
YValidation =mineSet.Labels;
% 计算正确率
accuracy = sum(YPred ==YValidation)/numel(YValidation);
% 绘制预测结果
figure;
D = dir(['ceshi/*.jpg'])
nSample=length(D);%照片数量
ind = randperm(size(YPred,1),nSample);
for i = 1:nSample
  subplot(t,ceil(nSample/t),i)
% ceil(4,(nSample/4),i)
imshow(char(mineSet.Files(ind(i))))
% title(['预测：' char(YPred(ind(i)))])
% 
%     xlabel(['照片:' char(YValidation(ind(i)))],'color','b')
end