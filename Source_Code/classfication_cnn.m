clear all;
close all;
clc;

image_folder = 'D:\DRImp\oriaugseg'; % give the path of the folder where your images are present
imds = imageDatastore(image_folder,'IncludeSubfolders',true, 'LabelSource', 'foldernames');
total_split = countEachLabel(imds);
% Number of Images
num_images=length(imds.Labels);

% Visualize random 20 images
perm=randperm(num_images,40);
figure;
for idx=1:20
    subplot(4,5,idx);
    imshow(imread(imds.Files{perm(idx)}));
    title(sprintf('%s',imds.Labels(perm(idx))))
    
end
% Split the Training and Testing Dataset
train_percent=0.4;
[imdsTrain,imdsTest]=splitEachLabel(imds,train_percent,'randomize');
 
% Split the Training and Validation
valid_percent=0.01;
[imdsValid,imdsTrain]=splitEachLabel(imdsTrain,valid_percent,'randomize');

% Converting images to 299 x 299 to suit the architecture
augimdsTrain = augmentedImageDatastore([336 448],imdsTrain);
augimdsValid = augmentedImageDatastore([336 448],imdsValid);


layers = [ ...
    imageInputLayer([336 448 1])
    maxPooling2dLayer(2,'Stride',2)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(9,10)
    maxPooling2dLayer(2,'Stride',3)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(6,10)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% Set the training options
%options = trainingOptions('adam','MaxEpochs',20,'MiniBatchSize',32,...
%'Plots','training-progress','Verbose',0,...
%'ValidationData',augimdsValid,'ValidationFrequency',50,'ValidationPatience',3);

%options = trainingOptions('sgdm', ...
 %   'MaxEpochs',20,...
  %  'InitialLearnRate',1e-4, ...
   % 'Verbose',false, ...
    %'Plots','training-progress');
maxEpochs = 15;
miniBatchSize = 32;

    options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,layers,options);



% Test
% Reshape the test images match with the network 
augimdsTest = augmentedImageDatastore([336 448],imdsTest);

% Predict Test Labels
[predicted_labels,posterior] = classify(net,augimdsTest);

% Actual Labels
actual_labels = imdsTest.Labels;

% Confusion Matrix
figure
plotconfusion(actual_labels,predicted_labels)
title('Confusion Matrix');

% ROC Curve
test_labels=double(nominal(imdsTest.Labels));
[fp_rate,tp_rate,T,AUC] = perfcurve(test_labels,posterior(:,2),2);
figure;
plot(fp_rate,tp_rate,'b-');hold on;
grid on;
xlabel('False Positive Rate');
ylabel('Detection Rate');
save net;