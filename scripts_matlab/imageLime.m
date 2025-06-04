%net = importTensorFlowNetwork("D:/MODELS_DISPLASIA/myModelDispKOConv");

net = trainedNetwork_1;
imagefiles = dir('D:/dataSet-IC2/displasiaWTDataAug/Carcinoma WT/*.tiff');

%imagefiles = natsortfiles(imagefiles);

%[~,ind]=sort({imagefiles.name});
%imagefiles = natsortfiles(imagefiles);

%W = imagefiles(ind);

nfiles = length(imagefiles); 

feat = [];
inputSize = net.Layers(1).InputSize(1:2);

for ii=1:nfiles

currentfilename = imagefiles(ii).name;

disp(currentfilename);

X = imread(fullfile('D:/dataSet-IC2/displasiaWTDataAug/Carcinoma WT',currentfilename));
%inputSize = net.Layers(1).InputSize(1:2);
X = imresize(X,inputSize);

label = classify(net,X);


[scoreMap,featureMap,featureImportance]  = imageLIME(net,X,label,'Segmentation','grid','NumFeatures',64,'NumSamples',500);


numTopFeatures = 5;
[~,idx] = maxk(featureImportance,numTopFeatures);


feat = [feat idx];

end

disp(feat)
disp(length(feat))
disp(size(feat))


writematrix(feat,'D:/MODELS_DISPLASIA/imageLime/features_dispWT_Carcinoma_WT_Den.csv');