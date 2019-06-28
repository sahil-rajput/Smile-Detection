categories={'images_smile','image_serious'};
imds=imageDatastore('D:\fold1','includeSubfolder',true,'LabelSource','foldernames');
g=length(imds.Files);
features=[];
for k=1:g
    im=imread(imds.Files{k});
    h=HOG(im);
    features=[features,h];
end
training_data=[features(:,1:400),features(:,508:908)];
test_data=[features(:,401:507),features(:,909:end)];
 training_label=[ones(400,1);2*ones(401,1)];
 test_label=[ones(107,1);2*ones(358,1)];
   sv=svmtrain(training_data,training_label,'kernel_function','rbf');
   out=svmclassify(sv,test_data');