%Generative Model to detect face and background regions 

clc;
clear all;
close all;
warning('off', 'Images:initSize:adjustingMag');

%file directory
trainingImageDirectory = 'trainingImages\';
trainingRectDirectory = 'trainingRects\'; %rect: [xmin ymin width height].
testingImageDirectory = 'testingImages\';
testingRectDirectory = 'testingRects\'; %rect: [xmin ymin width height].

%Defining parameters, e.g. number of bins % for likelihood
nDim = 33;
%N_face =0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt= cputime;
Pr_x_given_w_equalsTo_1 = zeros(nDim,nDim,nDim);
Pr_x_given_w_equalsTo_0 = zeros(nDim,nDim,nDim);
N_facialRegionPixels = 0;
trainingImageFiles = dir(trainingImageDirectory);
trainingRectFiles = dir(trainingRectDirectory);
for iFile = 3:size(trainingImageFiles,1)
    %load the image and facial image regions
    origIm=imread([trainingImageDirectory trainingImageFiles(iFile).name]);
    origIm = floor(origIm/8);
    load([trainingRectDirectory,trainingRectFiles(iFile).name],'allrects');

    %visualization and generate the mask indicating the facial regions
    [nrows,ncols,~]= size(origIm);
    bwMask = zeros(nrows,ncols);    
    figure; imshow(origIm,[]); hold on;
    for ii = 1:size(allrects,1)
        x=allrects(ii,1)+[0, allrects(ii,3), allrects(ii,3), 0, 0];
        y=allrects(ii,2)+[0, 0, allrects(ii,4), allrects(ii,4), 0];
        plot(x,y,'r','Linewidth',2);        
        bwMask(allrects(ii,2):allrects(ii,2)+allrects(ii,4),...
                    allrects(ii,1):allrects(ii,1)+allrects(ii,3))=1;                    
    end

    % computing prior
    N_facialRegionPixels = N_facialRegionPixels + sum(bwMask(:));

    % computing likelihood
    for irow = 1:nrows
        for icol = 1:ncols
            r = origIm(irow,icol,1)+1;
            g = origIm(irow,icol,2)+1;
            b = origIm(irow,icol,3)+1;
            if(bwMask(irow,icol)==1)
                Pr_x_given_w_equalsTo_1(r,g,b) = Pr_x_given_w_equalsTo_1(r,g,b) + 1;
            else
                Pr_x_given_w_equalsTo_0(r,g,b) = Pr_x_given_w_equalsTo_0(r,g,b) + 1;
            end
        end
    end
           
        
end
%converting histograms into probability distributions:

Pr_w_equalsTo_1 = N_facialRegionPixels/((iFile-3+1)*ncols*nrows);
Pr_w_equalsTo_0 = 1 - Pr_w_equalsTo_1;

Pr_x_given_w_equalsTo_1 = Pr_x_given_w_equalsTo_1/sum(Pr_x_given_w_equalsTo_1(:));
Pr_x_given_w_equalsTo_0 = Pr_x_given_w_equalsTo_0/sum(Pr_x_given_w_equalsTo_0(:));



disp(['traning: ' num2str(cputime-tt)]);
save('FacialRegionDetection_TrainedProbs.mat','Pr_x_given_w_equalsTo_1','Pr_x_given_w_equalsTo_0','Pr_w_equalsTo_1','Pr_w_equalsTo_0');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%End of the training process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('FacialRegionDetection_TrainedProbs.mat','Pr_x_given_w_equalsTo_1','Pr_x_given_w_equalsTo_0','Pr_w_equalsTo_1','Pr_w_equalsTo_0');
testingFiles = dir(testingImageDirectory);
testingRectFiles = dir(testingRectDirectory);
truePositives = 0;
falsePositives = 0;
falseNegtives = 0;

precision = zeros(12,1);
recall = zeros(12,1);
for iFile = 3:size(testingFiles,1)
    tt = cputime;
    
    %load the image and facial image regions, gtMask is the groundtruth
    origIm=imread([testingImageDirectory testingFiles(iFile).name]);
    origIm=floor(origIm/8);
    [nrows, ncols,~] = size(origIm);
    load([testingRectDirectory,testingRectFiles(iFile).name],'allrects');
    gtMask = zeros(nrows,ncols);
    detectedFacialMask = zeros(nrows,ncols);
    %TP = zeros(nrows,ncols);
    for ii = 1:size(allrects,1)
        gtMask(allrects(ii,2):allrects(ii,2)+allrects(ii,4),allrects(ii,1):allrects(ii,1)+allrects(ii,3))=1;                    
    end
    gtMask = gtMask(1:nrows,1:ncols);
    
    % Bayesian inference on the input image (origIm)
    for irow = 1:nrows
        for icol = 1:ncols
            r = origIm(irow,icol,1)+1;
            g = origIm(irow,icol,2)+1;
            b = origIm(irow,icol,3)+1;
            if(Pr_x_given_w_equalsTo_1(r,g,b)*Pr_w_equalsTo_1 > Pr_x_given_w_equalsTo_0(r,g,b)*Pr_w_equalsTo_0)
                detectedFacialMask(irow,icol) = 1;
            end
        end
    end
   
    %(detectedFacialMask),the following are some visualization codes
    showIm = zeros(nrows,ncols,3);
    showIm(:,:,1) = detectedFacialMask; showIm(:,:,2) = gtMask;
    figure; imshow(showIm,[]); title('red: detection, green: groundtruth');
%     figure; imshow(detectedFacialMask); title('detection');
%     figure; imshow(gtMask); title('ground truth');
    showIm = origIm; showIm(nrows*ncols+find(detectedFacialMask)) = 255;
    figure; imshow([origIm repmat(255*detectedFacialMask,[1 1 3]) showIm],[]);
    
   % compute the precision and recall
    TP = detectedFacialMask & gtMask;
    TP = sum(TP(:));
    FP = detectedFacialMask & ~gtMask;
    FP = sum(FP(:));
    FN = ~detectedFacialMask & gtMask;
    FN = sum(FN(:));
    
    precision(iFile-3+1) = TP/(TP+FP);
    recall(iFile-3+1) = TP/(TP+FN);
    disp([num2str(iFile-2) ' testing: ' num2str(cputime-tt)]);
end

%output precision and recall
precision
recall