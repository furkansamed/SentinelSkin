%this is the main file of the Matlab application

% test_benign
% test_benign
imageFolderPath = fullfile('path');
saveFolderPath = fullfile('path');
segmentImages(imageFolderPath, saveFolderPath);
disp('segmentetation done!');


imageFolderPath = fullfile('path');
saveFolderPath = fullfile('path');
excelFileName = 'name.xlsx';
feature_extraction(imageFolderPath, saveFolderPath, excelFileName);
disp('test_benign_features done!');

% test_malignant
% test_malignant
imageFolderPath = fullfile('path');
saveFolderPath = fullfile('path');
segmentImages(imageFolderPath, saveFolderPath);
disp('segmentetation done!');


imageFolderPath = fullfile('path');
saveFolderPath = fullfile('path');
excelFileName = 'name.xlsx';
feature_extraction(imageFolderPath, saveFolderPath, excelFileName);
disp('test_malignant_features done!');

% train_benign
% train_benign
imageFolderPath = fullfile('path');
saveFolderPath = fullfile('path');
segmentImages(imageFolderPath, saveFolderPath);
disp('segmentetation done!');


imageFolderPath = fullfile('path');
saveFolderPath = fullfile('path');
excelFileName = 'name.xlsx';
feature_extraction(imageFolderPath, saveFolderPath, excelFileName);
disp('train_benign_features done!');

% train_malignant
% train_malignant
imageFolderPath = fullfile('path');
saveFolderPath = fullfile('path');
segmentImages(imageFolderPath, saveFolderPath);
disp('segmentetation done!');


imageFolderPath = fullfile('path');
saveFolderPath = fullfile('path');
excelFileName = 'name.xlsx';
feature_extraction(imageFolderPath, saveFolderPath, excelFileName);
disp('train_malignant_features done!');
disp('All operations succesfuly completed');
