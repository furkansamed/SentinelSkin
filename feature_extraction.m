function feature_extraction(imageFolderPath, saveFolderPath, excelFileName)
        imageFolder = imageFolderPath;
        % Header Array for Excel File Name and Properties
        columnHeaders = {'Images','Mean', 'Median', 'Standard_Deviation', 'Skewness', 'Kurtosis', 'Energy', 'Volume', 'Surface_Area', ...
            'Compactness', 'Sphericity', 'Contrast', 'Homogeneity', 'Energy_(GLCM)', 'Correlation', 'Entropy_(GLCM)', ...
            'Flatness', 'Entropy_(GLSZM)', 'Energy_(GLSZM)', 'Entropy_(GLDM)', 'Energy_(GLDM)','Intensity_Range', 'Mean_Intensity', 'Median_Intensity', ...
            'Ferets_Diameter', 'Perimeter', 'Solidity', 'Eccentricity', 'Area', 'Elongation', ...
            'Fractal_Dimension', 'Roundness', 'Minimum_Intensity','Maximum_Intensity', ...
            'Perimeter','Convexity','Dissimilarity','Maximum_Probability','Gray_Level_Variance','Short_Run_Low_Gray_Level_Emphasis',...
            'Short_Run_High_Gray_Level_Emphasis','Long_RunLow_Gray_Level_Emphasis','Long_Run_High_Gray_Level_Emphasis',...
            'ContrastGLCM','HomogeneityGLCM','CorrelationGLCM','Short_Run_High_Gray_Level_Emphasis_(GLRLM)','Long_Run_High_Gray_Level_Emphasis_(GLRLM)',...
            'Total_Energy', 'Interquartile_Range','Range','Mean_Absolute_Deviation_(MAD)','rMAD','RMS','Uniformity','Spherical_Disproportion','Autocorrelation','GLN'};
            
            
        % Excel File Name and Path
        excelFilePath = fullfile(saveFolderPath, excelFileName);
        % Create an Excel file and write the headers
        xlswrite(excelFilePath, columnHeaders, 'Sheet1', 'A1');
        
        % Loop through all the images in the folder
        imageFiles = dir(fullfile(imageFolder, '*.jpg'));
        if ~isempty(imageFiles)
        for j = 1:length(imageFiles)
            % Resmi yükle ve gri tonlamalı bir görüntüye dönüştür
        image = imread(fullfile(imageFolder, imageFiles(j).name));
        grayImage = rgb2gray(image);
        
        
        
        %%% part7
        % Define the run lengths and gray levels
        runLengths = 1:255;
        grayLevels = 0:255;
        % Pad the image with zeros
        paddedImage = padarray(grayImage, [0 max(runLengths)-1], 'post');
        % Calculate the differences along each row of the padded image
        differences = diff(paddedImage, 1, 2);
        % Find the run starts and ends
        runStarts = find(differences ~= 0);
        runEnds = runStarts + 1;
        % Calculate the run lengths
        runLengthsImage = runEnds - runStarts;
        % Calculate the gray levels
        grayLevelsImage = paddedImage(runStarts);
        % Initialize the GLRLM mat
        glrlm = zeros(length(runLengths), length(grayLevels));
            % Calculate the GLRLM matrix
            for i = 1:length(runLengths)
                for k = 1:length(grayLevels)
                    runLength = runLengths(i);
                    grayLevel = grayLevels(k);
                    % Find the runs with the specified run length and gray level
                    matchingRuns = (runLengthsImage == runLength) & (grayLevelsImage == grayLevel);
                    % Count the number of matching runs
                    glrlm(i, k) = sum(matchingRuns);
                end
            end
            % Calculate GLRLM features
            shortRunHighGrayLevelEmphasis = sum(glrlm ./ (grayLevels.^2));
            % Find columns with NaN values
            nanColumns = any(isnan(shortRunHighGrayLevelEmphasis));
            % Remove columns with NaN values
            shortRunHighGrayLevelEmphasis(:, nanColumns) = [];
            % Calculate the mean if there are remaining columns
            meanShortRunHighGrayLevelEmphasis = 0;
            if ~isempty(shortRunHighGrayLevelEmphasis)
                meanShortRunHighGrayLevelEmphasis = mean(shortRunHighGrayLevelEmphasis);
            end
        longRunHighGrayLevelEmphasis = sum(glrlm(:) .* (runLengths.^2));
        meanlongRunHighGrayLevelEmphasis = mean(longRunHighGrayLevelEmphasis);
        % Calculate the GLCM using the graycomatrix function
        glcm = graycomatrix(grayImage);
        % Calculate GLCM features
        contrastGLCM = graycoprops(glcm, 'Contrast');
        homogeneityGLCM = graycoprops(glcm, 'Homogeneity');
        correlationGLCM = graycoprops(glcm, 'Correlation');
        
            
        %%% part1    
        meanValue = mean(image(:));
        medianValue = median(image(:));
        stdValue = std(double(image(:)));
        skewnessValue = skewness(double(image(:)));
        kurtosisValue = kurtosis(double(image(:)));
        energyValue = sum(image(:).^2);
        volumeValue = sum(image(:) > 0);
        surfaceAreaValue = sum(image(:) > 0);
        
        
        %%% part2
        % Create binary image
        binaryImage = imbinarize(grayImage);
        % Calculate GLCM (Gradient Line Correlation Matrix) features
        glcm = graycomatrix(grayImage);
        % Calculate Compactness feature
        compactness = (4 * pi * sum(binaryImage(:))) / (sum(binaryImage(:))^2);
        % Calculate Sphericity feature
        stats = regionprops(binaryImage, 'Area', 'Perimeter');
        sphericity = (pi * (4 .* [stats.Area])) ./ ([stats.Perimeter].^2);
        meanSphericity = mean(sphericity(isfinite(sphericity))); % Remove infinite values
        % Calculate Contrast feature
        contrast = mean2(glcm .* (glcm > 0));
        % Calculate Homogeneity feature
        homogeneity = sum(sum(glcm ./ (1 + abs(repmat((0:(size(glcm,1)-1))', [1, size(glcm,2)]) - repmat((0:(size(glcm,2)-1)), [size(glcm,1), 1])))));
        % Calculate Energy (GLCM) feature
        energyGLCM = sum(sum(glcm.^2));
        % Calculate Correlation feature
        correlation = (sum(sum(repmat((0:(size(glcm,1)-1))', [1, size(glcm,2)]) .* glcm)) - mean2(repmat((0:(size(glcm,1)-1))', [1, size(glcm,2)]) .* glcm)) / (std2(repmat((0:(size(glcm,1)-1))', [1, size(glcm,2)]) .* glcm) * std2(repmat((0:(size(glcm,1)-1))', [1, size(glcm,2)])));
        % Calculate Entropy (GLCM) feature
        entropyGLCM = -sum(sum(glcm .* log(glcm + eps)));
        % Calculate Flatness feature
        flatness = sqrt([stats.Area]) ./ [stats.Perimeter];
        meanFlatness = mean(flatness(isfinite(flatness))); % Remove infinite values
        
        %%% part3
        glszm = graycomatrix(grayImage);
        % Calculate Entropy (GLSZM) feature
        entropyGLSZM = entropy(glszm);
        % Calculate Energy (GLSZM) feature
        energyGLSZM = sum(sum(glszm.^2));
        % Extract GLDM features
        [~, gmag] = imgradient(grayImage);
        glcm = graycomatrix(gmag);
        % Calculate Entropy (GLDM) feature
        entropyGLDM = entropy(glcm);
        % Calculate Energy (GLDM) feature
        energyGLDM = sum(sum(glcm.^2));
        
        
        
        %%% part4
        grayImage = im2double(grayImage);
        % Compute GLCM matrix
        % glcm = graycomatrix(grayImage);
        % Normalize the GLCM matrix
        glcm = glcm ./ sum(glcm(:));
        % Compute Dissimilarity
        [row, col] = size(glcm);
        dissimilarity = 0;
            for i = 1:row
                for k = 1:col
                    dissimilarity = dissimilarity + abs(i - k) * glcm(i, k);
                end
            end
         % Compute ASM (Angular Second Moment)
         asm = sum(glcm(:).^2);
         % Compute Maximum Probability
         maximumProbability = max(glcm(:));
         grayLevelVariance = var(grayImage(:));
         [X, Y] = meshgrid(1:col, 1:row);
         meanX = sum(X(:).*glcm(:));
         meanY = sum(Y(:).*glcm(:));
         stdX = sqrt(sum(((X(:) - meanX).^2).*glcm(:)));
         stdY = sqrt(sum(((Y(:) - meanY).^2).*glcm(:)));
         correlation_other = sum(((X(:) - meanX).*(Y(:) - meanY).*glcm(:)))./(stdX*stdY);
         % Compute Short Run Low Gray Level Emphasis
         shortRunLowGrayLevelEmphasis = sum(glcm(:)./(1 + abs(X(:) - Y(:))));
         % Compute Long Run Low Gray Level Emphasis
         longRunLowGrayLevelEmphasis = sum(glcm(:).*(abs(X(:) - Y(:)) == 1));
         % Compute Short Run High Gray Level Emphasis
         shortRunHighGrayLevelEmphasis_other = sum(glcm(:).*(abs(X(:) - Y(:)) > 1).^2);
         % Compute Long Run High Gray Level Emphasis
         temp = abs(X(:) - Y(:));
         temp(temp <= 1) = eps;  % Replace values <= 1 with a very small value
         longRunHighGrayLevelEmphasis_other = sum(glcm(:).*(temp > 1)./temp);
        
        %%% part5
        % Extract features
        rangeIntensity = range(grayImage(:));
        meanIntensity = mean(grayImage(:));
        medianIntensity = median(grayImage(:));
        % Convert the image to binary
        binaryImage = imbinarize(grayImage);
        % Label the objects
        labeledImage = bwlabel(binaryImage);
        % Extract regional properties
        stats = regionprops(labeledImage, 'MajorAxisLength', 'Perimeter', 'Solidity', 'Eccentricity', 'Area');
        feretsDiameter = [stats.MajorAxisLength];
        meanFeretsDiameter = mean(feretsDiameter);
        perimeter = [stats.Perimeter];
        meanPerimeter = mean(perimeter);
        solidity = [stats.Solidity];
        meanSolidity = mean(solidity);
        eccentricity = [stats.Eccentricity];
        meanEccentricity = mean(eccentricity);
        area = [stats.Area];
        meanArea = mean(area);
        % Calculate elongation for each object
        elongation = (4 * pi * [stats.Area]) ./ ([stats.Perimeter].^2);
        % Remove any NaN or Inf values from elongation
        elongation = elongation(~isinf(elongation) & ~isnan(elongation));
        % Calculate mean elongation
        meanElongation = mean(elongation);
        % Calculate fractal dimension for each object
        fractalDimension = log([stats.Perimeter]) ./ log(1 / 3);
        % Remove any NaN or Inf values from fractal dimension
        fractalDimension = fractalDimension(~isinf(fractalDimension) & ~isnan(fractalDimension));
        % Calculate mean fractal dimension
        meanFractalDimension = mean(fractalDimension);
        roundness = (4 * pi * area) ./ ((pi * (feretsDiameter / 2)).^2);
        meanRoudness = mean(roundness);
     
        %%% part6
        % Görüntüyü ikili bir şekle dönüştürün
        binaryImg = imbinarize(grayImage);
        % Convex Area (Convexity) hesapla
        convexity = sum(binaryImg(:));
        % Min ve Max Intensity hesapla
        minIntensity = min(grayImage(:));
        maxIntensity = max(grayImage(:));
        % Mean Intensity hesapla
        meanIntensity = mean(grayImage(:));
        % Median Intensity hesapla
        medianIntensity = median(grayImage(:));
        % Major Axis Length (Feret's Diameter) hesapla
        feretDiameter = max(size(binaryImg));
        meanFeretsDiameter = mean (feretDiameter);
        % Perimeter hesapla 
        perimeter = bwarea(bwperim(binaryImg));
        % Range hesapla
        rangeValue = maxIntensity - minIntensity;
        % GLCM için görüntüyü grimsiye dönüştürün
        grayImgGLCM = uint8(grayImage);
        % GLCM matrisini hesaplayın
        glcm = graycomatrix(grayImgGLCM, 'Offset', [0 1], 'Symmetric', true);
        % Dissimilarity (GLCM) hesaplayın
        dissimilarityGLCM = mean(mean(glcm));


        % part 8
        %%% total enegry
        c = 1;
        [rows, cols] = size(image);
        % Görüntünün piksel sayısını hesaplayın
        N = rows * cols;
        % Toplam enerjiyi hesaplamak için boş bir matris oluşturun
        totalEnergy = 0;
        % Görüntüdeki her bir piksel için döngü
        for i = 1:N
            % Görüntü pikselinin değerini alın
            X = image(i);
            % Denklemdeki işlemleri uygulayın
            energy = (X + c)^2;
            % Toplam enerjiye ekleyin
            totalEnergy = totalEnergy + energy;
        end
        
        %%% Interquartile Range
        image = double(image); % Görüntüyü double türüne dönüştürün
        % İnterkartil Aralık işlemini hesaplayın
        p25 = prctile(image(:), 25); % 25. persentili hesaplayın
        p75 = prctile(image(:), 75); % 75. persentili hesaplayın
        interquartileRange = p75 - p25; % İnterkartil Aralık hesaplayın
        
        %%% Range
        rangeValue = max(image(:)) - min(image(:)); % 11. Range hesapla
        
        %%% Mean Absolute Deviation (MAD)
        [Np, ~] = size(image);
        meanValue = mean(image(:));
        MAD = (1/Np) * sum(abs(image(:) - meanValue));
        
        %%% rMAD
        image = double(image) / 255;
        % Denklemi uygulama
        N = numel(image);
        X = sort(image(:), 'ascend');
        X_10th = X(floor(N * 0.1) + 1 : floor(N * 0.9));
        X_bar_10th = mean(X_10th);
        rMAD = (1 / (N * 0.8)) * sum(abs(X_10th - X_bar_10th));
        
        %%% RMS
        Np = numel(image);
        c = 14;
        RMS = sqrt(sum((image(:) + c).^2) / Np);
        
        %%% Uniformity
        uniformity = sum(image(:).^2);
        
        %%% Spherical Disproportion
        volumeValue = sum(image(:));
        % Calculate the surface area of the sphere
        surfaceAreaValue = sum(image(:) > 0);
        % Apply the Spherical Disproportion equation
        sphericalDisproportion = (36 * pi * volumeValue^2)^(1/3) / (4 * pi * surfaceAreaValue);
        
        %%% Autocorrelation
        doubleimage = double(image); % Convert image to double for calculations
        [height, width] = size(doubleimage);
        autocorr = 0;
        % Calculate autocorrelation
        for i = 1:height
            for j = 1:width
                autocorr = autocorr + doubleimage(i, j) * i * j;
            end
        end
        
       %%% GLN
        % Equation constants
        N = size(image, 1); % Height of the image
        M = size(image, 2); % Width of the image
        Z = max(image(:)) - min(image(:)) + 1; % Intensity range of the image
        
        % Convert the image to integers
        doubledimage = round(double(image));
        gln = sum((accumarray(doubledimage(:)+1, 1) / (N * M)).^2) / Z;
        
        % Writing the features
        rowData = {imageFiles(j).name, meanValue, medianValue, stdValue, skewnessValue, kurtosisValue, energyValue, volumeValue, surfaceAreaValue, ...
            compactness, meanSphericity, contrast, homogeneity, energyGLCM, correlation, entropyGLCM, meanFlatness, entropyGLSZM, ...
            energyGLSZM, entropyGLDM, energyGLDM,rangeIntensity, meanIntensity, medianIntensity, meanFeretsDiameter, ...
            meanPerimeter, meanSolidity, meanEccentricity, meanArea, meanElongation, meanFractalDimension, meanRoudness,minIntensity,maxIntensity, ...
            perimeter,convexity,dissimilarity,maximumProbability,grayLevelVariance,shortRunLowGrayLevelEmphasis,...
            shortRunHighGrayLevelEmphasis_other,longRunLowGrayLevelEmphasis,longRunHighGrayLevelEmphasis_other,...
            contrastGLCM.Contrast, homogeneityGLCM.Homogeneity, correlationGLCM.Correlation,meanShortRunHighGrayLevelEmphasis,meanlongRunHighGrayLevelEmphasis,...
            totalEnergy,interquartileRange,rangeValue,MAD,rMAD,RMS,uniformity,sphericalDisproportion,autocorr,gln};
        
            xlswrite(excelFilePath, rowData, 'Sheet1', sprintf('A%d', j+1));

        end
    end
end
