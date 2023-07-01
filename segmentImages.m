function segmentImages(imageFolderPath, saveFolderPath)
    % Get all files in the folder
    fileList = dir(fullfile(imageFolderPath, '*.jpg')); % Adjust file extension as needed

    % Perform operations for each image in the folder
    for i = 1:numel(fileList)
        % Load the image
        imagePath = fullfile(imageFolderPath, fileList(i).name);
        image = imread(imagePath);

        % Convert the image to grayscale
        grayscaleImage = rgb2gray(image);

        % Perform segmentation using the Active Contour model
        initialMask = false(size(grayscaleImage));
        initialMask(50:end-50, 50:end-50) = true; % Set the initial mask
        segmentedMask = activecontour(grayscaleImage, initialMask, 100, 'Chan-Vese');

        % Apply the segmentation result to the original image
        segmentedImage = uint8(double(image) .* repmat(segmentedMask, [1, 1, size(image, 3)]));

        % Save the results
        [~, name, ~] = fileparts(imagePath);
        savePath = fullfile(saveFolderPath, [name '_segmented.jpg']); % Path for saving the file
        imwrite(segmentedImage, savePath);
    end
end
