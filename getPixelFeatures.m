% clc; clear; close all;
% 
% imgLoc = '~/Downloads/things-images';
% 
% imgCats = dir(imgLoc);
% imgCats = {imgCats(3:end).name};
% 
% imgCount = 1;
% for cc=1:length(imgCats)
%     disp(['category ' num2str(cc) ': ' imgCats{cc}])
%     imgFiles = dir([imgLoc '/' imgCats{cc}]);
%     imgFiles = {imgFiles(3:end).name};
%     imgs = cellfun(@(x) imread([imgLoc '/' imgCats{cc} '/' x]),imgFiles,'UniformOutput',false);
%     imgs = cellfun(@(x) imresize(x,[800 800]),imgs,'UniformOutput',false);
% 
%     for ii=1:length(imgs)
%         pixelFeatures(imgCount) = extractAll(imgs{ii},imgFiles{ii});
%         imgCount = imgCount + 1;
%     end
% end
% tmp = rmfield(pixelFeatures,'exempName');
% pixelFeatures_mat = reshape(struct2array(tmp),[],26107)';
% save('data/pixelFeatures.mat','pixelFeatures','pixelFeatures_mat')

function features = getPixelFeatures(img,imgName)
    features.exempName = imgName;

    % Convert to grayscale
    gray_img = rgb2gray(img);
    
    % 1. Contrast (Standard deviation of grayscale image)
    features.contrast = std2(gray_img);

    % 2. Orientation (Gradient directions)
    [gradient_x, gradient_y] = gradient(double(gray_img));
    orientation = atan2d(gradient_y, gradient_x);
    features.orientation = mean(orientation(:));  % Mean orientation
    
    % 3. Curvature Content (Laplacian of the image)
    laplacian = del2(double(gray_img));
    features.curvature_content = std2(laplacian);

    % 4. Sharpness (Variance of the Laplacian)
    features.sharpness = var(double(laplacian(:)));
    
    % 5. Skewness and Kurtosis of the intensity distribution
    features.skewness = skewness(double(gray_img(:)));
    features.kurtosis = kurtosis(double(gray_img(:)));

    % 6. Fourier Energy (Energy in frequency domain)
    F = fft2(double(gray_img));  
    F_shifted = fftshift(F);  
    features.fourier_energy = mean(abs(F_shifted(:)).^2);
    
    % 7. Color Histogram (For RGB image)
    if size(img, 3) == 3
        imgTmp = img;
    else
        imgTmp = repmat(gray_img,1,1,3);
    end
    features.color_hist_red = mean(imgTmp(:,:,1),'all')./mean(gray_img,'all'); % histcounts(img(:,:,1), 256);
    features.color_hist_green = mean(imgTmp(:,:,2),'all')./mean(gray_img,'all');
    features.color_hist_blue = mean(imgTmp(:,:,3),'all')./mean(gray_img,'all');
    
    % 8. Texture Features using GLCM (Gray Level Co-occurrence Matrix)
    glcm = graycomatrix(gray_img);
    stats = graycoprops(glcm, {'contrast', 'correlation', 'energy', 'homogeneity'});
    features.texture_contrast = stats.Contrast;
    features.texture_correlation = stats.Correlation;
    features.texture_energy = stats.Energy;
    features.texture_homogeneity = stats.Homogeneity;

    % 9. Symmetry (Using mirror comparison)
    flipped_img = flipud(gray_img);  % Flip image upside down
    symmetry = sum(sum(abs(gray_img - flipped_img))) / numel(gray_img);  % Measure symmetry
    features.symmetry = symmetry;
    
    % 10. Edge Density
    edges = edge(gray_img, 'Sobel');
    edge_density = sum(edges(:)) / numel(edges);  % Fraction of edge pixels
    features.edge_density = edge_density;
    
    % 11. Aspect Ratio (Bounding box of connected components)
    bw = imbinarize(gray_img);  % Binary image
    stats = regionprops(bw, 'BoundingBox');
    if ~isempty(stats)
        boundingBox = stats(1).BoundingBox;
        features.aspect_ratio = boundingBox(3) / boundingBox(4);
    else
        features.aspect_ratio = NaN;  % No object found
    end
    features.boundingBox_size = boundingBox(3)*boundingBox(4)/numel(gray_img);

    
    % 12. Corner Detection (Harris Corner Detection)
    corners = detectHarrisFeatures(gray_img);
    features.corner_count = corners.Count;

    % 13. Histogram of Oriented Gradients (HOG)
    cellSize = [8 8]; % Cell size for HOG
    hog_features = extractHOGFeatures(gray_img, 'CellSize', cellSize);
    features.hog_length = mean(hog_features);

    % 14. ORB features
    points = detectORBFeatures(gray_img);
    features.orb_count = points.Count;
    
    % 15. Local Binary Patterns (LBP)
    % lbp_features = extractLBPFeatures(gray_img);
    % features.lbp = lbp_features;

    % 16. SURF features
    points = detectSURFFeatures(gray_img);
    features.surf_blobs = points.Count;
end
