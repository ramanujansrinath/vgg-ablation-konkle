% --- getPixelFeatures: extract low-level visual features from a single image ---
% Returns a struct containing 15+ scalar features spanning luminance statistics,
% frequency content, texture, edge structure, and keypoint counts; fields that
% are later dropped (exempName, color_hist_red/green, aspect_ratio,
% boundingBox_size) are still computed here for completeness.
function features = getPixelFeatures(img,imgName)
    features.exempName = imgName;

    % Convert to grayscale for features that operate on luminance only.
    if size(img,3) == 3
        gray_img = rgb2gray(img);
    else
        gray_img = img;
    end

    % 1. Contrast — global luminance variability of the image.
    features.contrast = std2(gray_img);

    % 2. Orientation — mean gradient direction across all pixels (in degrees).
    [gradient_x, gradient_y] = gradient(double(gray_img));
    orientation = atan2d(gradient_y, gradient_x);
    features.orientation = mean(orientation(:));  % Mean orientation

    % 3. Curvature content — spread of the Laplacian, capturing second-order
    %    spatial variation (bending/curving structure).
    laplacian = del2(double(gray_img));
    features.curvature_content = std2(laplacian);

    % 4. Sharpness — variance of the Laplacian; high values indicate sharp edges.
    features.sharpness = var(double(laplacian(:)));

    % 5. Intensity distribution shape — skewness and kurtosis of pixel values.
    features.skewness = skewness(double(gray_img(:)));
    features.kurtosis = kurtosis(double(gray_img(:)));

    % 6. Fourier energy — mean squared magnitude in the frequency domain,
    %    summarising overall spatial frequency content.
    F = fft2(double(gray_img));
    F_shifted = fftshift(F);
    features.fourier_energy = mean(abs(F_shifted(:)).^2);

    % 7. Relative color channel means — each channel mean divided by the overall
    %    luminance mean, giving a hue-like summary independent of brightness.
    if size(img, 3) == 3
        features.color_hist_red = mean(img(:,:,1),'all')./mean(gray_img,'all'); % histcounts(img(:,:,1), 256);
        features.color_hist_green = mean(img(:,:,2),'all')./mean(gray_img,'all');
        features.color_hist_blue = mean(img(:,:,3),'all')./mean(gray_img,'all');
    else
        features.color_hist_red = mean(gray_img,'all');
        features.color_hist_green = mean(gray_img,'all');
        features.color_hist_blue = mean(gray_img,'all');
    end
    

    % 8. GLCM texture statistics — contrast, correlation, energy, and homogeneity
    %    derived from the gray-level co-occurrence matrix.
    glcm = graycomatrix(gray_img);
    stats = graycoprops(glcm, {'contrast', 'correlation', 'energy', 'homogeneity'});
    features.texture_contrast = stats.Contrast;
    features.texture_correlation = stats.Correlation;
    features.texture_energy = stats.Energy;
    features.texture_homogeneity = stats.Homogeneity;

    % 9. Vertical symmetry — mean absolute pixel difference between the image
    %    and its vertical mirror; lower values indicate greater top-bottom symmetry.
    flipped_img = flipud(gray_img);  % Flip image upside down
    symmetry = sum(sum(abs(gray_img - flipped_img))) / numel(gray_img);  % Measure symmetry
    features.symmetry = symmetry;

    % 10. Edge density — fraction of pixels classified as edges by the Sobel detector.
    edges = edge(gray_img, 'Sobel');
    edge_density = sum(edges(:)) / numel(edges);  % Fraction of edge pixels
    features.edge_density = edge_density;

    % 11. Bounding box geometry — aspect ratio and relative area of the largest
    %     connected component in the binarized image.
    bw = imbinarize(gray_img);  % Binary image
    stats = regionprops(bw, 'BoundingBox');
    if ~isempty(stats)
        boundingBox = stats(1).BoundingBox;
        features.aspect_ratio = boundingBox(3) / boundingBox(4);
        features.boundingBox_size = boundingBox(3)*boundingBox(4)/numel(gray_img);
    else % No object found
        features.aspect_ratio = NaN;
        features.boundingBox_size = NaN;
    end
    

    % 12. Harris corner count — number of interest points detected by the
    %     Harris corner detector, reflecting local geometric complexity.
    corners = detectHarrisFeatures(gray_img);
    features.corner_count = corners.Count;

    % 13. HOG summary — mean of the Histogram of Oriented Gradients descriptor,
    %     capturing average gradient orientation energy across the image.
    cellSize = [8 8]; % Cell size for HOG
    hog_features = extractHOGFeatures(gray_img, 'CellSize', cellSize);
    features.hog_length = mean(hog_features);

    % 14. ORB keypoint count — number of oriented FAST keypoints, reflecting
    %     the density of repeatable local features.
    points = detectORBFeatures(gray_img);
    features.orb_count = points.Count;

    % 15. Local Binary Patterns (disabled — would expand feature vector significantly)
    % lbp_features = extractLBPFeatures(gray_img);
    % features.lbp = lbp_features;

    % 16. SURF blob count — number of SURF interest regions, another measure
    %     of local structural richness at multiple scales.
    points = detectSURFFeatures(gray_img);
    features.surf_blobs = points.Count;
end
