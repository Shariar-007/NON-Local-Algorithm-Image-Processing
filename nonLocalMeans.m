% Load a demo image provided by MATLAB
img = imread('cameraman.tif');
I = im2double(img); % Convert to double for calculations


% Add Gaussian noise to the image
sigma = 0.1; % Standard deviation of noise
noisy_img = imnoise(I, 'gaussian', 0, sigma^2);

% Display the noisy image
figure_handle1 = figure;
imshow(noisy_img);
title('Noisy Image');
saveas(figure_handle1, 'Noisy Image.png');


% Define parameters
h = 10 * sigma; % Filtering parameter, proportional to noise level
patch_size = 7; % Size of the neighborhood patch
search_window = 21; % Size of the search window

% Pad the image for boundary handling
half_search = floor(search_window / 2);
half_patch = floor(patch_size / 2);
padded_img = padarray(noisy_img, [half_search, half_search], 'symmetric');

% Gaussian kernel for spatial weighting within patches
[X, Y] = meshgrid(-half_patch:half_patch, -half_patch:half_patch);
gaussian_kernel = exp(-(X.^2 + Y.^2) / (2 * (patch_size / 2)^2));
gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:)); % Normalize

% Initialize the denoised image
[m, n] = size(noisy_img);
denoised_img = zeros(m, n);

% Iterate over each pixel in the image
for i = 1:m
    for j = 1:n
        % Map the pixel position to the padded image
        i1 = i + half_search;
        j1 = j + half_search;

        % Extract the search window around the current pixel
        search_region = padded_img(i1-half_search:i1+half_search, j1-half_search:j1+half_search);

        % Extract the reference patch
        reference_patch = padded_img(i1-half_patch:i1+half_patch, j1-half_patch:j1+half_patch);

        % Compute weights for each patch in the search region
        weights = zeros(search_window, search_window);
        for p = 1:search_window
            for q = 1:search_window
                % Map the center of the current patch in the search region
                p_center = p; % Current position in the search region
                q_center = q;

                % Extract the current patch centered at (p, q)
                if p_center-half_patch > 0 && p_center+half_patch <= search_window && ...
                   q_center-half_patch > 0 && q_center+half_patch <= search_window
                    current_patch = search_region(p_center-half_patch:p_center+half_patch, ...
                                                  q_center-half_patch:q_center+half_patch);

                    % Compute the weighted Euclidean distance
                    distance = sum(sum(gaussian_kernel .* (reference_patch - current_patch).^2));
                    weights(p, q) = exp(-distance / h^2);
                end
            end
        end

        % Normalize weights
        weights = weights / sum(weights(:));

        % Compute the denoised pixel value
        denoised_img(i, j) = sum(sum(weights .* search_region));
    end
end

% Display the denoised image
figure_handle = figure;
imshow(denoised_img);
title('Denoised Image with Non-Local Means');

saveas(figure_handle, 'Denoised Image with Non-Local Means.png');

% Calculate Mean Square Error (MSE)
mse = mean((I(:) - denoised_img(:)).^2);
disp(['Mean Square Error (MSE): ', num2str(mse)]);