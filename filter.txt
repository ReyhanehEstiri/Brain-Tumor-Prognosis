% Read the image
I = imread('https://static.hindawi.com/articles/cmmm/volume-2020/1405647/figures/1405647.fig.006b.jpg'); 
% Convert the image to double.
I = rgb2gray(im2double(I));
% Create the function you want to apply to the image — a geometric mean filter.
fun = @(x) geomean(x(:));
% Apply the filter to the image.
G = nlfilter(I,[5 5],fun); 

% Display the original image and the filtered image, side-by-side.
montage({I,G})
title('Original Image (Left) and Geometric-Mean Filtered Image (Right)')
