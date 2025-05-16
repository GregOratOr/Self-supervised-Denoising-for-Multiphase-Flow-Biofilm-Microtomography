% Define source and destination paths
src_path = 'results/999/run-0_2025-05-14_/Evaluation Tests/';   % e.g., 'C:\images'
dst_path = './'; % e.g., 'C:\results'

% Get list of matching files (outputXXX.png)
files = dir(fullfile(src_path, '*.png'));

% Initialize array to store results
results = [];

% Loop through each file
for k = 1:length(files)
    filename = files(k).name;
        
    % Read and convert to grayscale if needed
    img = imread(fullfile(src_path, filename));
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Compute PIQE and NIQE scores
    piqe_score = piqe(img);
    niqe_score = niqe(img);
    
    % Store result: [index, PIQE, NIQE]
    results = [results; k, piqe_score, niqe_score];
    
end

% Sort results by index
results = sortrows(results, 1);

% Write to text file
output_file = fullfile(dst_path, 'quality_scores.txt');
fid = fopen(output_file, 'w');
fprintf(fid, 'Index\tPIQE_Score\tNIQE_Score\n');
fprintf(fid, '%03d \t%.4f \t%.4f\n', results');
fclose(fid);

disp('PIQE and NIQE scores written to:');
disp(output_file);