% Define the folder containing PNG images
folderPath = ['./results/001-Full-seq/run-1_2025-05-31/For Report/LQ/']; 

% Get all PNG files
allFiles = dir(fullfile(folderPath, '*.png'));

% Suffixes to check
suffixes = {'input', 'output', 'original'};

% Initialize structures to hold image names and scores
scoreData = struct();

for s = 1:length(suffixes)
    suffix = suffixes{s};
    scoreData.(suffix).indices = [];
    scoreData.(suffix).piqeScores = [];
    scoreData.(suffix).niqeScores = [];
end

% Process each file
for i = 1:length(allFiles)
    fileName = allFiles(i).name;
    
    for s = 1:length(suffixes)
        suffix = suffixes{s};
        pattern = strcat(suffix, '(\d{3})\.png');
        match = regexp(fileName, pattern, 'tokens');
        
        if ~isempty(match)
            index = str2double(match{1}{1});
            imagePath = fullfile(folderPath, fileName);
            img = imread(imagePath);
            
            % Convert to grayscale if needed
            if size(img, 3) == 3
                imgGray = rgb2gray(img);
            else
                imgGray = img;
            end

            % Compute scores
            piqeScore = piqe(imgGray);
            niqeScore = niqe(imgGray);

            % Store results
            scoreData.(suffix).indices(end+1) = index;
            scoreData.(suffix).piqeScores(end+1) = piqeScore;
            scoreData.(suffix).niqeScores(end+1) = niqeScore;
        end
    end
end

% Save to CSV files
for s = 1:length(suffixes)
    suffix = suffixes{s};
    [sortedIndices, sortOrder] = sort(scoreData.(suffix).indices);
    piqeSorted = scoreData.(suffix).piqeScores(sortOrder);
    niqeSorted = scoreData.(suffix).niqeScores(sortOrder);

    % Create a table
    T = table(sortedIndices', piqeSorted', niqeSorted', ...
        'VariableNames', {'Index', 'PIQE', 'NIQE'});

    % Write CSV
    csvFileName = fullfile(folderPath, [suffix '_scores.csv']);
    writetable(T, csvFileName);
end

disp('All scores computed and saved.');
