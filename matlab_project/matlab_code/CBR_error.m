
clear all % Clear MatLab variables


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Case base
input_filename = 'Data_upperstruct_var2.xlsx';  % Excel file containing data
input_sheet = 1;                        % Sheet in the Excel file
input_range_string = 'A2:L4507';        % Excel range for upperstruct
%input_range_string = 'A2:J527';        % Excel range for foundation
input_range_ooal = [1:4];               % Columns of binary inputs (building types)
input_range_num = [5:8];                % Columns of numerical inputs
output_range = [9:12];                  % Columns of output variables

ntest = 100;

W = [57.9,103.1,92.3,58.9,67.2,19.8,3.7,4.5];   % Weights to be evaluated

tolerance = 0.1;                % Matching tolerance for numerical values


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

input_range = [input_range_ooal,input_range_num];
input_count = length(input_range);      % Number of input variables
output_count = length(output_range);    % Number of output variables

if length(W) ~= input_count
    disp(['Error: length(W) is ' num2str(length(W)) '. Expected: ' num2str(input_count)]); return
end
W = abs(W);
sumW = sum(W);

% Read input data from Excel file
Data = xlsread(input_filename,input_sheet,input_range_string);
disp(['Input file read (' input_filename ').']);

minima = min(Data);                     % Minima of the columns
maxima = max(Data);                     % Maxima of the columns

Itest = zeros(length(input_range_ooal)*ntest,input_count);            % Inputs of test samples
Otest = zeros(length(input_range_ooal)*ntest,output_count);           % Outputs of test samples
Otruetest = zeros(length(input_range_ooal)*ntest,output_count);       % Desired outputs of test samples


% Select ntest random samples for each type of structure and delete them
% from the case base data
Data = Data(randperm(size(Data,1)),:);  % Scramble samples
Data_keep = true(size(Data,1),1);
[r,c,v] = find(Data(:,input_range_ooal));
for j = 1:length(input_range_ooal)
    index = r(c==j);
    
    Itest((j-1)*ntest+1:j*ntest,:) = Data(index(1:ntest),input_range);
    Otruetest((j-1)*ntest+1:j*ntest,:) = Data(index(1:ntest),output_range);
    Data_keep(index(1:ntest)) = false;
end
Data = Data(find(Data_keep),:);

ntest = size(Itest,1);

Otruetestmin = ones(ntest,1)*minima(:,output_range); % Minima of desired test outputs
Otruetestmax = ones(ntest,1)*maxima(:,output_range); % Maxima of desired test outputs

case_count = size(Data,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCULATE OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for m = 1:ntest % Calculate output of each test sample

    S = zeros(case_count,1);    % Percentage similarity for each case
    for n = 1:case_count
        j = 1;
        for i = input_range_ooal
            if Data(n,i) == Itest(m,j)
                S(n) = S(n)+W(j);
            end
            j = j+1;
        end
        for i = input_range_num
            if abs((Data(n,i)-Itest(m,j))/Itest(m,j)) <= tolerance
                S(n) = S(n)+W(j);
            end
            j = j+1;
        end
        S(n) = S(n)/sumW;
    end

    % Find cases with highest percentage similarity
    [S,Index] = sort(S,'descend');
    S = [S,Index];
    S = S(S(:,1)==S(1,1),:);

    % Average outputs of cases with highest percentage similarity
    O = zeros(1,output_count);
    for n = transpose(S(:,2))
        O = O+Data(n,output_range);
    end
    O = O/size(S,1);

    Otest(m,:) = O;
    disp(['Progress: ' num2str(m) '/' num2str(ntest)]);
end

% Display error
Etest = sum(sum(((Otest-Otruetest)./(Otruetestmax-Otruetestmin)).^2))/ntest

