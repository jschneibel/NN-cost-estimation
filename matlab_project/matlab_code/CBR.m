
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

I = [1,0,0,0,4,55,0.3,25];          % New case (inputs)
W = [1,1,1,1,1,1,1,1];              % New case (weights)

tolerance = 0.1;                    % Matching tolerance for numerical values


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

input_range = [input_range_ooal,input_range_num];
input_count = length(input_range);      % Number of input variables
output_count = length(output_range);    % Number of output variables

if length(I) ~= input_count
    disp(['Error: length(I) is ' num2str(length(I)) '. Expected: ' num2str(input_count)]); return
end
if length(W) ~= input_count
    disp(['Error: length(W) is ' num2str(length(W)) '. Expected: ' num2str(input_count)]); return
end
W = abs(W);
sumW = sum(W);

% Read input data from Excel file
Data = xlsread(input_filename,input_sheet,input_range_string);
disp(['Input file read (' input_filename ').']);

case_count = size(Data,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCULATE OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S = zeros(case_count,1);    % Percentage similarity for each case
for n = 1:case_count
    j = 1;
    for i = input_range_ooal
        if Data(n,i) == I(j)
            S(n) = S(n)+W(j);
        end
        j = j+1;
    end
    for i = input_range_num
        if abs((Data(n,i)-I(j))/I(j)) <= tolerance
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

% Display output
for i = 1:output_count
    disp(['Output ' num2str(i) ': ' num2str(ceil(O(i)*100)/100)]);
end

