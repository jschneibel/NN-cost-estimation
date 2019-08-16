
clear all % Clear MatLab variables


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NN_filename = 'NN_config_13_0.xlsx';
NN_sheet = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,~,NN_info] = xlsread(NN_filename,NN_sheet);   % Read data from Excel file
disp('NN file read.');

input_count = NN_info{9,3};
output_count = NN_info{11,3};
Hidden_Layers_count = NN_info{24,3};

Hidden_Layers = zeros(1,Hidden_Layers_count);
for i = 1:Hidden_Layers_count
    Hidden_Layers(i) = NN_info{24+i,3}; % Nodes of each hidden layer without bias
end
Layers = horzcat(Hidden_Layers,output_count); % Layers including output buffer

input_range = cell2mat(NN_info(27+Hidden_Layers_count:27+Hidden_Layers_count+input_count-1,1));

% Weight matrices. Each cell represents a layer of the NN
W = cell(1,size(Layers,2));         % Weights between all layers
% Importance vector
S = cell(1,size(Layers,2));
S{end} = ones(Layers(end),1);

W_row_start = 0;
W_row_end = 0;
for i = 1:length(Layers)
    if i == 1
        W_row_start = 29+Hidden_Layers_count+max(input_count,output_count);
        W_row_end = 29+Hidden_Layers_count+2*max(input_count,output_count);
    else
        W_row_start = W_row_end+3;
        W_row_end = W_row_start+Layers(i-1);
    end
    W_column_end = Layers(i);
    
    W{i} = cell2mat(NN_info(W_row_start:W_row_end,1:W_column_end));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCULATE IMPORTANCE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for j = fliplr(0:length(W)-1)
    if j == 0
        S0 = W{j+1}(1:end-1,:)*S{j+1};
    else
        S{j} = W{j+1}(1:end-1,:)*S{j+1};
    end
end

[~,Index] = sort(abs(S0),'descend');
[Sorted,Index] = sort(Index,'ascend');

disp('    input | importance | rank');
disp([input_range fix(S0*100)/100 Index])

