
clear all % Clear MatLab variables


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NN_filename = 'NN_config_5_15.xlsx';
NN_sheet = 1;

I0 = [1,0,0,0,5,50,0.7,20];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,~,NN_info] = xlsread(NN_filename,NN_sheet);   % Read data from Excel file
disp(['NN file read (' NN_filename ').']);

data_set = NN_info{5,3};
input_count = NN_info{9,3};
output_count = NN_info{11,3};
activation_function = NN_info{12,3};
Hidden_Layers_count = NN_info{24,3};

% Load activation function
if strcmp(activation_function,'tanh')
    act_func_name = 'tanh';                 % Name of activation function
    act_func = @(x) tanh(x);                % Activation function
    act_func_deriv = @(x) 1+x.^2;           % Derivative of activation function
    act_func_range = [-1, 1];               % Range of function output
elseif strcmp(activation_function,'logistic')
    act_func_name = 'logistic';             % Name of activation function
    act_func = @(x) 1./(1+exp(-x));         % Activation function
    act_func_deriv = @(x) x.*(1.-x);        % Derivative of activation function
    act_func_range = [0,1];                 % Range of function output
else
    disp('Error: Invalid activation function.'); return
end

I0min = transpose(cell2mat(NN_info(27+Hidden_Layers_count:27+Hidden_Layers_count+input_count-1,2)));
I0max = transpose(cell2mat(NN_info(27+Hidden_Layers_count:27+Hidden_Layers_count+input_count-1,3)));
Otruemin = transpose(cell2mat(NN_info(27+Hidden_Layers_count:27+Hidden_Layers_count+output_count-1,4)));
Otruemax = transpose(cell2mat(NN_info(27+Hidden_Layers_count:27+Hidden_Layers_count+output_count-1,5)));

Hidden_Layers = zeros(1,Hidden_Layers_count);
for i = 1:Hidden_Layers_count
    Hidden_Layers(i) = NN_info{24+i,3}; % Nodes of each hidden layer without bias
end
Layers = horzcat(Hidden_Layers,output_count); % Layers including output buffer

% Weight matrices. Each cell represents a layer of the NN
W = cell(1,size(Layers,2));         % Weights between all layers

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
% CALCULATE OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Scale input to range [-1,1]
X0 = 2*(I0-I0min)./(I0max-I0min)-1; % Output of input buffer
X0(:,size(X0,2)+1) = 1;             % Add bias signal (1)
 
[O,~] = NN_calc_output(Layers,X0,W,Otruemin,Otruemax,act_func,act_func_range);

% Display output
disp('Output:');
if strcmp(data_set,'upperstruct')
    disp([num2str(ceil(O(1)*100)/100) ' (total concrete)']);
    disp([num2str(ceil(O(2)*100)/100) ' (total reinforcement)']);
    disp([num2str(ceil(O(3)*100)/100) ' (total structural steel)']);
    disp([num2str(ceil(O(4)*100)/100) ' (total formwork)']);
elseif strcmp(data_set,'foundation')
    disp([num2str(ceil(O(1)*100)/100) ' (total concrete)']);
    disp([num2str(ceil(O(2)*100)/100) ' (max reinforcement)']);
else
    for i = 1:output_count
        disp(ceil(O(i)*100/100));
    end
end

