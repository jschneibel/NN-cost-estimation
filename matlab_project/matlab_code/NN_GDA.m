
clear all % Clear MatLab variables


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input data set
input_filename = 'Data_foundation_var1.xlsx';  % Excel file containing data
data_set = 'foundation';               % 'upperstruct' or 'foundation'
input_sheet = 1;                        % Sheet in the Excel file
%input_range_string = 'A2:L4507';        % Excel range for upperstruct
input_range_string = 'A2:I527';        % Excel range for foundation
input_range = [1:7];                    % Columns of input variables
input_range_type = [1:2];               % Columns of binary inputs (building types)
output_range = [8:9];                  % Columns of output variables

% Output file
output_filename = 'NN_batch_foundation_var1_tanh_2.xlsx';   % Output filename '*.xlsx'
output_sheet = 1;                       % Sheet in the Excel file
output_comment = '';                    % Comment to add to the Excel file

% Activation function (for all layers)
activation_function = 'tanh';           % 'tanh' or 'logistic'
% Number of training and test samples *per building type* (input_range_type)
n = 150;                                % Training samples
ntest = 50;                            % Test samples
% Learning rate eta
p_eta_start = 2.0;      % eta_start for the first iteration of middle loop
p_eta_end = 0.01;       % eta_start for the last iteration of middle loop
eta_ratio = 10;         % Ratio eta_start/eta_end of inner loop
% Learning momentum mu
mu = 0.9;
% Number of NNs per layout to optimize (the best one will be stored)
NN_iterations = 100;                    % Iterations of middle loop
% Number of learning iterations for each NN
GDA_iterations = 400;                   % Iterations of inner loop
% Maximum number of nodes in each hidden layer (without bias, buffer layers), maximum 2 layers.
max_nodes = [];                         % If empty: [2*input_count,2*input_count]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check if output file is open (don't open output file until program ends)
try xlswrite(output_filename,{'temp'},output_sheet,'A1');
catch; disp(['Error: Close output file (' output_filename ') before running the script.']); return
end

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

input_count = length(input_range);      % Number of input variables
output_count = length(output_range);    % Number of output variables

% Generate NN layouts to optimize
if isempty(max_nodes)
    max_nodes = [2*input_count,2*input_count];
end
max_nodes = ceil(max_nodes);
if length(max_nodes) == 1 && max_nodes(1) > 0
    for i = 1:max_nodes(1)
        NN_store_layers{i} = [i];
        NN_store_xlswrite(i,:) = [i,0];
    end
elseif length(max_nodes) == 2 && max_nodes(1) > 0 && max_nodes(2) > 0
    for i = 1:max_nodes(1)
        for j = 1:max_nodes(2)+1
            if j == 1
                NN_store_layers{(i-1)*(max_nodes(2)+1)+j} = [i];
                NN_store_xlswrite((i-1)*(max_nodes(2)+1)+j,:) = [i,0];
            else
                NN_store_layers{(i-1)*(max_nodes(2)+1)+j} = [i,j-1];
                NN_store_xlswrite((i-1)*(max_nodes(2)+1)+j,:) = [i,j-1];
            end
        end
    end
else
    disp('Error: Invalid max_nodes.'); return
end

% Read input data from Excel file
Data = xlsread(input_filename,input_sheet,input_range_string);
disp(['Input file read (' input_filename ').']);

minima = min(Data);                     % Minima of the columns
maxima = max(Data);                     % Maxima of the columns

I0 = zeros(length(input_range_type)*n,input_count);              % Inputs of training samples
Otrue = zeros(length(input_range_type)*n,output_count);          % Desired outputs of training samples
I0test = zeros(length(input_range_type)*ntest,input_count);      % Inputs of test samples
Otruetest = zeros(length(input_range_type)*ntest,output_count);  % Desired outputs of test samples

% Select n and ntest exclusive random samples for each type of structure
Data = Data(randperm(size(Data,1)),:);  % Scramble samples
[r,c,v] = find(Data(:,input_range_type));
for j = 1:length(input_range_type)
    index = r(c==j);
    
    I0((j-1)*n+1:j*n,:) = Data(index(1:n),input_range);
    Otrue((j-1)*n+1:j*n,:) = Data(index(1:n),output_range);
    
    I0test((j-1)*ntest+1:j*ntest,:) = Data(index(n+1:n+ntest),input_range);
    Otruetest((j-1)*ntest+1:j*ntest,:) = Data(index(n+1:n+ntest),output_range);
end

n = size(I0,1);            % Total number of training samples
ntest = size(I0test,1);    % Total number of test samples

% Scale inputs to range [-1,1]
I0min = ones(n,1)*minima(:,input_range);    % Minima of training inputs
I0max = ones(n,1)*maxima(:,input_range);    % Maxima of  training outputs
X0 = 2*(I0-I0min)./(I0max-I0min)-1;         % Training outputs of input buffer
X0(:,size(X0,2)+1) = 1;                     % Add bias signal 1
I0testmin = ones(ntest,1)*minima(:,input_range);  % Minima of test inputs
I0testmax = ones(ntest,1)*maxima(:,input_range);  % Maxima of  test outputs
X0test = 2*(I0test-I0testmin)./(I0testmax-I0testmin)-1; % Test outputs of input buffer
X0test(:,size(X0test,2)+1) = 1;             % Add bias signal 1

% Calculate the desired outputs of the output buffer (scaled to act_func_range)
Otruemin = ones(n,1)*minima(:,output_range); % Minima of desired training outputs
Otruemax = ones(n,1)*maxima(:,output_range); % Maxima of desired training outputs
Xhtrue = (Otrue-Otruemin)./(Otruemax-Otruemin)*(act_func_range(2)-act_func_range(1))+act_func_range(1); % Desired training outputs of output buffer
Otruetestmin = ones(ntest,1)*minima(:,output_range); % Minima of desired test outputs
Otruetestmax = ones(ntest,1)*maxima(:,output_range); % Maxima of desired test outputs
Xhtruetest = (Otruetest-Otruetestmin)./(Otruetestmax-Otruetestmin)*(act_func_range(2)-act_func_range(1))+act_func_range(1); % Desired test outputs of output buffer

% Variables used to write the output Excel file
NN_store_eta = zeros(length(NN_store_layers),2);    % eta_start and eta_end
NN_store_E = zeros(length(NN_store_layers),1);      % Training error
NN_store_Etest = zeros(length(NN_store_layers),1);  % Test error

time_start = tic; % Track time

for q = 1:length(NN_store_layers)      % OUTER LOOP: Change NN layout
    
    Hidden_Layers = NN_store_layers{q};            % Nodes in each hidden layer
    Layers = horzcat(Hidden_Layers,size(Otrue,2)); % Layers including output buffer

    
    % Weight matrices. Each cell represents a layer of the NN
    W = cell(1,size(Layers,2));         % Weights between all layers
    dW = cell(1,size(Layers,2));        % Weight change for all layers of current iteration
    dWold = cell(1,size(Layers,2));     % Weight change of last iteration

    for p = 1:NN_iterations             % MIDDLE LOOP: New weights and eta

        % eta of the first and last iteration of the inner loop
        eta_start = p_eta_start/(p_eta_start/p_eta_end)^((p-1)/(NN_iterations-1));
        eta_end = eta_start/eta_ratio;
        
        % Initialize random weights in the range [-1,1]
        g = 1;
        for nodes = Layers
            if g == 1
                W{g} = rand(size(X0,2),nodes)*2-1;
            else
                W{g} = rand(size(W{g-1},2)+1,nodes)*2-1;
            end
            g = g+1;
        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % GRADIENT DESCENT ALGORITHM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for i = 1:GDA_iterations   % INNER LOOP: Iterate learning algorithm

            % Calculate training outputs of output buffer
            [~,X] = NN_calc_output(Layers,X0,W,Otruemin,Otruemax,act_func,act_func_range);

            % Back-propagation
            if i == 1
                for k = 1:length(W)
                    dWold{k} = zeros(size(W{k}));
                end
            end
            eta = eta_start/(eta_start/eta_end)^((i-1)/(GDA_iterations-1)); % eta of current iteration
            [W,dWold] = NN_GDA_step(Layers,X0,X,Xhtrue,W,dWold,n,eta,mu,act_func_deriv); % Update weights, keep last weight changes
        end

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % NN EVALUATION (TEST PERFORMANCE) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Calculate test outputs of output buffer
        [~,Xtest] = NN_calc_output(Layers,X0test,W,Otruetestmin,Otruetestmax,act_func,act_func_range);

        % Calculate NN test error
        Etest = sum(sum((Xtest{end}-Xhtruetest).^2))/ntest/(act_func_range(2)-act_func_range(1))^2;
        
        % Store best performing NN parameters of the given NN layout
        if Etest < NN_store_Etest(q) || p == 1
            NN_store_eta(q,:) = [eta_start, eta_end];
            E = sum(sum((X{end}-Xhtrue).^2))/n/(act_func_range(2)-act_func_range(1))^2; % Training error
            NN_store_E(q) = E;
            NN_store_Etest(q) = Etest;
        end

    end

    % Track and display time and script progress
    time_elapsed = toc(time_start);
    time_left = (length(NN_store_layers)-q)/q*time_elapsed;
    time_elapsed = [num2str(floor(time_elapsed/60)) ':' num2str(floor(time_elapsed-60*floor(time_elapsed/60)))];
    time_left = [num2str(floor(time_left/60)) ':' num2str(floor(time_left-60*floor(time_left/60)))];
    disp(['Progress: ' num2str(round(q/length(NN_store_layers)*10000)/100) '% (' time_elapsed ' elapsed, ~' time_left ' left)']);
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STORE RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Store program input parameters
% and parameters/errors of best performing NN of each NN configuration
NN_store_xlswrite = horzcat(NN_store_xlswrite,NN_store_eta,NN_store_E,NN_store_Etest);
NN_store_xlswrite = vertcat({...
    'Gradient Descent Algorithm','','','','','';...
    '','','','','','';...
    'input_filename','',input_filename,'','','';...
    'input_sheet','',input_sheet,'','','';...
    'data_set','',data_set,'','','';...
    'input_range_string','',input_range_string,'','','';...
    'input_range','',num2str(input_range),'','','';...
    'input_range_type','',num2str(input_range_type),'','','';...
    'output_range','',num2str(output_range),'','','';...
	'activation_function','',activation_function,'','','';...
    'n','',n/length(input_range_type),'','','';...
    'ntest','',ntest/length(input_range_type),'','','';...
    'p_eta_start','',p_eta_start,'','','';...
    'p_eta_end','',p_eta_end,'','','';...
    'eta_ratio','',eta_ratio,'','','';
    'mu','',mu,'','','';...
    'NN_iterations','',NN_iterations,'','','';...
    'GDA_iterations','',GDA_iterations,'','','';...
    'max_nodes','',['[' num2str(max_nodes(1)) ',' num2str(max_nodes(2)) ']'],'','','';...
    '','','','','','';...
    'Layer 1', 'Layer 2','eta_start','eta_end','E','Etest'},...
    num2cell(NN_store_xlswrite));
    
xlswrite(output_filename,NN_store_xlswrite,output_sheet,'A1');
xlswrite(output_filename,{'Comment:';output_comment},output_sheet,'F3');

disp(['Output file written (' output_filename ').']);

