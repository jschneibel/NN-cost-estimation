
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
output_filename = 'NN_config_1_1.xlsx';   % Output filename '*.xlsx'
output_sheet = 1;                       % Sheet in the Excel file
output_comment = '';                    % Comment to add to the Excel file

% Activation function (for all layers)
activation_function = 'logistic';           % 'tanh' or 'logistic'
% Number of training and test samples *per building type* (input_range_type)
n = 150;                                % Training samples
ntest = 50;                            % Test samples
% Learning rate eta
eta_start = 12.5;         % eta of the first iteration of the inner loop
eta_end = eta_start/10;  % eta of the last iteration of the inner loop
% Learning momentum mu
mu = 0.9;
% Number of NNs to optimize (the best one will be stored)
NN_iterations = 100;                    % Iterations of middle loop
% Number of learning iterations for each NN
GDA_iterations = 400;                   % Iterations of inner loop
% Nodes in each hidden layer (without bias, buffer layers)
Hidden_Layers = [14,6];                 % Row vector (no 0 elements)


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

% Read input data from Excel file
Data = xlsread(input_filename,input_sheet,input_range_string);
disp(['Input file read (' input_filename ').']);

minima = min(Data);                     % Minima of the columns
maxima = max(Data);                     % Maxima of the columns

I0 = zeros(length(input_range_type)*n,input_count);                    % Inputs of training samples
Otrue = zeros(length(input_range_type)*n,output_count);                % Desired outputs of training samples
I0test = zeros(length(input_range_type)*ntest,input_count);            % Inputs of test samples
Otruetest = zeros(length(input_range_type)*ntest,output_count);        % Desired outputs of test samples

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

% Nodes of the layers, including output buffer
Layers = horzcat(Hidden_Layers,size(Otrue,2));

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
  
% Weight matrices. Each cell represents a layer of the NN
W = cell(1,size(Layers,2));         % Weights between all layers
dW = cell(1,size(Layers,2));        % Weight change for all layers of current iteration
dWold = cell(1,size(Layers,2));     % Weight change of last iteration

% Generate NN_iterations NNs of the chosen layout starting with newly
% randomized weights to find the one with the lowest test error
time_start = tic; % Track time
for p = 1:NN_iterations

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


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % GRADIENT DESCENT ALGORITHM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for i = 1:GDA_iterations   % Iterate learning algorithm

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

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % NN EVALUATION (TEST PERFORMANCE) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Calculate test outputs of output buffer
    [Otest,Xtest] = NN_calc_output(Layers,X0test,W,Otruetestmin,Otruetestmax,act_func,act_func_range);

    % Calculate NN test error
    Etest = sum(sum((Xtest{end}-Xhtruetest).^2))/ntest/(act_func_range(2)-act_func_range(1))^2;

    Etest

    % Store NN with best performance
    if p == 1 || Etest < NN_store_Etest
        NN_store_weights = W;
        E = sum(sum((X{end}-Xhtrue).^2))/n/(act_func_range(2)-act_func_range(1))^2; % Training error
        NN_store_E = E;
        NN_store_Etest = Etest;
    end

    % Track and display time and script progress
    time_elapsed = toc(time_start);
    time_elapsed = [num2str(floor(time_elapsed/60)) ':' num2str(floor(time_elapsed-60*floor(time_elapsed/60)))];
    disp([num2str(p) ' NNs tested, ' time_elapsed ' elapsed.']);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STORE RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp([NN_store_E,NN_store_Etest])

% Store input parameters and test error of chosen NN
NN_store_xlswrite = {...
    'Gradient Descent Algorithm','','';...
    '','','';...
    'input_filename','',input_filename;...
    'input_sheet','',input_sheet;...
    'data_set','',data_set;...
    'input_range_string','',input_range_string
    'input_range','',num2str(input_range);...
    'input_range_type','',num2str(input_range_type);...
    'input_count','',input_count;...
    'output_range','',num2str(output_range);...
    'output_count','',output_count;...
	'activation_function','',activation_function;...
    'n','',n/length(input_range_type);...
    'ntest','',ntest/length(input_range_type);...
    'eta_start','',eta_start;...
    'eta_end','',eta_end;...
    'mu','',mu;...
    'NN_iterations','',NN_iterations;...
    'GDA_iterations','',GDA_iterations;...
    '','','';...
    'E','',NN_store_E
    'Etest','',NN_store_Etest;...
    '','','';...
    'Hidden layers','',length(Hidden_Layers)};

h = 1;
for nodes = Hidden_Layers
    NN_store_xlswrite = vertcat(NN_store_xlswrite,{['Nodes in layer ' num2str(h)],'',nodes});
    h = h+1;
end

NN_store_xlswrite = horzcat(NN_store_xlswrite,cell(size(NN_store_xlswrite,1),2));

% Store maxima and minima of data
NN_store_xlswrite = vertcat(NN_store_xlswrite,...
    {'','','','',''},...
    {'input_range','I0min','I0max','Otruemin','Otruemax'});

if input_count >= output_count
    NN_store_xlswrite = vertcat(NN_store_xlswrite,...
        horzcat(...
                num2cell(transpose(input_range)),...
                num2cell(transpose(I0min(1,:))),...
                num2cell(transpose(I0max(1,:))),...
        vertcat(num2cell(transpose(Otruemin(1,:))),cell(input_count-output_count,1)),...
        vertcat(num2cell(transpose(Otruemax(1,:))),cell(input_count-output_count,1)))...
        );
else
    NN_store_xlswrite = vertcat(NN_store_xlswrite,...
        horzcat(...
        vertcat(num2cell(transpose(input_range)),cell(output_count-input_count,1)),...
        vertcat(num2cell(transpose(I0min(1,:))),cell(output_count-input_count,1)),...
        vertcat(num2cell(transpose(I0max(1,:))),cell(output_count-input_count,1)),...
                num2cell(transpose(Otruemin(1,:))),...
                num2cell(transpose(Otruemax(1,:))))...
        );
end
    
xlswrite(output_filename,NN_store_xlswrite,output_sheet,'A1');

% Store weight matrices of best performing NN
row_target = size(NN_store_xlswrite,1)+2;
for k = 1:length(W)
    xlswrite(output_filename,{['Weights of layer ' num2str(k)]},output_sheet,['A' num2str(row_target)]);
    
    row_target = row_target+1;
    xlswrite(output_filename,NN_store_weights{k},output_sheet,['A' num2str(row_target)]);
    
    row_target = row_target+1+size(W{k},1);
end

xlswrite(output_filename,{'Comment:';output_comment},output_sheet,'F3');

disp(['Output file written (' output_filename ').']);

