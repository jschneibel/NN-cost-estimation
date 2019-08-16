function [W,dW] = NN_GDA_step(Layers,X0,X,Xhtrue,W,dWold,n,eta,mu,act_func_der)

D = cell(1,size(Layers,2));                 % Error signals of all layers
nE = cell(1,size(Layers,2));                % Error gradients of all layers
dW = cell(size(W));                         % Weight changes of all layers

g = size(Layers,2);                         % Start at output buffer
for nodes = fliplr(Layers)                  % Go through each layer g
    if g == size(Layers,2)                  % Calculate error signals
        D{g} = act_func_der(X{g}).*(X{g}-Xhtrue);
    elseif g == size(Layers,2)-1
        D{g} = act_func_der(X{g}).*transpose((W{g+1}*transpose(D{g+1})));
    else
        D{g} = act_func_der(X{g}).*transpose((W{g+1}*transpose(D{g+1}(:,1:size(D{g+1},2)-1))));
    end

    if g == size(Layers,2)                  % Calculate error gradients
        nE{g} = transpose(X{g-1})*D{g}/n;
    elseif g ~= 1
        nE{g} = transpose(X{g-1})*D{g}(:,1:size(D{g},2)-1)/n;
    else
        nE{g} = transpose(X0)*D{g}(:,1:size(D{g},2)-1)/n;
    end

    dW{g} = -eta*nE{g}+mu*dWold{g};         % Calculate weight changes
    
    W{g} = W{g}+dW{g};                      % Update weights
    
    g = g-1;
end

end

    