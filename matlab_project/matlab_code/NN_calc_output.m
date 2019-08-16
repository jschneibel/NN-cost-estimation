function [O,X] = NN_calc_output(Layers,X0,W,Otruemin,Otruemax,act_func,act_func_range)

I = cell(1,size(Layers,2));             % Incoming signals of all layers
X = cell(1,size(Layers,2));             % Outgoing signals of all layers

g = 1;                                  % Start at layer 1
for nodes = Layers                      % Go through each layer g
    if g == 1
        I{g} = X0*W{g};                 % Calculate inputs to layer 1
    else
        I{g} = X{g-1}*W{g};             % Calculate inputs to layer g
    end

    X{g} = act_func(I{g});              % Calculate outputs of layer g

    if g ~= size(Layers,2)              % Skip output buffer
        X{g}(:,size(X{g},2)+1) = 1;     % Add bias node (1) to layer g
        g = g+1;
    end
end

O = (X{end}-act_func_range(1)).*(Otruemax-Otruemin)/(act_func_range(2)-act_func_range(1))+Otruemin;   % Scale output

end

