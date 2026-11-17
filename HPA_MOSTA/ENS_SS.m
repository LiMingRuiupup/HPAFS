function [FrontNo,N_fronts] = ENS_SS (ObjValues,operation)
% ENS-SS non-dominated sorting.
% Inputs: ObjValues - objective values [rho, E]; operation - optional sorting mode.
% Outputs: FrontNo - Pareto front numbers; N_fronts - number of fronts.

[N_solutions,N_obj] = size(ObjValues);   
    Fronts = cell(N_solutions,1);              % Solution index sets for the Pareto fronts
    N_fronts = 0;                    
    % Sort the objective vectors lexicographically.
    [~,sort_order] = sortrows(ObjValues);  
    % Initialize the first Pareto front.
    Fronts{1} = sort_order(1);
    N_fronts = N_fronts + 1;
    % Assign the remaining solutions to Pareto fronts.
    for i = 2 : N_solutions
        k = 1;	
        while 1
            % Check dominance against the current front.
            for j = size(Fronts{k},2) : -1 : 1
                x = 2;
                for j2 = 2 : N_obj
                    if ObjValues(sort_order(i),j2) < ObjValues(Fronts{k}(j),j2)
                        x = 0;
                        break;
                    end
                end
                if x == 2 || N_obj == 2
                    break;
                end
            end
            if x ~= 2   % The current solution is not dominated by this front
                Fronts{k} = [Fronts{k},sort_order(i)];	
                break;
            else        % A solution in this front dominates the current solution
                if k < N_fronts
                    k = k + 1;
                else
                    N_fronts = N_fronts + 1;
                    Fronts{N_fronts} = sort_order(i); 	% Create a new Pareto front
                    break;
                end
            end
        end
    end
    % Assign Pareto front numbers to all solutions.
    FrontNo = zeros(1,N_solutions) + inf;
    if nargin>1 && strcmp(operation,'half')
        for i = 1 : N_fronts
            if sum(FrontNo ~= inf) >200
                N_fronts = i - 1;
                break;
            else
                FrontNo(Fronts{i}) = i;
            end
        end
    else
        for i = 1 : N_fronts
        	FrontNo(Fronts{i}) = i;
        end
    end
end

