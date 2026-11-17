function ranked_pool = non_dominated_sort_and_crowding(pool, N_obj, n)
% Non-dominated sorting and crowding distance calculation.
% Inputs: pool - [Z, ObjValues], with ObjValues ordered as [rho, E];
%         N_obj - number of objectives; n - number of harmonic sampling points.
% Output: ranked_pool - ranked solution set [Z, ObjValues, FrontNo, D].

[N_pool, ~] = size(pool);
clear m
front = 1;
Fronts(front).indices = [];
individual = [];

for i = 1 : N_pool
    individual(i).domination_count = 0;% Number of solutions dominating the current solution
    individual(i).dominated_indices = [];% Indices of solutions dominated by the current solution
    for j = 1 : N_pool
        dom_less = 0;
        dom_equal = 0;
        dom_more = 0;
        for k = 1 : N_obj        % Compare objective values to determine dominance
            if (pool(i,n + k) < pool(j,n + k))  
                dom_less = dom_less + 1;
            elseif (pool(i,n + k) == pool(j,n + k))
                dom_equal = dom_equal + 1;
            else
                dom_more = dom_more + 1;
            end
        end
        if dom_less == 0 && dom_equal ~= N_obj % The comparison solution dominates the current solution
            individual(i).domination_count = individual(i).domination_count + 1;
        elseif dom_more == 0 && dom_equal ~= N_obj % The current solution dominates the comparison solution
            individual(i).dominated_indices = [individual(i).dominated_indices j];
        end
    end   
    if individual(i).domination_count == 0 % Identify the first Pareto front
        pool(i,N_obj + n + 1) = 1;
        Fronts(front).indices = [Fronts(front).indices i];
    end
end
% Construct the subsequent Pareto fronts.
while ~isempty(Fronts(front).indices)
   Q = []; % Indices of solutions in the next front
   for i = 1 : length(Fronts(front).indices)
       if ~isempty(individual(Fronts(front).indices(i)).dominated_indices)
        	for j = 1 : length(individual(Fronts(front).indices(i)).dominated_indices)% Traverse solutions dominated by the current solution
            	individual(individual(Fronts(front).indices(i)).dominated_indices(j)).domination_count = ...% Update the domination count
                	individual(individual(Fronts(front).indices(i)).dominated_indices(j)).domination_count - 1;
        	   	 if individual(individual(Fronts(front).indices(i)).dominated_indices(j)).domination_count == 0% Assign solutions with zero domination count to the next front
               		pool(individual(Fronts(front).indices(i)).dominated_indices(j),N_obj + n + 1) = ...% Assign the next front number
                        front + 1;
                    Q = [Q individual(Fronts(front).indices(i)).dominated_indices(j)];
                end
            end
       end
   end
   front =  front + 1;
   Fronts(front).indices = Q;
end

[temp,index_of_fronts] = sort(pool(:,N_obj + n + 1));% Sort by Pareto front number
for i = 1 : length(index_of_fronts)
    sorted_based_on_front(i,:) = pool(index_of_fronts(i),:);
end
current_index = 0;

% Compute the crowding distance.

for front = 1 : (length(Fronts) - 1)% Exclude the trailing empty front
    D_front = 0;
    front_pool = [];
    previous_index = current_index + 1;
    for i = 1 : length(Fronts(front).indices)
        front_pool(i,:) = sorted_based_on_front(current_index + i,:);% Extract the current front
    end
    current_index = current_index + i;% Update the front index boundary
    sorted_based_on_objective = [];% Temporary storage for sorting by objective value
    for i = 1 : N_obj
        [sorted_based_on_objective, index_of_objectives] = ...
            sort(front_pool(:,n + i));% Sort by the current objective value
        sorted_based_on_objective = [];
        for j = 1 : length(index_of_objectives)
            sorted_based_on_objective(j,:) = front_pool(index_of_objectives(j),:);
        end
        f_max = ...
            sorted_based_on_objective(length(index_of_objectives), n + i);
        f_min = sorted_based_on_objective(1, n + i);
        front_pool(index_of_objectives(length(index_of_objectives)),N_obj + n + 1 + i)...% Assign infinite crowding distance to boundary solutions
            = Inf;
        front_pool(index_of_objectives(1),N_obj + n + 1 + i) = Inf;
         for j = 2 : length(index_of_objectives) - 1% Compute normalized distance components for interior solutions
            next_obj  = sorted_based_on_objective(j + 1,n + i);
            previous_obj  = sorted_based_on_objective(j - 1,n + i);
            if (f_max - f_min == 0)
                front_pool(index_of_objectives(j),N_obj + n + 1 + i) = Inf;
            else
                front_pool(index_of_objectives(j),N_obj + n + 1 + i) = ...
                     (next_obj - previous_obj)/(f_max - f_min);
            end
         end
    end
    D_front = [];
    D_front(:,1) = zeros(length(Fronts(front).indices),1);
    for i = 1 : N_obj
        D_front(:,1) = D_front(:,1) + front_pool(:,N_obj + n + 1 + i);
    end
    front_pool(:,N_obj + n + 2) = D_front;
    front_pool = front_pool(:,1 : N_obj + n + 2);
    sorted_pool(previous_index:current_index,:) = front_pool;
end
ranked_pool = sorted_pool();
