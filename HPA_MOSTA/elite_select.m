function elite_pool  = elite_select(ranked_pool, N_obj, n,N_set)
% Non-dominated elite selection.
% Inputs: ranked_pool - [Z, ObjValues, FrontNo, D]; N_obj - number of objectives;
%         n - number of harmonic sampling points; N_set - target solution set size.
% Output: elite_pool - elite solution set [Z, ObjValues, FrontNo, D].

[N_pool, N_columns] = size(ranked_pool);
[unused_fronts,front_order] = sort(ranked_pool(:,N_obj + n + 1));

clear unused_fronts N_columns
for i = 1 : N_pool
    sorted_pool(i,:) = ranked_pool(front_order(i),:);
end

max_rank = max(ranked_pool(:,N_obj + n + 1));

previous_index = 0;
for i = 1 : max_rank
    current_index = max(find(sorted_pool(:,N_obj + n + 1) == i));
    if current_index > N_set
        remaining = N_set - previous_index;
        front_pool = ...
            sorted_pool(previous_index + 1 : current_index, :);
        [unused_distances,crowding_order] = ...
            sort(front_pool(:, N_obj + n + 2),'descend');
        for j = 1 : remaining
            elite_pool(previous_index + j,:) = front_pool(crowding_order(j),:);
        end
        return;
    elseif current_index < N_set
        elite_pool(previous_index + 1 : current_index, :) = ...
            sorted_pool(previous_index + 1 : current_index, :);
    else
        elite_pool(previous_index + 1 : current_index, :) = ...
            sorted_pool(previous_index + 1 : current_index, :);
        return;
    end
    previous_index = current_index;
end
