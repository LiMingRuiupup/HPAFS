function ObjValues = evaluate_objectives(objective_fun,Z,X_tr,y_tr)
% Objective evaluation of a solution set.
% Inputs: objective_fun - objective function handle; Z - binary solution set;
%         X_tr, y_tr - training samples and concentration labels.
% Output: ObjValues - objective values in the order [rho, E].

Data{1}=X_tr;
Data{2}=y_tr;

N_solutions = size(Z,1); 
for i = 1:N_solutions
Data{3}=Z(i,:);
ObjValues(i,:) = feval(objective_fun, Data);
end 
end