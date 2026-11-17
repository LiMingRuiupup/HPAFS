function hv=hypervolume(ObjValues,ref_point,N_mc)
% Normalized hypervolume (HV) estimation.
% Inputs: ObjValues - objective values; ref_point - reference point in the same objective order;
%         N_mc - optional sample count or test-point matrix.
% Output: hv - estimated normalized HV.

error(nargchk(2,3,nargin));
error(nargoutchk(0,1,nargout));

ObjValues=ObjValues*diag(1./ref_point);
[N_points,N_obj]=size(ObjValues);
if nargin<3
    N_mc=1000;
end
if ~isscalar(N_mc)
    sample_points=N_mc;
    N_mc=size(sample_points,1);
else
    sample_points=rand(N_mc,N_obj);
end

is_dominated=false(N_mc,1);
lower_bound=min(ObjValues);
needs_check=all(bsxfun(@gt, sample_points, lower_bound),2);

for k=1:N_points
    if any(needs_check)
        dominated_now=all(bsxfun(@gt, sample_points(needs_check,:), ObjValues(k,:)),2);
        is_dominated(needs_check)=dominated_now;
        needs_check(needs_check)=~dominated_now;
    end
end

hv=sum(is_dominated)/N_mc;
