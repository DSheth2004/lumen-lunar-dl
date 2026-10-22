% https://cxwangyi.wordpress.com/2009/03/18/to-generate-random-numbers-from-a-dirichlet-distribution/
% a is alpha vector. n is number of samples.
function r = drchrnd(a, n)
%--------------------------------------------------------------------------
% TAKE A SAMPLE FROM A DIRICHLET DISTRIBUTION (
%   1) a = alpha vector
%   2) n = number of samples
% Outputs:
%   1) r = corresponding reflectance
%--------------------------------------------------------------------------
% take a sample from a dirichlet distribution
p = length(a);
r = gamrnd(repmat(a, n, 1), 1, n, p);
r = (r ./ repmat(sum(r, 2), 1, p))';
end