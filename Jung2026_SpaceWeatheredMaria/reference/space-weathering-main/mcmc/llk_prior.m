function llk = llk_prior(theta,D)

% theta is a vector that has N grain sizes followed by N abundances
% D is a structure containing the data D.rho, D.n, D.k, D_lam_SPEC,
% D.R_SPEC

% MODEL ABUNDANCES AND GRAIN SIZES
N = length(theta)/2;                   % # of endmembers
sizes = theta(1:N);                    % grain sizes
abundances = theta(N+1:2*N);           % abundances

llk=0;
for i=1:N % Grain sizes
    llk=llk+log(unifpdf(theta(i),D.prior_size_low(i),D.prior_size_high(i)));
end

% Updated May 1, 2023 by Sarah to also have a prior on abundances:
% Abundances
dirichlet_prior_fun=@(x,a)sum((a(:)-1).*log(x(:)));
llk = llk + dirichlet_prior_fun(abundances,D.prior_alpha);

% Note: per https://en.wikipedia.org/wiki/Dirichlet_distribution
% The Dirichlet PDF function is 1/B(ALPHA) * prod(x.^(prior_alpha-1) where
% B(ALPHA) is a normalization constant that we can ignore for this purpose.
% Then ln(Dirichlet PDF) ~ log( prod(x.^(prior_alpha-1) ) =
% sum( log(x.^(prior_alpha-1)) ) = sum( (prior_alpha-1).*log(x) )


return
