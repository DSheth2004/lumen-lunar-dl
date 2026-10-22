function D = load_data

%-------------------------------------------------------------------------
% INPUT PARAMETERS
%-------------------------------------------------------------------------

% MINERAL ENDMEMBERS
%  *** Density vector
rho = [3.32 2.69 3.40 3.38 4.72 3.3 3.0 3.3 3.0]'; %jung

% *** Load optical constants for each laboratory endmember; 
% columns matrices (wavelengths in meters, n, and k)
P{1} = dlmread('data-jung/olivine.txt');
P{2} = dlmread('data-jung/plagioclase.txt');
P{3} = dlmread('data-jung/pigeonite.txt');
P{4} = dlmread('data-jung/augite.txt');
P{5} = dlmread('data-jung/ilmenite.txt');
P{6} = dlmread('data-jung/pyroxeneglass.txt');
P{7} = dlmread('data-jung/plagioclaseglass.txt');
P{8} = dlmread('data-jung/orangeglass.txt');
P{9} = dlmread('data-jung/agglutinate.txt');


N = length(P);
% Sort them into n (real) and k (imaginary) refractive indices
for i =1:N
    O =P{i};
    l=O(:,1);
    nn=O(:,2);
    kk=O(:,3);
    lambda{i}=l;
    n{i}=nn;
    k{i}=kk;
    clear O l nn kk
end

%-------------------------------------------------------------------------
% MCMC PARAMETERS
%-------------------------------------------------------------------------

% MCMC Parameters
% *** Length of each Markov chain;
Nm = 2000;
% *** Number of samples per cooling step
Nsteps = 10;

% Grain size prior : uniform distribution
% *** Low bounds size in microns
prior_size_low =[0 0 0 0 0 0 0 0 0];
% *** High bounds size in microns
prior_size_high = [990 990 990 990 990 990 990 990 990]; 


% Mineral prior: Dirichlet distribution (BY SARAH 5/1/2023)
Ncomponents = length(rho); 
component_alpha0 = 30; 
% ABUNDANCE PRIOR ADDED BY SARAH 5/1/2023
component_means = [0.07 0.39 0.16 0.16 0.10 0.115 0.005/3 0.005/3 0.005/3];
if sum(component_means) ~= 1; error('Component means should sum to one.'); end
if length(component_means) ~= Ncomponents; error('Somehow you do not have the same number of abundances as grain sizes'); end
component_alpha=component_means*component_alpha0; 

% SPECTRAL DATA
R1_vs_SSA0 = 1; % *** If reflectance data, set to 1; if SSA, set to 0;
if R1_vs_SSA0
    inc = 30; % *** incident angle for data reflectance spectrum
    e = 0; % *** emission angle for data reflectance spectrum
else
    inc = NaN;
    e = NaN;
end
% *** Data (two columns matrix with wavelength and reflectance/SSA)
S = dlmread('data-jung/c1lr152.txt'); 
lam_SPEC = S(:,1);                   
R_SPEC = S(:,2);  


if min(lam_SPEC)>1
    lam_SPEC = lam_SPEC.*1e-9;      
end

% FIT PARAMETERS
Cov = 2 * 1e-4; % crism

% *** Covariance matrix = (Cd + Cp); increase to allow for more variance
min_lam = 0.55.*1e-6; % Minimum wavelength of range over which spectral fit is performed, in meters
max_lam = 2.50.*1e-6; % Maximum wavelength of range over which spectral fit is performed, in meters

% Note: Data and endmember spectra needed over this wavelength range
% Creating Data Structure
D.inc = inc;
D.e = e;
D.rho = rho';

D.lambda = lambda;
D.n = n;
D.k = k;
D.lam_SPEC = lam_SPEC;
D.R_SPEC = R_SPEC;
D.rho = rho'; 
D.Cinv = 1/Cov; 
D.Ncomponents = Ncomponents; 
D.prior_size_low = prior_size_low; 
D.prior_size_high = prior_size_high; 
D.prior_alpha = component_alpha; 
D.Nm = Nm;
D.Nsteps = Nsteps;
D.R_vs_SSA = R1_vs_SSA0;
D.lmin = min_lam;
D.lmax = max_lam;