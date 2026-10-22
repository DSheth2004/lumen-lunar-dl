import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect  
from scipy import interpolate
import pysptools.spectro as spectro

class HapkeRTM:
    def __init__(self, emergence_angle, incidence_angle, phase_angle, wavelengths, reflectances):
        """
        Initializes the Hapke Radiative Transfer Model.

        Parameters:
        - emergence_angle (float): Emergence angle in degrees.
        - incidence_angle (float): Incidence angle in degrees.
        - phase_angle (float): Phase angle in degrees.
        - wavelengths (list): List of wavelengths.
        - reflectances (list): List of reflectance values.
        """
        self.mu_0 = np.cos(np.radians(incidence_angle))
        self.mu = np.cos(np.radians(emergence_angle))
        self.g = phase_angle
        self.w = np.array(wavelengths)
        self.r = np.array(reflectances)

    def _Chandrasekhar(self, ssa, x):
        """
        Computes Chandrasekhar's H function from the Hapke model.
        Input: 1) ssa = SSA, 2) x = mu or mu_0
        Output: 1) H = corresponding Chandrasekar H-function
        """
        gamma = np.sqrt(1 - ssa)
        r0 = (1 - gamma) / (1 + gamma)
        H = (1 - ssa * x * (r0 + (1 - 2 * r0 * x)/2 * np.log((1 + x) / x)))**(-1)
        return H

    def _phaseFunction(self, g):
        """
        Computes phase function in the Hapke model.
        Input: 1) g = phase angle
        Output: 1) P(g) = phase function
        """
        b = -0.4
        c = 0.25
        g_rad = np.radians(self.g)
        return 1 + b * np.cos(g_rad) + c * (1.5 * np.cos(g_rad)**2 - 0.5)

    def _backscatterFunction(self):
        """
        Computes backscattering function in the Hapke model.
        Input: 1) g = phase angle
        Output: 1) B(g) = backscattering function
        """
        h = -3 / 8 * np.log(1 - 0.41)
        return 1 / (1 + (1 / h)) * np.tan(np.radians(self.g / 2))

    def _ssa2ref(self, ssa):
        """
        Converts SSA to reflectance using instance angles.
        Input: 1) ssa = SSA, 2) mu or mu_0 = cosines of incidence and emergence angles
        Output: 1) r = corresponding reflectance 
        """
        use_backscatter = False

        H_mu0 = self._Chandrasekhar(ssa, self.mu_0) 
        H_mu = self._Chandrasekhar(ssa, self.mu)

        if use_backscatter:
            B = self._backscatterFunction()
            P = self._phase_function(self.phase_angle)
        else: # no backscattering (B=0) and isotropic scattering (P=1)
            B = 0
            P = 1  

        reflectance = ssa / 4 / (self.mu_0 + self.mu) * ((1 + B) * P + H_mu0 * H_mu - 1)
        
        return reflectance

    def _S_e(self, n, k):
        """
        Computes surface reflection coefficient.
        Input: 1) n = real refractive index, 2) k = imaginary refractive index
        Output: 1) S_e = surface reflection coefficient
        """
        return ((n - 1)**2 + k**2) / ((n + 1)**2 + k**2) + 0.05

    def _S_i(self, n):
        """
        Computes internally scattered light component.
        Input: 1) n = real refractive index
        Output: 1) S_i = internally scattered light
        """
        return 1.014 - 4 / (n * (n + 1)**2)

    def _theta(self, ri, a, s, d_mean):
        """
        Computes reflection coefficient for externally incident light.
        Input: 1) ri = internal diffusive reflectance, 2) a = internal absorption coefficient, 3) s = scattering coefficient, 4) d_mean = mean free path length
        Output: 1) theta = reflection coefficient for externally incident light
        """
        exp_term = np.exp(-np.sqrt(a * (a + s) * d_mean))
        return (ri + exp_term) / (1 + ri * exp_term)

    def _ssa(self, se, si, th):
        """
        Computes single scattering albedo.
        Input: 1) se = the surface reflection coefficient, 2) si = the internally scattered light, 3) th = reflection coefficient 
        Output: 1) ssa = single scattering albedo
        """
        return se + (1 - se) * (1 - si) / (1 - si * th) * th

    def _r_i(self, alpha, s):
        """
        Computes internal diffusive reflectance.
        Input: 1) alpha = internal absorption coefficient, 2) s = scattering coefficient
        Output: 1) r_i = internal diffusive reflectance
        """
        return (1 - np.sqrt(alpha / (alpha + s))) / (1 + np.sqrt(alpha / (alpha + s)))

    def _alpha(self, k, n, wi):
        """
        Computes internal absorption coefficient.
        Input: 1) k = imaginary refractive index, 2) n = real refractive index, 3) wi = wavelength
        Output: 1) alpha = internal absorption coefficient
        """
        a = 4 * np.pi * k / wi
        return a

    def _D_mean(self, n, D):
        """
        Computes mean free path length of transmitted rays.
        Input: 1) n = real refractive index, 2) D = grain size
        Output: 1) d_mean = mean free path length
        """
        d_mean = 2 / 3 * (n**2 - 1 / n * (n**2 - 1)**(3 / 2)) * D
        return d_mean

    def _nk2ssa(self, k, wi, s, n, D):
        """
        Computes SSA from optical constants.
        Input: 1) k = imaginary refractive index, 2) wi = wavelength, 3) s = scattering coefficient, 4) n = real refractive index, 5) D = grain size
        Output: 1) ssa = single scattering albedo
        """
        d_mean = self._D_mean(n, D)
        a = self._alpha(k, n, wi)
        ri = self._r_i(a, s)
        th = self._theta(ri, a, s, d_mean)
        si = self._S_i(n)
        se = self._S_e(n, k)
        return self._ssa(se, si, th)

    def ref2ssa(self):
        """
        *** Appendix C. Derivation of single scattering albedo using the Hapke radiative transfer model.
        
        Inputs:
        - self.w (numpy.ndarray): Wavelengths.
        - self.r (numpy.ndarray): Reflectance values.
        - self.mu_0 (float): Cosine of incidence angle.
        - self.mu (float): Cosine of emergence angle.
        - self.g (float): Phase angle.

        Outputs:
        - (numpy.ndarray, list): Wavelengths and computed SSA values.
        """

        ssa_values = []

        for ref in self.r:  # Iterate over individual reflectance values
            try:
                # Solve for SSA using bisection method
                ssa_value = bisect(lambda ssa: self._ssa2ref(ssa) - ref, 0.1, 1, xtol=1e-6)
            except ValueError:
                ssa_value = np.nan  # Store NaN if no solution found
            
            ssa_values.append(ssa_value)

        return self.w, ssa_values


    def k(self, tol, st, ed, n, D):
        """
        *** Appendix D. Calculation of mineral endmember's imaginary refractive index (k) from reflectance data using bisection method.

        Inputs:
        - self.w (numpy.ndarray): Wavelengths.
        - self.r (numpy.ndarray): Reflectance values.
        - self.mu_0 (float): Cosine of incidence angle.
        - self.mu (float): Cosine of emergence angle.
        - self.g (float): Phase angle.

        - tol (float): Tolerance for numerical convergence.
        - st, ed (int): Start and end indices for computation.
        - n (float): Real refractive index.
        - D (float): Grain size.

        Outputs:
        - (list, list): Wavelength and k values.
        """
        k_spec = []

        for i in range(st, ed):
            try:

                k_value = bisect(
                    lambda k: self._ssa2ref(self._nk2ssa(k, self.w[i], 0, n, D)) - self.r[i], 
                    1e-7, 1, xtol=tol
                )

            except ValueError:
                k_value = np.nan  # If no solution found, store NaN
            k_spec.append(k_value)

        return self.w[st:ed], k_spec

class HapkeCorrection:
    def __init__(self, wavelengths, ssa_mix, ssa_sw):
        """
        Initializes the Hapke Correction model.

        Parameters:
        - wavelengths (list): List of wavelengths.
        - ssa_mix (list): List of SSA values for the mixture.
        - ssa_sw (list): List of SSA values for the space weathered sample.
        """
        self.w = wavelengths
        self.ssa_mix = ssa_mix
        self.ssa_sw = ssa_sw
        self.d_dc = 0.6422 # averaged from Apollo mare basalt samples

    def _d(self, ssa):
        """
        Computes the depth of the 970 nm band.
        Input: 1) ssa = single scattering albedo
        Output: 1) d = depth of the 970 nm band after continuum removal
        """
        schq = spectro.convex_hull_removal(ssa, self.w)

        # find the index of 970 nm or close to 970 nm (band depth)
        index_970 = (np.abs(np.array(self.w) - 0.970)).argmin()

        return schq[0][index_970]

    def _delta(self, d_mix, d_sw):
        """
        Computes the depth ratio delta of the 970 nm band.
        Input: 1) d_mix = depth of the 970 nm band of the mixture, 2) d_sw = depth of the 970 nm band of the space weathered sample
        Output: 1) delta 
        """
        d_dc = self.d_dc
        d_mix = self._d(self.ssa_mix)
        d_sw = self._d(self.ssa_sw)
        delta = (d_mix - d_dc) / (d_sw - d_dc)
    
        return delta
    
    def _theta_sw(self):
        """
        Computes the space weathering degree theta, based on ism vs delta
        Input: 1) delta = depth ratio of the 970 nm band
        Output: 1) theta_sw = space weathering degree
        """
        delta = self._delta(self.ssa_mix, self.ssa_sw)
        ism = 65.68 * delta**(3.731)
        theta_sw = ism/100
        return theta_sw

    def ssa_dc(self):
        """
        Computes the depth-corrected SSA.
        Input: 1) ssa_mix = SSA of the mixture, 2) ssa_sw = SSA of the space weathered sample
        Output: 1) ssa_dc = depth-corrected SSA
        """
        theta_sw = self._theta_sw()
        ssa_dc = (np.array(self.ssa_mix) - np.array(self.ssa_sw) * theta_sw) / (1 - theta_sw)
        return self.w, ssa_dc



