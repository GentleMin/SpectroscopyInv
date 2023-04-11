import numpy as np
import time
from scipy import special as specfun
from . import utils


def ellipp(n, m):
    if np.any(m >= 1):
        raise ValueError('m must be < 1')
    y = 1 - m
    rf = specfun.elliprf(0, y, 1)
    rj = specfun.elliprj(0, y, 1, 1 - n)
    return rf + rj * n / 3

def ellippinc(phi, n, m):
    nc = np.floor(phi / np.pi + 0.5)
    phi -= nc * np.pi
    sin_phi  = np.sin(phi)
    sin2_phi = sin_phi  * sin_phi;
    sin3_phi = sin2_phi * sin_phi;
    x = 1 - sin2_phi;
    y = 1 - m * sin2_phi;
    rf = specfun.elliprf(x, y, 1)
    rj = specfun.elliprj(x, y, 1, 1 - n * sin2_phi)
    val = sin_phi * rf + sin3_phi * rj * n / 3
    # if nc != 0:
    #     rp = ellipp(n, m)
    #     val += 2 * nc * rp
    return val


class AbsorbanceModel:
    
    def __init__(self, c, c_type="coeffs") -> None:
        self.set_absorbance(c, c_type=c_type)
        
    def set_absorbance(self, c, c_type="coeffs"):
        if c_type == "coeffs":
            assert c[0] >= c[1] and c[1] >= c[2]
            self.coeffs = c
            self.absorb = utils.coeffs_to_absorbance(c)
        elif c_type == "absorbance":
            assert c[0] <= c[1] and c[1] <= c[2]
            self.coeffs = utils.absorbance_to_coeffs(c)
            self.absorb = c
        else:
            raise KeyError("Invalid input type!")
        
    def eval_absorbance(self, x, y, z, x_type="Cartesian"):
        if x_type == "spherical":
            x, y, z = utils.spherical_to_cartesian(x, y, z)
        r = np.sqrt(x**2 + y**2 + z**2)
        A = (self.coeffs[0]*x**2 + self.coeffs[1]*y**2 + self.coeffs[2]*z**2)/r
        return A
    
    def total_absorbance(self):
        return np.sum(self.absorb)
    
    def moments(self):
        E_A = np.mean(self.coeffs)
        sigma_A = np.sqrt(2*((self.coeffs[0] - self.coeffs[1])**2 + (self.coeffs[1] - self.coeffs[2])**2 + (self.coeffs[2] - self.coeffs[0])**2)/45)
        return E_A, sigma_A
    
    def pdf(self, A_obs):
        A_obs = np.atleast_1d(A_obs)
        density = np.zeros(A_obs.shape)
        
        idx = np.isclose(A_obs, self.coeffs[1])
        density[idx] = np.nan
        
        idx = np.logical_and(A_obs >= self.coeffs[2], A_obs < self.coeffs[1])
        prefactor = 1/np.pi/np.sqrt((self.coeffs[1] - self.coeffs[2])*(self.coeffs[0] - A_obs[idx]))
        ellip_arg = (self.coeffs[0] - self.coeffs[1])*(A_obs[idx] - self.coeffs[2])/(self.coeffs[1] - self.coeffs[2])/(self.coeffs[0] - A_obs[idx])
        density[idx] = prefactor*specfun.ellipk(ellip_arg)
        
        idx = np.logical_and(A_obs > self.coeffs[1], A_obs < self.coeffs[0])
        prefactor = 1/np.pi/np.sqrt((self.coeffs[0] - self.coeffs[1])*(A_obs[idx] - self.coeffs[2]))
        ellip_arg = (self.coeffs[1] - self.coeffs[2])*(self.coeffs[0] - A_obs[idx])/(self.coeffs[0] - self.coeffs[1])/(A_obs[idx] - self.coeffs[2])
        density[idx] = prefactor*specfun.ellipk(ellip_arg)
        
        return density
    
    def cdf(self, A_obs):
        A_obs = np.atleast_1d(A_obs)
        cumulative = np.zeros(A_obs.shape)
        
        idx = np.isclose(A_obs, self.coeffs[1])
        if np.any(idx):
            cumulative[idx] = 2/np.pi*np.arctan(np.sqrt((self.coeffs[1] - self.coeffs[2])/(self.coeffs[0] - self.coeffs[1])))
        
        idx = (A_obs > self.coeffs[2]) & (A_obs < self.coeffs[1])
        if np.any(idx):
            prefactor = 2/np.pi*(self.coeffs[0] - A_obs[idx])/np.sqrt((self.coeffs[0] - self.coeffs[2])*(self.coeffs[1] - A_obs[idx]))
            ellip_arg_n = -(self.coeffs[0] - self.coeffs[1])/(self.coeffs[1] - A_obs[idx])
            ellip_arg_m = -(self.coeffs[0] - self.coeffs[1])*(A_obs[idx] - self.coeffs[2]) / \
                (self.coeffs[0] - self.coeffs[2])/(self.coeffs[1] - A_obs[idx])
            cumulative[idx] = 1 - prefactor*ellipp(ellip_arg_n, ellip_arg_m)
        
        idx = (A_obs > self.coeffs[1]) & (A_obs < self.coeffs[0])
        if np.any(idx):
            prefactor = 2/np.pi*(A_obs[idx] - self.coeffs[2])/np.sqrt((self.coeffs[0] - self.coeffs[2])*(A_obs[idx] - self.coeffs[1]))
            ellip_arg_n = -(self.coeffs[1] - self.coeffs[2])/(A_obs[idx] - self.coeffs[1])
            ellip_arg_m = -(self.coeffs[1] - self.coeffs[2])*(self.coeffs[0] - A_obs[idx]) / \
                (self.coeffs[0] - self.coeffs[2])/(A_obs[idx] - self.coeffs[1])
            cumulative[idx] = prefactor*ellipp(ellip_arg_n, ellip_arg_m)
        
        idx = A_obs >= self.coeffs[0]
        cumulative[idx] = 1.
        
        return cumulative
    
    def pdf_smooth(self, A_obs, delta):
        return (self.cdf(A_obs + delta) - self.cdf(A_obs - delta)) / (2*delta)
    
    def contour_absorbance(self, A, N_phi=100):
        assert A >= self.coeffs[2] and A <= self.coeffs[0]
        phi_array = np.linspace(0, 2*np.pi, num=N_phi)
        if A < self.coeffs[1]:
            cos_squared = (self.coeffs[0]*np.cos(phi_array)**2 + self.coeffs[1]*np.sin(phi_array)**2 - A) / \
                (self.coeffs[0]*np.cos(phi_array)**2 + self.coeffs[1]*np.sin(phi_array)**2 - self.coeffs[2])
            theta_array = np.arccos(np.sqrt(cos_squared))
            x1 = np.sin(theta_array)*np.cos(phi_array)
            x2 = np.sin(theta_array)*np.sin(phi_array)
            x3 = np.cos(theta_array)
            closed_dir = 0
        elif A > self.coeffs[1]:
            cos_squared = (self.coeffs[1]*np.cos(phi_array)**2 + self.coeffs[2]*np.sin(phi_array)**2 - A) / \
                (self.coeffs[1]*np.cos(phi_array)**2 + self.coeffs[2]*np.sin(phi_array)**2 - self.coeffs[0])
            theta_array = np.arccos(np.sqrt(cos_squared))
            x1 = np.cos(theta_array)
            x2 = np.sin(theta_array)*np.cos(phi_array)
            x3 = np.sin(theta_array)*np.sin(phi_array)
            closed_dir = 2
        else:
            cos_squared = (self.coeffs[0] - self.coeffs[1])*np.cos(phi_array)**2 / \
                (self.coeffs[0]*np.cos(phi_array)**2 + self.coeffs[1]*np.sin(phi_array)**2 - self.coeffs[2])
            theta_array = np.arccos(np.sqrt(cos_squared))
            x1 = np.sin(theta_array)*np.cos(phi_array)
            x2 = np.sin(theta_array)*np.sin(phi_array)
            x3 = np.cos(theta_array)
            closed_dir = 0
        return x1, x2, x3, closed_dir
    
    def log_likelihood(self, obs_data, delta):
        obs_data = np.atleast_1d(obs_data)
        lk = self.pdf_smooth(obs_data, delta)
        if np.any(np.isclose(lk, 0.)):
            return np.finfo(np.float64).min
        return np.sum(np.log(lk))
    


def abs_ParamSearchSize(x, y, z):
    """Calculate number of feasible combinations of principal absorbances in grid"""

    xv, yv, zv = np.meshgrid(x, y, z)
    bl = (xv - yv <= zv) & (zv <= yv) & (yv <= xv) # locations of grid points that satisfy constraints
    return len(xv[bl])

def grid_llk(data, delta, n):
    """Perform grid search for principal absorbances over a 3-D rectangle"""
    
    # set up Cartesian search grid for parameters (a1, a2, a3)
    a2_start = 0.9*np.min(data)  # set limits of a2
    a2_stop = 1.1*np.max(data)   # set limits of a2
    a3_start = 0.0               # set lower limit of a3 using eqn. (41) Jackson et al. (2018)
    a3_stop = a2_stop            # set upper limit of a3 using eqn. (40) Jackson et al. (2018)
    a1_start = a2_start          # set lower limit of a1 using eqn. (41) Jackson et al. (2018)
    a1_stop = 2.0*a2_stop        # set upper limit of a1 using eqn. (40) Jackson et al. (2018)
    x = np.linspace(a1_start, a1_stop, n[0]) # a1 discretized values
    y = np.linspace(a2_start, a2_stop, n[1]) # a2 discretized values
    z = np.linspace(a3_start, a3_stop, n[2]) # a3 discretized values

    print('\nPerforming grid search on grid', n[0], 'x', n[1], 'x', n[2],
          '\nDiscretization intervals delta a1 =', (a1_stop-a1_start)/n[0],
          'delta a2 =', (a2_stop-a2_start)/n[1],
          'delta a3 =', (a3_stop-a3_start)/n[2],
          '\nwith', abs_ParamSearchSize(x, y, z), 'feasible points...\n')
    
    abs_mod = AbsorbanceModel(np.array([6, 5, 4]))
    start_time = time.time()
    llk = np.zeros((x.size, y.size, z.size))
    for ix, xpt in enumerate(x):
        for iy, ypt in enumerate(y):
            for iz, zpt in enumerate(z):
                if (xpt - ypt <= zpt) and (zpt <= ypt) and (ypt <= xpt):
                    abs_mod.set_absorbance(np.array([xpt, ypt, zpt]))
                    llk[ix, iy, iz] = abs_mod.log_likelihood(data, delta)
                else:
                    llk[ix, iy, iz] = np.finfo(np.float64).min
    print("--- {:g} seconds compute time ---".format(time.time() - start_time))
    return x, y, z, llk
    
