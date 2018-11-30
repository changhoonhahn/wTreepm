'''
Class to store cosmology functions.

Author[s]: Andrew Wetzel.
'''

# system -----
from __future__ import division
import numpy as np
from numpy import log10
from scipy import integrate
# local -----
from . import constants as const
from . import utility as ut


class CosmologyClass(dict):
    '''
    Store & derive cosmological parameters.
    '''
    def __init__(self, hubble=0.7, omega_m=0.27, omega_l=0.73, omega_b=0.047, sigma_8=0.82,
                 n_s=0.95, w=-1.0):
        '''
        Import dimensionless hubble constant, Omega_matter_0, Omega_lambda_0, Omega_baryon_0,
        sigma_8_0, scalar index of primordial power spectrum, dark energy equation of state.
        Default is Bolshoi run.
        Planck 2013:
            hubble=0.68, omega_m=0.31, omega_l=0.69, omega_b=0.048, sigma_8=0.83, n_s=0.96, w=-1.0
        '''
        self['hubble'] = hubble
        self['omega_m'] = omega_m
        self['omega_l'] = omega_l
        self['omega_b'] = omega_b
        self['omega_k'] = 1 - self['omega_m'] - self['omega_l']
        self['sigma_8'] = sigma_8
        self['n_s'] = n_s
        self['w'] = w
        self.DistanceRedshift = None
        self.AgeRedshift = None

    # parameters ----------
    def omega_matter(self, redshifts):
        '''
        Get Omega_matter at redshift[s].

        Import redshift[s].
        '''
        return self['omega_m'] / (self['omega_m'] + self['omega_l'] / (1 + redshifts) ** 3 +
                                  self['omega_k'] / (1 + redshifts) ** 3)

    def omega_lambda(self, redshifts):
        '''
        Get Omega_lambda at redshift[s].

        Import redshift[s].
        '''
        return self['omega_l'] / (self['omega_m'] * (1 + redshifts) ** 3 + self['omega_l'] +
                                  self['omega_k'] * (1 + redshifts) ** 2)

    def density_critical(self, redshifts, assume_hubble=True):
        '''
        Get critical density {M_sun[/h] / Mpc[/h] ^ 3 comoving} at redshift[s].

        Import redshift[s], whether to use assigned hubble constant value.
        '''
        den = const.density_critical_0 * (self['omega_l'] / (1 + redshifts) ** 3 + self['omega_m'] +
                                          self['omega_k'] / (1 + redshifts))
        if assume_hubble:
            den *= self['hubble'] ** 2
        return den

    def density_matter(self, redshifts, assume_hubble=True):
        '''
        Get average matter density {M_sun[/h] / Mpc[/h] ^ 3 comoving} at redshift[s].

        Import redshift[s], whether to use assigned hubble constant value.
        '''
        return self.omega_matter(redshifts) * self.density_critical(redshifts, assume_hubble)

    def hubble_parameter(self, redshifts):
        '''
        Get Hubble parameter {1 / sec} at redshift[s].

        Import redshift[s].
        '''
        return (const.hubble_parameter_0 * self['hubble'] *
                (self['omega_m'] * (1 + redshifts) ** 3 + self['omega_l'] +
                 self['omega_k'] * (1 + redshifts) ** 2) ** 0.5)

    # density ----------
    def growth_factor(self, redshift):
        '''
        Get growth factor by which density perturbations were smaller at z,
        normalized so g(z = 0) = 1.

        Import redshift.
        '''
        def get_dgrowth(aexp, omega_m, omega_l, omega_k):
            return (omega_m / aexp + omega_l * aexp ** 2 + omega_k) ** -1.5
        aexp = 1 / (1 + redshift)
        aexp_min = 1e-3
        aexp_max = 1.0
        omega_m = self.omega_matter(redshift)
        omega_l = self.omega_lambda(redshift)
        omega_k = 1 - omega_m - omega_l
        g_0 = self['omega_m'] * integrate.quad(get_dgrowth, aexp_min, aexp_max,
                                               (self['omega_m'], self['omega_l'],
                                                self['omega_k']))[0]
        g_aexp = omega_m * integrate.quad(get_dgrowth, aexp_min, aexp_max,
                                          (omega_m, omega_l, omega_k))[0]
        return g_aexp / g_0 * aexp

    def transfer_function(self, k, source='e&h'):
        '''
        Get transfer function at input wave-number k.

        Import wave-number, source (e&h, e&h-paper, ebw).
        '''
        if 'eh' in source:
            # Eisenstein & Hu 1999
            # CMB temperature conversion from 2.7 K
            theta = 2.728 / 2.7
            # comoving distance that sound wave can propagate
            s = (44.5 * np.log(9.83 / (self['omega_m'] * self['hubble'] ** 2)) /
                 (1 + 10 * (self['omega_b'] * self['hubble'] ** 2) ** 0.75) ** 0.5 * self['hubble'])
            alpha = (1 - 0.328 * np.log(431 * (self['omega_m'] * self['hubble'] ** 2)) *
                     self['omega_b'] / self['omega_m'] +
                     0.380 * np.log(22.3 * (self['omega_m'] * self['hubble'] ** 2)) *
                     (self['omega_b'] / self['omega_m']) ** 2)
            gamma = (self['omega_m'] * self['hubble'] ** 2) * (alpha + (1 - alpha) /
                                                               (1 + (0.43 * k * s) ** 4))
            # convert q to h/Mpc
            q = k * theta ** 2 / gamma * self['hubble']
            if source == 'e&h':
                # modified version
                L = np.log(2 * np.e + 1.8 * q)
                C = 14.2 + 731 / (1 + 62.5 * q)
            elif source == 'e&h-paper':
                # original paper version
                beta = 1 / (1 - 0.949 * self.omega_b0 / self['omega_m'])
                L = np.log(np.e + 1.84 * beta * alpha * q)
                C = 14.4 + 325 / (1 + 60.5 * q ** 1.11)
            return L / (L + C * q ** 2)
        elif source == 'ebw':
            # Efstathiou, Bond & White 1992
            shape = self['omega_m'] * self['hubble']
            a = 6.4 / shape
            b = 3.0 / shape
            c = 1.7 / shape
            nu = 1.13
            return (1 + (a * k + (b * k) ** (3 / 2) + (c * k) ** 2) ** nu) ** (-1 / nu)
        else:
            raise ValueError('not recognize transfer function source = %s' % source)

    def delta_2(self, k, source='e&h'):
        '''
        Get *non-normalized* Delta ^ 2(k).
        Need to scale to sigma_8 at z = 0, then need to scale by growth function at z.

        Import wave number, transfer function source.
        '''
        return ((const.hubble_distance * k) ** (3 + self['n_s']) *
                self.transfer_function(k, source) ** 2)

    # time, distance ----------
    def make_age_v_redshift_spline(self, z_lim=[0, 25], z_num=250):
        '''
        Make & store spline to get ages from redshifts.

        Import redshift range & number of bins.
        '''
        self.AgeRedshift = ut.math.SplineFunctionClass(self.age, z_lim, z_num)

    def age(self, redshifts):
        '''
        Get age[s] of the Universe {Gyr} at redshift[s].

        Import redshift[s].
        '''
        def get_dt(a, self):
            return (self['omega_m'] / a + self['omega_l'] * a ** 2 + self['omega_k']) ** -0.5
        aexps = 1 / (1 + redshifts)
        if np.isscalar(redshifts):
            ages = const.hubble_time / self['hubble'] * integrate.quad(get_dt, 0, aexps, (self))[0]
        else:
            if self.AgeRedshift is None:
                self.make_age_v_redshift_spline()
            ages = self.AgeRedshift.val(redshifts)
        return ages

    def print_times(self, kind='a', lim=[0.1, 0.99], num=60, spacing='lin'):
        '''
        Print scale factors, redshifts, times, delta times for each snapshot.

        Import time kind (a, z), range, number of snapshots, spacing (lin, log).
        '''
        if spacing == 'lin':
            vals = np.linspace(min(lim), max(lim), num)
        elif spacing == 'log':
            vals = np.logspace(log10(min(lim)), log10(max(lim)), num)
        if kind == 'a':
            aexps = vals
            redshifts = 1 / aexps - 1
            ts = [self.age(z) for z in redshifts]
        elif kind == 'z':
            redshifts = vals[::-1]
            aexps = 1 / (1 + redshifts)
            ts = [self.age(z) for z in redshifts]
        else:
            raise ValueError('not recognize time kind = %s' % kind)
        print '# %d snapshots, %s spacing in %s' % (num, spacing, kind)
        print '# a:',
        for aexp in aexps:
            print '%.4f' % aexp,
        print '\n# z:',
        for z in redshifts:
            print '%.4f' % z,
        print '\n# t:',
        for t in ts:
            print '%.4f' % t,
        print '\n# dt:',
        for ti in xrange(len(ts) - 1):
            print '%.3f' % (ts[ti + 1] - ts[ti]),
        print '\n'

    def make_distance_v_redshift_spline(self, z_lim=[0, 0.2], z_num=100, assume_hubble=False):
        '''
        Make & store spline to get distances from redshifts.

        Import redshift range & number of bins, whether to include assumed Hubble constant.
        '''
        self.DistanceRedshift = ut.math.SplineFunctionClass(self.distance, z_lim, z_num,
                                                            assume_hubble=assume_hubble)

    def distance(self, redshifts, compute_kind='integrate', redshift_lim=[0, 0.2],
                 assume_hubble=False):
        '''
        Get radial distance {Mpc[/h] comoving} from z = 0 to redshift[s].

        Import redshift[s], compute kind (integrate, spline), redshift range (if spline),
        whether to include Hubble constant.
        '''
        def get_ddist(redshifts, self):
            return (self['omega_m'] * (1 + redshifts) ** 3 + self['omega_l'] +
                    self['omega_k'] * (1 + redshifts) ** 2) ** -0.5
        if compute_kind == 'spline' or not np.isscalar(redshifts):
            if self.DistanceRedshift is None:
                self.make_distance_v_redshift_spline(redshift_lim, assume_hubble=assume_hubble)
            dist = self.DistanceRedshift.val(redshifts)
        elif compute_kind == 'integrate':
            dint = integrate.quad(get_ddist, 0, redshifts, (self))[0]
            dist = const.hubble_distance * dint
            if assume_hubble:
                dist /= self['hubble']
        return dist

    def distance_angular_diameter(self, redshifts, compute_kind='integrate', assume_hubble=False):
        '''
        Get angular diameter distance[s] {Mpc[/h] physical} from z = 0 to redshift[s].
        *** Inherently in *physical* units.

        Import redshift[s], compute kind (integrate, spline), whether to include Hubble constant.
        '''
        return self.distance(redshifts, compute_kind, assume_hubble=assume_hubble) / (1 + redshifts)

    def distance_luminosity(self, redshifts, compute_kind='integrate', assume_hubble=False):
        '''
        Get luminosity distance[s] {Mpc[/h] comoving} from z = 0 to redshift[s].

        Import redshift[s], compute kind (integrate, spline), whether to include Hubble constant.
        '''
        return self.distance(redshifts, compute_kind, assume_hubble=assume_hubble) * (1 + redshifts)

    def size_per_angle(self, size_kind='kpc/h comoving', angle_kind='arcsec', redshifts=0,
                       compute_kind='integrate'):
        '''
        Get size per angle at redshift[s].

        Import size kind (kpc, kpc/h, Mpc, Mpc/h + comoving, physical),
        angle kind (radian, degree, arcsec), redshift[s], compute kind (integrate, spline).
        '''
        dists = self.distance(redshifts, compute_kind, assume_hubble=False)
        if 'kpc' in size_kind:
            dists *= const.kilo_per_mega
        if '/h' not in size_kind:
            dists /= self['hubble']
        if angle_kind == 'degree':
            dists *= const.radian_per_degree
        elif angle_kind == 'arcsec':
            dists *= const.radian_per_arcsec
        else:
            raise ValueError('not recognize angle kind = %s' % angle_kind)
        if 'physical' in size_kind:
            dists *= 1 / (1 + redshifts)
        elif 'comoving' in size_kind:
            pass
        else:
            raise ValueError('need to specify comoving or physics in size_kind = %s' % size_kind)
        return dists

    def volume(self, redshifts=[0, 0.1], area=7966, volume_kind='regular', assume_hubble=True):
        '''
        Get comoving volume {Mpc[/h] ^ 3} in redshift slice.

        Import redshift range, observed area [degrees ^ 2]
        ('' = full sky, default is SDSS 'legacy' DR7/DR8 spectroscopic sample),
        whether to include Hubble constant.
        '''
        area_frac = 1
        if area:
            area_frac = area / const.deg2_per_sky
        dist_min = self.distance(redshifts[0], assume_hubble=assume_hubble)
        dist_max = self.distance(redshifts[1], assume_hubble=assume_hubble)
        if volume_kind == 'luminosity':
            dist_min *= redshifts[0]
            dist_max *= redshifts[1]
        return 4 / 3 * np.pi * area_frac * (dist_max ** 3 - dist_min ** 3)

    # galaxies ----------
    def convert_mag(self, mags, redshifts, kcorrects=0):
        '''
        Get converted magnitude (either absolute from apparent or vice versa).
        Assumes absolute magnitudes represented as positive.

        Import magnitude[s], redshift[s], k-correction[s].
        '''
        if np.isscalar(redshifts):
            dists_lum = self.distance(redshifts, assume_hubble=False) * (1 + redshifts)
        else:
            redshifts = ut.array.arrayize(redshifts)
            self.make_distance_v_redshift_spline(ut.array.get_limit(redshifts), 100)
            dists_lum = self.distance_luminosity(redshifts, 'spline', assume_hubble=False)
        return -mags + 5 * log10(dists_lum * 1e6 / 10) + kcorrects

    def mag_v_redshift_spline(self, mag_app_max=17.72, z_lim=[0, 0.5], z_wid=0.001):
        '''
        Get spline of absolute magnitude limit v redshift. Neglects evolution of k-correct.

        Import apparent magnitude limit, redshift range & bin width.
        '''
        zs = ut.array.arange_safe(np.array(z_lim) + z_wid, z_wid)
        mags_abs = self.convert_mag(mag_app_max, zs)
        return ut.math.SplinePointClass(zs, mags_abs)

    # simulation ----------
    def particle_mass(self, sim_len, num_per_dimen):
        '''
        Get particle mass {M_sun} given cosmology.

        Import box length {Mpc comoving}, number of particles per dimension.
        '''
        return (const.density_critical_0 * self['hubble'] ** 2 * self['omega_m'] *
                (sim_len / num_per_dimen) ** 3)

    def cell_length(self, sim_len, grid_root_num=7, grid_refine_num=8, redshift=None,
                    units='Mpc comoving'):
        '''
        Get length of grid cell at refinement level,
        in units {comoving or physical} corresponding to sim_len.

        Import box length {Mpc comoving}, number of root grid refinement levels,
        number of adaptive refinement levels, redshift, length units.
        '''
        length = sim_len / 2 ** (grid_root_num + grid_refine_num)
        if 'kpc' in units:
            length *= 1000
        elif 'cm' in units:
            length *= const.cm_per_Mpc
        if '/h' in units:
            length *= self['hubble']
        if 'physical' in units:
            if redshift is None:
                raise ValueError('need to input redshift to scale to physical length')
            else:
                length *= 1 / (1 + redshift)
        return length

    def gas_mass_per_cell(self, sim_len, grid_root_num=7, grid_refine_num=8, num_den=0.3,
                          redshift=0, units='M_sun'):
        '''
        Get mass in cell of given size at given number density.

        Import box length {Mpc comoving}, number of root grid refinement levels,
        number of adaptive refinement levels, hydrogen number density {cm ^ -3 physical},
        redshift, mass units.
        '''
        cell_length = self.cell_length(sim_len, grid_root_num, grid_refine_num, redshift, 'cm')
        mass = const.proton_mass * num_den * cell_length ** 3
        if 'M_sun' in units:
            mass /= const.sun_mass
        if '/h' in units[-2:]:
            mass *= self['hubble']
        return mass
