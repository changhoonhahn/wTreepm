'''
Utility functions for position, velocity in simulation or observation.

Author[s]: Andrew Wetzel.
'''

# system -----
from __future__ import division
import numpy as np
# local -----
from .. import constants as const


#===================================================================================================
# distance/velocity conversion
#===================================================================================================
def position(poss, sim_len=None):
    '''
    Get position in range [0, sim_len).

    Import position[s], simulation box length (if none, return array as is).
    '''
    if sim_len is None:
        return poss
    if np.isscalar(poss):
        if poss >= sim_len:
            poss -= sim_len
        elif poss < 0:
            poss += sim_len
    else:
        poss[poss >= sim_len] -= sim_len
        poss[poss < 0] += sim_len
    return poss


def position_dif(pos_difs, sim_len=None):
    '''
    Get separation (with direction) [-sim_len/2, sim_len/2).

    Import position difference[s], simulation box length (if none, return array as is).
    '''
    if sim_len is None:
        return pos_difs
    if np.isscalar(pos_difs):
        if pos_difs >= 0.5 * sim_len:
            pos_difs -= sim_len
        elif pos_difs < -0.5 * sim_len:
            pos_difs += sim_len
    else:
        pos_difs[pos_difs >= 0.5 * sim_len] -= sim_len
        pos_difs[pos_difs < -0.5 * sim_len] += sim_len
    return pos_difs


def distance(dist_kind='vector', poss_1=None, poss_2=None, sim_len=None, dim_is=[0, 1, 2]):
    '''
    Get distance between objects.

    Import distance kind to get (vector, scalar, scalar^2), position arrays,
    simulation box length, (if none, not use periodic), spatial dimensions to take difference of.
    '''
    poss_1 = np.array(poss_1)
    poss_2 = np.array(poss_2)
    if len(poss_1.shape) == 1:
        poss_1 = poss_1[dim_is]
    else:
        poss_1 = poss_1[:, dim_is]
    if len(poss_2.shape) == 1:
        poss_2 = poss_2[dim_is]
    else:
        poss_2 = poss_2[:, dim_is]
    if len(poss_1.shape) == 1 and len(poss_2.shape) == 1:
        shape_pos = 0
    else:
        shape_pos = 1
    dists = position_dif(poss_2 - poss_1, sim_len)
    if 'scalar' in dist_kind:
        dists = (dists ** 2).sum(shape_pos)
        if dist_kind == 'scalar':
            dists = dists ** 0.5
    return dists


def distance_ang(dist_kind='scalar', poss_1=None, poss_2=None, ang_sphere=360):
    '''
    Get angular separation, valid for small separations.

    Import distance kind (scalar, scalar^2), 2 position (RA, dec) arrays,
    angular size of sphere (360 = degrees, 2 * pi = radians).
    '''
    if ang_sphere == 360:
        ang_scale = const.radian_per_degree
    elif ang_sphere == 2 * np.pi:
        ang_scale = 1
    else:
        raise ValueError('angphere = %.2f not make sense' % ang_sphere)
    if np.ndim(poss_1) == 1 and poss_1.size == 2:
        ras_1, decs_1 = poss_1[0], poss_1[1]
    else:
        ras_1, decs_1 = poss_1[:, 0], poss_1[:, 1]
    if np.ndim(poss_2) == 1 and poss_2.size == 2:
        ras_2, decs_2 = poss_2[0], poss_2[1]
    else:
        ras_2, decs_2 = poss_2[:, 0], poss_2[:, 1]
    dist2s = ((position_dif(ras_1 - ras_2, ang_sphere) *
               np.cos(ang_scale * 0.5 * (decs_1 + decs_2))) ** 2 + (decs_1 - decs_2) ** 2)
    if dist_kind == 'scalar^2':
        pass
    elif dist_kind == 'scalar':
        dist2s = dist2s ** 0.5
    else:
        raise ValueError('not recognize distance kind = %s' % dist_kind)
    return dist2s
    '''
    if compkind == 'exact':
        # gives poor results - rounding error in arccos?
        temp = (np.sin(dec1s * ang_scale) * np.sin(dec2s * ang_scale) +
                        np.cos(dec1s * ang_scale) * np.cos(dec2s * ang_scale) *
                        np.cos(position_dif(ra1s - ra2s, ang_sphere) * ang_scale)).clip(-1, 1)
        return np.arccos(temp) / ang_scale
    '''


# position - velocity conversion:
# dr/dt = a * dx/dt + da/dt * x = a(t) * dx/dt + r * H(t)

def velocity_from_redshift(redshift, solve_kind='exact'):
    '''
    Get velocity along the line of sight {km/s} from redshift.
    Independent of cosmology.

    Import redshift, solve_kind (exact, approx).
    '''
    if solve_kind == 'exact':
        return (((1 + redshift) ** 2 - 1) / ((1 + redshift) ** 2 + 1) * const.speed_light *
                const.kilo_per_centi)
    elif solve_kind == 'approx':
        return const.speed_light * redshift * const.kilo_per_centi


def redshift_from_velocity(vel, solve_kind='exact'):
    '''
    Get reshift from velocity {km/s} along the line of sight.
    Independent of cosmology.

    Import velocity {km/s}, solve_kind (exact, approx).
    '''
    if solve_kind == 'exact':
        return ((1 + vel * const.centi_per_kilo / const.speed_light) /
                (1 - vel * const.centi_per_kilo / const.speed_light)) ** 0.5 - 1
    elif solve_kind == 'approx':
        return vel * const.centi_per_kilo / const.speed_light


def position_dif_from_velocity_dif(vel_dif, hubble_time):
    '''
    Get position difference {Mpc comoving} from velocity difference (redshift-space distortion).

    Import peculiar velocity {Mpc comoving / Gyr}, hubble time = 1 / H.
    '''
    return vel_dif * hubble_time


def position_dif_from_redshift_dif(z_dif, hubble_time):
    '''
    Get position difference {Mpc comoving} from redshift difference (redshift-space distortion).
    *** Get *approximate* distance, in non-relativistic limit.

    Import redshift difference, hubble time = 1 / H.
    '''
    return velocity_from_redshift(z_dif, 'approx') * hubble_time


def position_redspace(pos, vel, hubble_time, sim_len=None):
    '''
    Get position in redshift space.

    Import position {Mpc}, peculiar velocity {Mpc comoving / Gyr}, hubble time = 1 / H,
    simulation box length.
    '''
    return position(pos + position_dif_from_velocity_dif(vel, hubble_time), sim_len)


def velocity_unit(aexp, hubble, vel_to='km/s'):
    '''
    Get multiplicative factor to convert velocity from {comoving peculiar Mpc/Gyr} to
    {physical peculiar km/s}, or vice versa.

    Import scale factor, dimensionless hubble constant, velocity units to get (km/s, Mpc/Gyr).
    '''
    factor = aexp / hubble * const.km_per_Mpc / const.sec_per_Gyr
    if vel_to == 'km/s':
        return factor
    elif vel_to == 'Mpc/Gyr':
        return 1 / factor
    else:
        raise ValueError('not recognize velocity to = %s' % vel_to)
