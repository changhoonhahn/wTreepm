'''
Utility functions for [sub]halo or galaxy catalog.

Author[s]: Andrew Wetzel.
'''

# system -----
from __future__ import division
import numpy as np
from numpy import log10, Inf
# local -----
from .. import constants as const
from . import utility_array as ut_array
from . import utility_coord as ut_coord
from . import utility_io as ut_io
from . import utility_neighbor as ut_neighbor


MRAT_200M_B168 = log10(1.2)    # M_200m / M_fof(b = 0.168)


#===================================================================================================
# list properties of object in catalog
#===================================================================================================
def print_properties(cat, ci):
    '''
    Print (array) properties of object.

    Import [sub]halo / galaxy / galmock catalog, index.
    '''
    for k in sorted(cat.keys()):
        print '%15s' % k, cat[k][ci]


#===================================================================================================
# limits
#===================================================================================================
def get_sfr_limit(sfr_kind='ssfr'):
    '''
    Get limit for SFR distribution.

    Import SFR kind.
    '''
    if 'ssfr' in sfr_kind:
        sfr_lim = [-1e10, -7.0]
    elif sfr_kind == 'dn4k':
        sfr_lim = [1.0, 2.2]
    elif sfr_kind == 'h-alpha.flux':
        sfr_lim = [1e-10, 10]
    elif sfr_kind == 'h-alpha.ew':
        sfr_lim = [1e-10, 3000]
    elif sfr_kind in ('am-qu.spec', 'am-qu.dn4k', 'am-qu.nsa'):
        sfr_lim = [0, 1.01]
    elif sfr_kind == 'g-r':
        sfr_lim = [0, 1.5]
    elif sfr_kind == 'am-qu.color':
        sfr_lim = [0, 2.1]
    elif sfr_kind == 'metal':
        sfr_lim = [0, 0.051]
    else:
        raise ValueError('not recognize sfr kind: %s' % sfr_kind)
    return np.array(sfr_lim)


def get_sfr_bimodal_limit(sfr_kind='ssfr'):
    '''
    Get limit for quiescent & active galaxies in dictionary.

    Import SFR kind.
    '''
    if 'ssfr' in sfr_kind:
        sfr_break = -11
        lo_lim = [-Inf, sfr_break]
        hi_lim = [sfr_break, -1]
    elif sfr_kind == 'dn4k':
        sfr_break = 1.6
        lo_lim = [sfr_break, Inf]
        hi_lim = [1e-10, sfr_break]
    elif sfr_kind == 'h-alpha.flux':
        sfr_break = 0.8
        lo_lim = [1e-10, sfr_break]
        hi_lim = [sfr_break, Inf]
    elif sfr_kind == 'h-alpha.ew':
        sfr_break = 2
        lo_lim = [1e-10, sfr_break]
        hi_lim = [sfr_break, Inf]
    elif sfr_kind in ('am-qu.spec', 'am-qu.dn4k', 'am-qu.nsa'):
        sfr_break = 0.5
        lo_lim = [sfr_break, 1.01]
        hi_lim = [0, sfr_break]
    elif sfr_kind == 'g-r':
        sfr_break = 0.76
        lo_lim = [sfr_break, Inf]
        hi_lim = [-Inf, sfr_break]
    elif sfr_kind == 'am-qu.color':
        sfr_break = 1.5
        lo_lim = [1, 1.1]
        hi_lim = [2, 2.1]
    elif sfr_kind == 'metal':
        sfr_break = 0.023
        lo_lim = [0, sfr_break]
        hi_lim = [sfr_break, Inf]
    else:
        raise ValueError('not recognize sfr kind: %s' % sfr_kind)
    return {'lo': lo_lim, 'hi': hi_lim, 'break': sfr_break, 'sfrlo': lo_lim, 'sfrhi': hi_lim}


def get_morph_bimodal_limit(morph_kind='zooc'):
    '''
    Get limit for concentrated & extended galaxies.

    Import morphology kind (zoo-c, zoo-f, zoo-f:[fraction], concen, sersic).
    '''
    if morph_kind == 'concen':
        mobreak = 2.6
        lo_lim = [1., mobreak]
        hi_lim = [mobreak, 20.]
    elif morph_kind == 'sersic':
        mobreak = 2.6
        lo_lim = [0, mobreak]
        hi_lim = [mobreak, 20.]
    elif 'zoof' in morph_kind:
        if ':' in morph_kind:
            ii = morph_kind.find(':')
            mobreak = float(morph_kind[ii + 1:])
        else:
            mobreak = 0.6
        lo_lim = [0., mobreak]
        hi_lim = [mobreak, 1.01]
    elif morph_kind == 'zooc':
        mobreak = 1
        lo_lim = [0, 0]
        hi_lim = [1, 1]
    else:
        raise ValueError('not recognize morphology kind: %s' % morph_kind)
    return {'hi': np.array(hi_lim), 'lo': np.array(lo_lim), 'break': mobreak}


def get_morph_bimodal_kind_limit(morph_kind='moell_zooc'):
    '''
    Get key & limit for morphology kind.

    Import morphology kind (mospi, moell, mounc, moseo + _ + zooc, zoof,
    zoof:[fraction], concen, sersic).
    '''
    ii = morph_kind.find('_')
    bi_kind = morph_kind[:ii]
    mo_kind = morph_kind[ii + 1:]
    mo_bi_lim = get_morph_bimodal_limit(mo_kind)
    if mo_kind in ('concen', 'sersic'):
        dict_kind = mo_kind
        if bi_kind == 'mospi':
            bilim = mo_bi_lim['lo']
        elif bi_kind == 'moell':
            bilim = mo_bi_lim['hi']
        else:
            raise ValueError('bi_kind = %s not valid for morph_kind = %s' % (bi_kind, morph_kind))
    elif mo_kind == 'zooc':
        dict_kind = bi_kind + '.c'
        bilim = mo_bi_lim['hi']
    elif 'zoof' in mo_kind:
        dict_kind = bi_kind + '.f'
        bilim = mo_bi_lim['hi']
    return dict_kind, bilim


#===================================================================================================
# indices of sub-sample
#===================================================================================================
def assign_id_to_index(cat, id_name='id', id_min=1):
    '''
    Assign to catalog array that points from id to index in array.
    Safely set null values to -length of array.

    Import catalog, id name, minimum id.
    '''
    if id_name in cat:
        # make sure no duplicate id in tree
        if (cat[id_name][cat[id_name] >= id_min].size !=
            np.unique(cat[id_name][cat[id_name] >= id_min]).size):
            Say = ut_io.SayClass(assign_id_to_index)
            Say.say('! ids are not unique')
        cat[id_name + '-to-index'] = ut_array.initialize_array(cat[id_name].max() + 1)
        cat[id_name + '-to-index'][cat[id_name]] = ut_array.arange_length(cat[id_name])


def indices_catalog(cat, prop_lim={}, cis=None):
    '''
    Import catalog, dictionary with properties to cut on as keys & limits as values,
    prior indices.
    '''
    Say = ut_io.SayClass(indices_catalog)
    for k in prop_lim:
        if k not in cat:
            Say.say('%s in property dict but not in catalog, not use for sub-sample slicing')
            continue
        if np.isscalar(prop_lim[k]):
            lim = [prop_lim[k], Inf]
        elif len(prop_lim[k]) == 2:
            lim = prop_lim[k]
        else:
            Say.say('property %s has limit = %s, with length != 2, not use for sub-sample slicing' %
                    (k, prop_lim[k]))
            continue
        cis = ut_array.elements(cat[k], lim, cis)
    return cis


def indices_subhalo(subz, gm_kind='', gm_lim=None, hm_lim=None, ilk='', dis_mf=0, sis=None):
    '''
    Get sub-sample indices slicing on defined limits.

    Import catalog of subhalo at snapshot, mass kind & range, halo mass range, ilk,
    disrupt mass fraction, prior sub-sample indices.
    '''
    if gm_kind and gm_lim is not None and gm_lim != []:
        sis = ut_array.elements(subz[gm_kind], gm_lim, sis)
    if hm_lim is not None and hm_lim != []:
        sis = ut_array.elements(subz['halo.m'], hm_lim, sis)
    if dis_mf > 0:
        sis = ut_array.elements(subz['m.frac.min'], [dis_mf, Inf], sis)
    if 'cen' in ilk or 'sat' in ilk:
        sis = indices_ilk(subz, ilk, sis)
    return sis


def indices_galaxy(gal, gm_kind='', gm_lim=None, hm_lim=None, ilk='', z_lim=None, ra_lim=None,
                   sfr_kind='', sfr_lim=None, dvir_lim=None, gis=None):
    '''
    Get sub-sample indices slicing on defined limits.

    Import galaxy catalog, property limits, prior sub-sample indices.
    '''
    if gm_kind and gm_lim is not None and gm_lim != []:
        gis = ut_array.elements(gal[gm_kind], gm_lim, gis)
    if hm_lim is not None and hm_lim != []:
        gis = ut_array.elements(gal['halo.m'], hm_lim, gis)
    if ilk is not None and ilk != '':
        gis = indices_ilk(gal, ilk, gis)
    if z_lim is not None and z_lim != []:
        gis = ut_array.elements(gal['z'], z_lim, gis)
    if sfr_kind and sfr_lim is not None and sfr_lim != []:
        gis = ut_array.elements(gal[sfr_kind], sfr_lim, gis)
    if ra_lim is not None and ra_lim != []:
        gis = ut_array.elements(gal['pos'][:, 0], ra_lim, gis)
    if dvir_lim is not None and dvir_lim != []:
        gis = ut_array.elements(gal['dist.vir'], dvir_lim, gis)
    return gis


def indices_ilk(catz, ilk='all', cis=None, cis_2=None, get_indices=False):
    '''
    Get sub-sample indices of those of ilk type.

    Import catalog of subhalo/galaxy at snapshot, ilk (sat, sat+ejected, cen, all),
    prior sub-sample indices, other array to get same indices of, whether to get selection indices.
    '''
    sat_prob_lim = {
        'cen': [0.0, 0.5], 'cen.clean': [0.0, 1e-5], 'cen.clean.neig': [0.0, 1e-5],
        'sat': [0.5, 1.01], 'sat.clean': [0.9, 1.01]
    }
    ilk_lim = {'cen': [1, 2.01], 'cen.clean': 1, 'sat': [-3, 0.01], 'sat.clean': [-2, 0.01]}
    m_rank_lim = {'cen': 1, 'sat': 2}
    if ilk == 'all':
        return cis
    elif ilk in sat_prob_lim:
        if 'prob.sat' in catz:
            cis = ut_array.elements(catz['prob.sat'], sat_prob_lim[ilk], cis, cis_2, get_indices)
            if ilk == 'cen.clean.neig':
                k = 'nearest.halo.dist.vir'
                if k in catz:
                    cis = ut_array.elements(catz[k], [4, Inf], cis)
                else:
                    raise ValueError('request %s, but %s not in catalog' % (ilk, k))
            return cis
        elif 'm.rank' in catz:
            return ut_array.elements(catz['m.rank'], m_rank_lim[ilk], cis, cis_2, get_indices)
        elif 'ilk' in catz:
            return ut_array.elements(catz['ilk'], ilk_lim[ilk], cis, cis_2, get_indices)
    else:
        raise ValueError('not recognize ilk = %s' % ilk)


def indices_sfr(gal, bi_kind='lo', sfr_kinds='ssfr', gis=None, gis_2=None, get_indices=False):
    '''
    If multiple SFR kinds, return overlapping indices in both sets for quiescent, in either set for
    active.

    Import galaxy catalog, SFR bimodality region (lo, hi),
    SFR kind[s] (ssfr, dn4k, g-r) [single string with spaces], prior sub-sample indices,
    other array to get same indices of, whether to get selection indices.
    '''
    sfr_kinds_split = sfr_kinds.split()
    if len(sfr_kinds_split) == 1:
        bi_lim = get_sfr_bimodal_limit(sfr_kinds)
        return ut_array.elements(gal[sfr_kinds], bi_lim[bi_kind], gis, gis_2, get_indices)
    elif len(sfr_kinds_split) == 2:
        if gis_2 is not None or get_indices:
            raise ValueError('not support second array or indices for multiple sfr_kinds')
        is_sfr = []
        for sfr_kind in sfr_kinds_split:
            bi_lim = get_sfr_bimodal_limit(sfr_kind)
            is_sfr.append(ut_array.elements(gal[sfr_kind], bi_lim[bi_kind], gis_2, gis))
        if 'lo' in bi_kind:
            return np.intersect1d(is_sfr[0], is_sfr[1])
        elif 'hi' in bi_kind:
            return np.union1d(is_sfr[0], is_sfr[1])


def indices_morphology(gal, morph_kinds='zoo.c', gis=None, gis_2=None, get_indices=False):
    '''
    If multiple morph kinds, return overlapping indices in both sets.

    Import galaxy catalog, bimodality region (mo.ell, mo.spi), morph kind[s] [string with spaces],
    prior sub-sample indices, other array to get same indices of,
    whether to get selection indices.
    '''
    morphkinds_split = morph_kinds.split()
    if len(morphkinds_split) == 1:
        dict_kind, bi_lim = get_morph_bimodal_kind_limit(morph_kinds)
        return ut_array.elements(gal[dict_kind], bi_lim, gis, gis_2, get_indices)
    elif len(morphkinds_split) == 2:
        if gis_2 is not None or get_indices:
            raise ValueError('not support second array or indices for multiple morph_kinds')
        gis_mo = []
        for morphkind in morphkinds_split:
            dict_kind, bi_lim = get_morph_bimodal_kind_limit(morphkind)
            gis_mo.append(ut_array.elements(gal[dict_kind], bi_lim, gis))
        return np.intersect1d(gis_mo[0], gis_mo[1])


def indices_in_halo(subz, si, m_kind='m.max', m_lim=[11, Inf], ilk='sat'):
    '''
    Get indices of subhalos in mass range in halo (can include self).

    Import catalog of subhalo at snapshot, subhalo index, mass kind & range, ilk (sat, all).
    '''
    ceni = subz['cen.i'][si]
    sis = ut_array.elements(subz['cen.i'], ceni)
    sis = ut_array.elements(subz[m_kind], m_lim, sis)
    if ilk == 'sat':
        sis = sis[sis != ceni]
    return sis


#===================================================================================================
# velocity/redshift
#===================================================================================================
def get_velocity_dif(catz_1, catz_2, cis_1, cis_2, vel_kind='vector', include_hubble_flow=True,
                     units='Mpc/Gyr', dimen_is=[0, 1, 2]):
    '''
    Get relative velocities {physical} of 2 wrt 1.

    Import catalogs of [sub]halos at snapshot, [sub]halo indices,
    velocity kind to get (vector, scalar, scalar^2),
    whether to include hubble flow, output units (Mpc/Gyr, km/s),
    spatial dimensions to find velocity of.
    '''
    if np.isscalar(cis_1) and np.isscalar(cis_2):
        dimen_shape = 0
    else:
        dimen_shape = 1
        if np.isscalar(cis_1):
            cis_1 = ut_array.arrayize(cis_1, bit_num=32)
        if np.isscalar(cis_2):
            cis_2 = ut_array.arrayize(cis_2, bit_num=32)
    vel_difs = catz_1['vel'][cis_2][:, dimen_is] - catz_1['vel'][cis_1][:, dimen_is]
    vel_difs *= catz_1.snap['a']    # relative peculiar velocities {Mpc / Gyr physical}
    # add hubble flow (dr/dt = a * dx/dt + da/dt * x)
    if include_hubble_flow:
        vel_difs += (catz_1.snap['a'] / catz_1.snap['t.hubble'] *
                     ut_coord.distance('vector', catz_1['pos'][cis_1], catz_2['pos'][cis_2],
                                       catz_1.info['box.length'], dimen_is))
    if units == 'km/s':
        vel_difs *= const.km_per_Mpc / const.sec_per_Gyr
    if 'scalar' in vel_kind and not np.isscalar(dimen_is):
        vel_difs = (vel_difs ** 2).sum(dimen_shape)
        if vel_kind == 'scalar':
            vel_difs = vel_difs ** 0.5
    return vel_difs


def get_redshift_in_box(catz, cis=None, dimi=2):
    '''
    Differential redshifts are valid (at least) locally.

    Import catalog of [sub]halo at snapshot, indices, which dimension to compute redshift from.
    Get redshift of each object in box, relative to snapshot redshift at pos = 0.
    '''
    if cis is None:
        cis = ut_array.arange_length(catz['pos'][:, dimi])
    # hubble flow
    vels = catz['pos'][cis][:, dimi] * catz.snap['a'] / catz.snap['t.hubble']
    vels += catz.snap['a'] * catz['vel'][cis][:, dimi]    # peculiar velocity
    vels *= ut_coord.velocity_unit(catz.snap['a'], catz.Cosmo['hubble'], 'km/s')
    return ut_coord.redshift_from_velocity(vels) + catz.snap['z']


#===================================================================================================
# history
#===================================================================================================
def get_tree_direction_info(zi_start=None, zi_end=None, direction_kind=''):
    '''
    Get snapshot step (+1 or -1) & dictionary key corresponding to parent/child.

    Import starting & ending snapshot indices (forward or backward) OR
    direction kind ('parent', 'child').
    '''
    if direction_kind:
        if direction_kind == 'parent':
            zi_step = 1
            family_key = 'par.i'
        elif direction_kind == 'child':
            zi_step = -1
            family_key = 'chi.i'
    else:
        if zi_end < 0:
            raise ValueError('u.end = %d out of bounds' % zi_end)
        elif zi_end == zi_start:
            raise ValueError('u.end = u.start')
        if zi_end > zi_start:
            zi_step = 1
            family_key = 'par.i'
        elif zi_end < zi_start:
            zi_step = -1
            family_key = 'chi.i'
    return zi_step, family_key


def indices_tree(cat, zi_start, zi_end, cis=None, get_indices=False):
    '''
    Get parent / child index[s] of input cis at zi_end.
    Assign negative value to [sub]halo if cannot track all the way back.

    Import [sub]halo catalog, starting & ending snapshots (forward or backward), index[s].
    '''
    if zi_start == zi_end:
        return cis
    elif zi_end >= len(cat):
        raise ValueError('zi.end = %d not within %s snapshot limit = %d' %
                         (zi_end, cat.info['kind'], len(cat) - 1))
    zi_step, tree_kind = get_tree_direction_info(zi_start, zi_end)
    if cis is None:
        cis_tree = ut_array.arange_length(cat[zi_start][tree_kind])
    else:
        cis_tree = ut_array.arrayize(cis, bit_num=32)
    for zi in xrange(zi_start, zi_end, zi_step):
        ciis = ut_array.elements(cis_tree, [0, Inf])
        cis_tree[ciis] = cat[zi][tree_kind][cis_tree[ciis]]
    ciis = ut_array.elements(cis_tree, [0, Inf])
    cis_end = np.zeros(cis_tree.size, cis_tree.dtype) - 1 - cat[zi_end][tree_kind].size
    cis_end[ciis] = cis_tree[ciis]
    if get_indices:
        return ut_array.scalarize(cis_end), ut_array.scalarize(ciis)
    else:
        return ut_array.scalarize(cis_end)


def is_in_same_halo(sub, hal, zi_child, si_child, zi_par, si_par):
    '''
    Get 1 or 0 if subhalo child's parent is in subhalo child's halo's parent.

    Import subhalo & halo catalog, subhalo child snapshot & index, subhalo parent snapshot & index.
    '''
    if zi_child < 0 or si_child <= 0 or zi_par < 0 or si_par <= 0:
        return 0
    hi_par = sub[zi_par]['halo.i'][si_par]
    hi = sub[zi_child]['halo.i'][si_child]
    if hi_par > 0 and hi > 0:
        for zi in xrange(zi_child, zi_par):
            hi = hal[zi]['par.i'][hi]
            if hi <= 0:
                return 0
        if hi_par == hi:
            return 1
    return 0


def is_orphan(hal, zi_now, zi_max, hi, m_kind='m.fof', sub=None):
    '''
    Get 1 or 0 if halo was ever orphan/ejected while above mass cut.

    Import halo catalog, current & maximum snapshot, halo index, halo mass kind.
    subhalo catalog (sanity check, not seem to matter much).
    '''
    m_min_res = log10(80 * hal.info['particle.m'])    # cannot determine orphan if too few particles

    m_min = max(hal[zi_now][m_kind][hi] - 0.3, m_min_res)    # 'new' halo if grew by > 50%
    par_zi, par_hi = zi_now, hi
    while par_zi < zi_max and par_hi > 0:
        if hal[par_zi][m_kind][par_hi] > m_min:
            if hal[par_zi]['par.i'][par_hi] <= 0:
                if sub is None:
                    return 1
                elif hal[par_zi]['cen.i'][par_hi] <= 0:
                    return 1
                elif sub[par_zi]['par.i'][hal[par_zi]['cen.i'][par_hi]] > 0:
                    return 1
                else:
                    return 0
        par_zi, par_hi = par_zi + 1, hal[par_zi]['par.i'][par_hi]
    return 0


def print_history(cat, zi_now, ci, zi_end=None, prop='m.max'):
    '''
    Print properties across snapshots.

    Import catalog of [sub]halo, current snapshot & index, ending snapshot, property to print.
    '''
    Say = ut_io.SayClass(print_history)
    if zi_end is None:
        zi_end = len(cat) - 1
    elif zi_end >= len(cat):
        Say.say('! zi.end = %d not within %s snapshot limit, setting to %d' %
                (zi_end, cat.info['kind'], len(cat) - 1))
        zi_end = len(cat) - 1
    zi_step, tree_kind = get_tree_direction_info(zi_now, zi_end)
    if zi_now > zi_end:
        zi_end -= 1
    elif zi_now < zi_end:
        zi_end += 1
    zi = zi_now
    for zi in xrange(zi_now, zi_end, zi_step):
        if cat.info['kind'] == 'subhalo':
            Say.say('zi %2d | ci %6d | ilk %2d | %s %6.3f' %
                    (zi, ci, cat[zi]['ilk'][ci], prop, cat[zi][prop][ci]))
        elif cat.info['kind'] == 'halo':
            Say.say('zi %2d  z %.4f | ci %6d | pos %7.3f, %7.3f, %7.3f | %s %6.3f' %
                    (zi, cat[zi].snap['z'], ci,
                     cat[zi]['pos'][ci, 0], cat[zi]['pos'][ci, 1], cat[zi]['pos'][ci, 2],
                     prop, cat[zi][prop][ci]))
        ci = cat[zi][tree_kind][ci]
        if ci < 0:
            break


def print_prop_extrema(cat):
    '''
    Print minimum & maximum value of each property across snapshots.

    Import [sub]halo catalog.
    '''
    if isinstance(cat, dict):
        cat = [cat]
    for k in cat[0]:
        prop_min, prop_max = Inf, -Inf
        for zi in xrange(len(cat)):
            if cat[zi][k].size:
                if cat[zi][k].min() < prop_min:
                    prop_min = cat[zi][k].min()
                if cat[zi][k].max() > prop_max:
                    prop_max = cat[zi][k].max()
        print '%s %.5f, %.5f' % (k, prop_min, prop_max)


#===================================================================================================
# neighbor finding
#===================================================================================================
def get_catalog_neighbor(
    catz, gm_kind='m.star', gm_lim=[9.7, Inf], hm_lim=[], ilk='', neig_gm_kind='m.star',
    neig_gm_lim=[9.7, Inf], neig_hm_lim=[], neig_ilk='', dis_mf=0, neig_num_max=2000,
    neig_dist_max=0.20, dist_kind='comoving', space='real', neig_veldif_maxs=None,
    find_kind='kd-tree', format_list=True, center_is=None, neig_is=None):
    '''
    For up to num_max neighbors within dist_max, assign counts, distances, & indices.

    Import catalog of [sub]halo at snapshot, galaxy mass kind & range, halo mass range, ilk,
    neighbor mass kind & range, halo mass range, ilk,
    disruption mass fraction for both (ignore for neighbor if just cut on its halo mass),
    maximum number of neighbors per center,
    neighbor distance maximum {Mpc}, distance kind (comoving, physical) & space (real, red, proj),
    neighbor line-of-sight velocity difference maximum[s] {km/s} (if space = proj),
    neighbor finder (direct, mesh, kd-tree), prior indices.
    '''
    neig = {
        'info': {
            'm.kind': gm_kind, 'm.lim': gm_lim,
            'neig.m.kind': neig_gm_kind, 'neig.m.lim': neig_gm_lim,
            'dist.max': neig_dist_max, 'dist.kind': dist_kind, 'dist.num.max': neig_num_max,
            'neig.num.tot': 0
        },
        'self.i': [],    # index of self in [sub]halo/galaxy catalog
        # 'self.i.inv': [],    # to go from index in [sub]halo/galaxy catalog to index in this list
        'num': [],    # number of neighbors within mass & distance range, up to neig_num_max
        'distances': [],    # distances of neighbors, sorted
        'indices': []    # indices of neighbors, sorted by distance
    }
    dimen_num = 3
    redspace_dimi = 2

    Say = ut_io.SayClass(get_catalog_neighbor)
    Neig = ut_neighbor.NeigClass()
    # objects around which to find neighbors
    if center_is is None:
        center_is = indices_subhalo(catz, gm_kind, gm_lim, hm_lim, ilk, dis_mf)
    # potential neighbor objects
    if neig_is is None:
        if neig_gm_lim and neig_gm_lim[0] <= 1 and neig_hm_lim and neig_hm_lim[0] > 1:
            # if cutting on neighbor halo mass, ignore subhalo disruption
            dis_mf = 0
        neig_is = indices_subhalo(catz, neig_gm_kind, neig_gm_lim, neig_hm_lim, neig_ilk, dis_mf)
    poss = catz['pos'][center_is]
    neig_poss = catz['pos'][neig_is]
    neig['info']['neig.num.tot'] = neig_is.size
    if space == 'red':
        # apply redshift space distortions to find via sphere in redshift space
        poss[:, redspace_dimi] = ut_coord.position_redspace(
            poss[:, 2], catz['vel'][center_is][:, 2], catz.snap['t.hubble'],
            catz.info['box.length'])
        neig_poss[:, redspace_dimi] = ut_coord.position_redspace(
            neig_poss[:, 2], catz['vel'][neig_is][:, 2], catz.snap['t.hubble'],
            catz.info['box.length'])
    elif space == 'proj':
        # get neighbors in projection, apply line-of-sight velocity cut later if defined
        proj_dim_is = np.setdiff1d(np.arange(dimen_num), [redspace_dimi])
        poss = poss[:, proj_dim_is]
        neig_poss = neig_poss[:, proj_dim_is]
    sim_len = catz.info['box.length']
    if dist_kind == 'physical':
        poss *= catz.snap['a']
        neig_poss *= catz.snap['a']
        sim_len *= catz.snap['a']
    neig['indices'], neig['distances'], neig['num'] = Neig.get_neig(
        'indices distances number', poss, neig_poss, neig_num_max, [5e-5, neig_dist_max], sim_len,
        find_kind, format_list, neig_is)
    if neig_veldif_maxs is not None:
        # keep only neighbors found in projection that are within velocity cut of center
        if space != 'proj':
            raise ValueError('neig_veldif_maxs defined, but space is not projected')
        if np.isscalar(neig_veldif_maxs):
            neig_veldif_maxs = np.zeros(len(neig['indices']), np.float32) + neig_veldif_maxs
        if not np.isscalar(neig_veldif_maxs):
            if len(neig_veldif_maxs) != len(neig['indices']):
                raise ValueError('neig_veldif_maxs size = %d but centers size = %d' %
                                 (len(neig_veldif_maxs), len(neig['indices'])))
        neig_iss = neig['indices']
        neig_distss = neig['distances']
        neig['indices'] = []
        neig['distances'] = []
        neig_keep_num = 0
        neig_tot_num = 0
        for neig_iis, neig_is in enumerate(neig_iss):
            keeps = (abs(get_velocity_dif(catz, catz, center_is[neig_iis], neig_is, 'scalar',
                                          units='km/s', dimen_is=redspace_dimi)) <
                     neig_veldif_maxs[neig_iis])
            neig['indices'].append(neig_is[keeps])
            neig['distances'].append(neig_distss[neig_iis][keeps])
            neig_keep_num += keeps.sum()
            neig_tot_num += keeps.size
        Say.say('keep %d of %d (%.2f) neig within velocity difference' %
                (neig_keep_num, neig_tot_num, neig_keep_num / neig_tot_num))
    if 'self.i' in neig:
        neig['self.i'] = center_is
    if 'self.i.inv' in neig:
        neig['self.i.inv'] = ut_array.arange_length(catz[gm_kind])
        neig['self.i.inv'][center_is] = ut_array.arange_length(center_is.size)
    return neig


class NearestNeigClass(ut_io.SayClass):
    '''
    Find nearest (minimum distance or mininum (distance / R_vir(neig))) more massive neighbor
    & store its properties.
    '''
    def __init__(self, nearest=None):
        '''
        Import previous nearest neighbor dictionary.
        '''
        if nearest is not None:
            self.nearest = nearest

    def assign(
        self, halz, m_kind='m.vir', m_lim=[11, Inf], neig_m_lim=[11, Inf], neig_num_max=100,
        neig_dist_max=1, dist_kind='comoving', dist_scaling_kind='virial', find_kind='kd-tree'):
        '''
        Store dictionary of nearest neighbor's index, distance, distance / R_vir(neig), mass.

        Import halo catalog at snapshot, mass kind & range,
        neighbor mass range & maximum number & maximum distance {Mpc} & kind (physical, comoving),
        distance scaling to identify nearest (virial, center), neighbor finding kind.
        '''
        # make sure neighbor mass minimum is above that of centers
        neig = get_catalog_neighbor(
            halz, m_kind, m_lim, [], '', m_kind, neig_m_lim, [], '', 0, neig_num_max, neig_dist_max,
            dist_kind, 'real', find_kind=find_kind, format_list=True)

        nearest = {
            'self.i': neig['self.i'],
            'i': ut_array.initialize_array(neig['self.i'].size),
            'dist.cen': np.zeros(neig['self.i'].size, np.float32) + np.array(Inf, np.float32),
            'dist.vir': np.zeros(neig['self.i'].size, np.float32) + np.array(Inf, np.float32),
            m_kind: np.zeros(neig['self.i'].size, np.float32)
        }

        for hii, hi in enumerate(neig['self.i']):
            # keep only neighbors more massive than self
            keeps = halz[m_kind][neig['indices'][hii]] > halz[m_kind][hi]
            if len(keeps) and keeps.max():
                neig_is = neig['indices'][hii][keeps]
                cen_dists = neig['distances'][hii][keeps]    # distance to center of neighbor
                vir_dists = cen_dists / halz['r.vir'][neig_is]
                if dist_scaling_kind == 'virial':
                    iinear = np.argmin(vir_dists)
                elif dist_scaling_kind == 'center':
                    iinear = np.argmin(cen_dists)
                else:
                    raise ValueError('not recognize dist_scaling_kind = %s' % dist_scaling_kind)
                nearest['dist.vir'][hii] = vir_dists[iinear]
                nearest['dist.cen'][hii] = cen_dists[iinear]
                nearest['i'][hii] = neig_is[iinear]
                nearest[m_kind][hii] = halz[m_kind][neig_is[iinear]]
        nearest_num = nearest['i'][nearest['i'] >= 0].size
        self.say('%d of %d (%.2f) have nearby more massive halo' %
                 (nearest_num, neig['self.i'].size, nearest_num / neig['self.i'].size))
        self.nearest = nearest
        self.m_kind = m_kind

    def assign_to_catalog(self, halz):
        '''
        Assign nearest neighbor properties to halo catalog.

        Import halo catalog at snapshot.
        '''
        base_name = 'nearest.halo.'
        props = self.nearest.keys()
        props.remove('self.i')
        for prop in props:
            halz[base_name + prop] = np.zeros(halz[self.m_kind].size,
                                              self.nearest[prop].dtype) - 1
            halz[base_name + prop][self.nearest['self.i']] = self.nearest[prop]

NearestNeig = NearestNeigClass()


#===================================================================================================
# mass, luminosity, SFR, morphology
#===================================================================================================
def convert_m(subz, m_from='m.star', m_min=10, m_to='m.max'):
    '''
    Get 'to' mass/magnitude corresponding to 'from' one, assuming no scatter in relation.

    Import catalog of subhalo at snapshot, input mass kind & value, output mass kind.
    '''
    temp = -subz[m_from]
    m_min = -m_min
    sis_sort = np.argsort(temp)
    si = temp[sis_sort].searchsorted(m_min)
    return subz[m_to][sis_sort[si]]


def get_ms_parent(cat, zi, ci, zi_par):
    '''
    Get all parent masses at zi.par, sort decreasing by mass.

    Import [sub]halo catalog, snapshot, index, parent shapshot minimum.
    '''
    def ms_parent_recursive(cat, zi, ci, zi_par, m_kind):
        '''
        Recursively walk each parent tree back to zi_par, append mass at zi_par.
        '''
        masses = []
        p_zi, p_ci = zi + 1, cat[zi]['par.i'][ci]
        while 0 <= p_zi < cat.snap['z'].size and p_ci > 0:
            if p_zi >= zi_par:
                masses.append(cat[zi_par][m_kind][p_ci])
            else:
                masses += ms_parent_recursive(cat, p_zi, p_ci, zi_par, m_kind)
            p_zi, p_ci = zi_par, cat[zi_par]['par.n.i'][p_ci]
        return masses
    m_kind = cat.info['m.kind']
    return np.sort(ms_parent_recursive(cat, zi, ci, zi_par, m_kind))[::-1]


def assign_num_rank(catz, prop='m.max'):
    '''
    Assign ranked number to objects based on prop.

    Import catalog of [sub]halo at snapshot, property to rank.
    '''
    cis = ut_array.arange_length(catz[prop])
    cis_sort = np.argsort(catz[prop])[::-1]
    prop_num = prop + '.rank'
    catz[prop_num] = np.zeros(cis.size)
    catz[prop_num][cis_sort] = cis
