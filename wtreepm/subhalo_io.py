'''
hacked light-weight version of andrew wetzel's treepm code 

Reads in subtree.dat & halotree.dat from tree directory.

Masses in log {M_sun}, distances in {Mpc comoving}.

simulation_length {Mpc/h}    particle_number_per_dimension    particle_mass {M_sun}
50    256    7.710e8
50    512    8.976e7
64    800    4.934e7
100   800    1.882e8
125   1024   1.753e8
200   1500   2.284e8
250   2048   1.976e8
720   1500   1.066e10

lcdm250
zi    aexp      redshift  t {Gyr} t_wid {Gyr}
 0    1.0000    0.0000    13.8099 0.6771
 1    0.9522    0.0502    13.1328 0.6604
 2    0.9068    0.1028    12.4724 0.6453
 3    0.8635    0.1581    11.8271 0.6291
 4    0.8222    0.2162    11.1980 0.6087
 5    0.7830    0.2771    10.5893 0.5905
 6    0.7456    0.3412     9.9988 0.5700
 7    0.7100    0.4085     9.4289 0.5505
 8    0.6760    0.4793     8.8783 0.5259
 9    0.6438    0.5533     8.3525 0.5060
10    0.6130    0.6313     7.8464 0.4830
11    0.5837    0.7132     7.3635 0.4587
12    0.5559    0.7989     6.9048 0.4382
13    0.5293    0.8893     6.4665 0.4152
14    0.5040    0.9841     6.0513 0.3916
15    0.4800    1.0833     5.6597 0.3724
16    0.4570    1.1882     5.2873 0.3496
17    0.4352    1.2978     4.9378 0.3298
18    0.4144    1.4131     4.6080 0.3099
19    0.3946    1.5342     4.2980 0.2901
20    0.3758    1.6610     4.0079 0.2735
21    0.3578    1.7949     3.7343 0.2541
22    0.3408    1.9343     3.4802 0.2394
23    0.3245    2.0817     3.2408 0.2235
24    0.3090    2.2362     3.0172 0.2094
25    0.2942    2.3990     2.8078 0.1942
26    0.2802    2.5689     2.6136 0.1821
27    0.2668    2.7481     2.4315 0.1704
28    0.2540    2.9370     2.2611 0.1576
29    0.2419    3.1339     2.1035 0.1466
30    0.2304    3.3403     1.9569 0.1371
31    0.2194    3.5579     1.8198 0.1280
32    0.2089    3.7870     1.6918 0.1192
33    0.1989    4.0277     1.5726 0.1106
34    0.1894    4.2798     1.4620 0.0000
'''
from __future__ import division
from numpy import log10, Inf, int32, float32
import numpy as np
import os
import glob
import copy
# local -----
from . import cosmology
from . import utility as ut


#===================================================================================================
# read in
#===================================================================================================
# Martin's TreePM ----------
class TreepmClass(ut.io.SayClass):
    '''
    Read [sub]halo catalog snapshots, return as list class.
    '''
    def __init__(self, sigma_8=0.8):
        self.treepm_directory = os.environ.get('TREEPM_DIR') 
        self.dimen_num = 3
        self.particle_num = {
            # connect simulation box length {Mpc/h comoving} to number of particles per dimension.
            50: 256,    # 512
            64: 800,
            100: 800,
            125: 1024,
            200: 1500,
            250: 2048,
            720: 1500
        }
        self.sigma_8 = sigma_8

    def read(self, catalog_kind='subhalo', box_length=250, zis=1, cat_in=None):
        '''
        Read snapshots in input range.

        Import catalog kind (subhalo, halo, both), simulation box length {Mpc/h comoving},
        snapshot index range, input [sub]halo catalog to appending snapshots to.
        '''
        if catalog_kind == 'both':
            sub = self.read('subhalo', box_length, zis)
            hal = self.read('halo', box_length, zis)
            return sub, hal
        elif catalog_kind == 'subhalo':
            cat = ut.array.ListClass()
            catz = ut.array.DictClass()
            catz['pos'] = []    # position (3D) of most bound particle {Mpc/h -> Mpc comoving}
            catz['vel'] = []    # velocity (3D) {Mpc/h/Gyr -> Mpc/Gyr comoving}
            #catz['m.bound'] = []    # mass of subhalo {M_sun}
            #catz['vel.circ.max'] = []    # maximum of circular velocity {km/s physical}
            catz['m.max'] = []    # maximum mass in history {M_sun}
            #catz['vel.circ.peak'] = []    # max of max of circular velocity {km/s physical}
            catz['ilk'] = []
            # 1 = central, 2 = virtual central, 0 = satellite, -1 = virtual satellite,
            # -2 = virtual satellite with no central, -3 = virtual satellite with no halo
            catz['par.i'] = []    # index of parent, at previous snapshot, with highest M_max
            #catz['par.n.i'] = []    # index of next parent to same child, at same snapshot
            catz['chi.i'] = []    # index of child, at next snapshot
            catz['m.frac.min'] = []    # minimum M_bound / M_max experienced
            #catz['m.max.rat.raw'] = []    # M_max ratio of two highest M_max parents (< 1)
            catz['cen.i'] = []    # index of central subhalo in same halo (can be self)
            catz['dist.cen'] = []    # distance from central {Mpc/h -> Mpc comoving}
            #catz['sat.i'] = []    # index of next highest M_max satellite in same halo
            catz['halo.i'] = []    # index of host halo
            catz['halo.m'] = []    # FoF mass of host halo {M_sun}
            #catz['inf.last.zi'] = []    # snapshot before fell into current halo
            #catz['inf.last.i'] = []    # index before fell into current halo
            #catz['inf.dif.zi'] = []    # snapshot when sat/central last was central/sat
            #catz['inf.dif.i'] = []    # index when sat/central last was central/sat
            #catz['inf.first.zi'] = []    # snapshot before first fell into another halo
            #catz['inf.first.i'] = []    # index before first fell into another halo
            # derived ----------
            catz['m.star'] = []    # stellar mass {M_sun}
            #catz['mag.r'] = []    # magnitude in r-band
            #catz['m.max.rat'] = []    # same as m.max.rat.raw, but incorporates disrupted subhalos
            #catz['m.star.rat'] = []    # M_star ratio of two highest M_star parents
            catz['ssfr'] = []
            #catz['dn4k'] = []
            #catz['g-r'] = []
        elif catalog_kind == 'halo':
            cat = ut.array.ListClass()
            catz = ut.array.DictClass()
            catz['pos'] = []    # 3D position of most bound particle {Mpc/h -> Mpc comoving}
            #catz['vel'] = []    # 3D velocity {Mpc/h/Gyr -> Mpc/Gyr comoving}
            catz['m.fof'] = []    # FoF mass {M_sun}
            catz['vel.circ.max'] = []    # maximum circular vel = sqrt(G * M(r) / r) {km/s physical}
            #catz['v.disp'] = []    # velocity dispersion {km/s} (DM, 1D, no hubble flow)
            catz['m.200c'] = []    # M_200c from unweighted fit of NFW M(< r) {M_sun}
            catz['c.200c'] = []    # concentration (200c) from unweighted fit of NFW M(< r)
            #catz['c.fof'] = []    # concentration derive from r_{2/3 mass} / r_{1/3 mass}
            catz['par.i'] = []    # index of parent, at previous snapshot, with max mass
            #catz['par.n.i'] = []    # index of next parent to same child, at same snapshot
            catz['chi.i'] = []    # index of child, at next snapshot
            #catz['m.fof.rat'] = []    # FoF mass ratio of two highest mass parents (< 1)
            #catz['cen.i'] = []     # index of central subhalo
        else:
            raise ValueError('catalog kind = %s not valid' % catalog_kind)
        self.directory_sim = self.treepm_directory + 'lcdm%d/' % box_length
        self.directory_tree = self.directory_sim + 'tree/'
        zis = ut.array.arrayize(zis)
        # read auxilliary data
        catz.Cosmo = self.read_cosmology()
        catz.info = {
            'kind': catalog_kind,
            'source': 'L%d' % box_length,
            'box.length.no-hubble': float(box_length),
            'box.length': float(box_length) / catz.Cosmo['hubble'],
            'particle.num': self.particle_num[box_length],
            'particle.m': catz.Cosmo.particle_mass(box_length, self.particle_num[box_length])
        }
        if catalog_kind == 'subhalo':
            catz.info['m.kind'] = 'm.max'
        elif catalog_kind == 'halo':
            catz.info['m.kind'] = 'm.fof'
        if cat_in is not None:
            cat = cat_in
        else:
            cat.Cosmo = catz.Cosmo
            cat.info = catz.info
            cat.snap = self.read_snapshot_time()
        # sanity check on snapshot range
        if zis.max() >= cat.snap['z'].size:
            zis = zis[zis < cat.snap['z'].size]
        for u_all in xrange(zis.max() + 1):
            if u_all >= len(cat):
                cat.append({})
        for zi in zis:
            cat[zi] = copy.copy(catz)
            cat[zi].snaps = cat.snap
            cat[zi].snap = {}
            for k in cat.snap.dtype.names:
                cat[zi].snap[k] = cat.snap[k][zi]
            cat[zi].snap['i'] = zi
            self.read_snapshot(cat[zi], zi)
        return cat

    def read_snapshot(self, catz, zi):
        '''
        Read single snapshot, assign to catalog.

        Import catalog of [sub]halo at snapshot, snapshot index.
        '''
        if catz.info['kind'] == 'subhalo':
            file_name_base = 'subhalo_tree_%s.dat' % str(zi).zfill(2)
            props = [
                'pos', 'vel', 'm.bound', 'vel.circ.max', 'm.max', 'vel.circ.peak', 'ilk', 'par.i',
                'par.n.i', 'chi.i', 'm.frac.min', 'm.max.rat.raw', 'inf.last.zi', 'inf.last.i',
                'inf.dif.zi', 'inf.dif.i', 'cen.i', 'dist.cen', 'sat.i', 'halo.i', 'halo.m'
            ]
        elif catz.info['kind'] == 'halo':
            file_name_base = 'halo_tree_%s.dat' % str(zi).zfill(2)
            props = [
                'pos', 'vel', 'm.fof', 'vel.circ.max', 'vel.disp', 'm.200c', 'c.200c', 'c.fof',
                'par.i', 'par.n.i', 'chi.i', 'm.fof.rat', 'cen.i'
            ]
            if catz.info['box.length.no-hubble'] == 720:
                props.remove('cen.i')
        file_name = self.directory_tree + file_name_base
        file_in = open(file_name, 'r')
        obj_num = int(np.fromfile(file_in, int32, 1))
        for prop in props:
            if len(prop) > 2 and (prop[-2:] == '.i' or prop[-3:] == '.zi' or prop == 'ilk'):
                dtype = int32
            else:
                dtype = float32
            self.read_property(file_in, catz, prop, dtype, obj_num)
            if prop == 'c.200c' and 'c.200c' in catz:
                catz['c.200c'][1:] = catz['c.200c'][1:].clip(1.5, 40)
        file_in.close()
        self.say('read %8d %s from %s' % (obj_num, catz.info['kind'], file_name_base))

    def read_property(self, file_in, cat, prop, dtype, obj_num):
        '''
        Read property from file, assign to [sub]halo catalog.

        Import input file, catalog at snapshot, property, data type, number of objects.
        '''
        if prop in ('pos', 'vel'):
            dimen_num = self.dimen_num
        else:
            dimen_num = 1
        temp = np.fromfile(file_in, dtype, obj_num * dimen_num)
        if prop in cat:
            if dimen_num > 1:
                cat[prop] = temp.reshape(obj_num, dimen_num)
            else:
                cat[prop] = temp
            if prop in ('pos', 'vel', 'dist.cen'):
                cat[prop] /= cat.Cosmo['hubble']

    def read_snapshot_time(self):
        '''
        Read time properties at snapshot, assign to simulation class/dictionary, return.
        '''
        file_name_base = 'snapshot.txt'
        file_name = self.directory_sim + file_name_base
        snaptime = np.loadtxt(file_name, comments='#', usecols=[1, 2, 3, 4, 5], dtype=[
            ('a', float32),
            ('z', float32),
            ('t', float32),
            ('t.wid', float32),
            ('t.hubble', float32)    # Hubble time = 1 / H(t) {Gyr}
        ])
        self.say('read ' + file_name)
        return snaptime

    def read_cosmology(self):
        '''
        Read cosmological parameters, save as class, return.
        '''
        Cosmo = cosmology.CosmologyClass(self.sigma_8)
        file_name_base = 'cosmology.txt'
        file_name = self.directory_sim + file_name_base
        file_in = open(file_name, 'r')
        for line in file_in:
            cin = np.array(line.split(), float32)
            if len(cin) == 7:
                Cosmo['omega_m'] = cin[0]
                Cosmo['omega_l'] = cin[1]
                Cosmo['w'] = cin[2]
                omega_b_0_h2 = cin[3]
                Cosmo['hubble'] = cin[4]
                Cosmo['n_s'] = cin[5]
                Cosmo['omega_b'] = omega_b_0_h2 / Cosmo['hubble'] ** 2
                break
            else:
                raise ValueError('%s not formatted correctly' % file_name_base)
        file_in.close()
        if (Cosmo['omega_m'] < 0 or Cosmo['omega_m'] > 0.5 or
            Cosmo['omega_l'] < 0.5 or Cosmo['omega_l'] > 1):
            self.say('! read strange cosmology in %s' % file_name_base)
        return Cosmo

    def pickle_first_infall(self, direction='read', sub=None, zis=np.arange(34), m_max_min=10.5):
        '''
        Read/write first infall times & subhalo indices at all snapshots in input range.

        Import pickle direction (read, write), subhalo catalog, properties for file name.
        '''
        inf_name = 'inf.first'
        file_name_short = 'subhalo_inf.first_m.max%.1f' % (m_max_min)
        file_name_base = (self.treepm_directory +
                          'lcdm%d/tree/' % sub.info['box.length.no-hubble'] + file_name_short)
        if direction == 'write':
            zis = np.arange(max(zis) + 1)
            siss = []
            inf_ziss = []
            inf_siss = []
            for zi in zis:
                subz = sub[zi]
                sis = ut.array.elements(subz[inf_name + '.zi'], [zi, Inf])
                siss.append(sis)
                inf_ziss.append(subz[inf_name + '.zi'][sis])
                inf_siss.append(subz[inf_name + '.i'][sis])
            ut.io.pickle_object(file_name_base, direction, [zis, siss, inf_ziss, inf_siss])
        elif direction == 'read':
            zis = ut.array.arrayize(zis)
            _zis_in, siss, inf_ziss, inf_siss = ut.io.pickle_object(file_name_base, direction)
            for zi in zis:
                subz = sub[zi]
                subz[inf_name + '.zi'] = np.zeros(subz['par.i'].size, int32) - 1
                subz[inf_name + '.zi'][siss[zi]] = inf_ziss[zi]
                subz[inf_name + '.i'] = ut.array.initialize_array(subz['par.i'].size)
                subz[inf_name + '.i'][siss[zi]] = inf_siss[zi]
        else:
            raise ValueError('not recognize i/o direction = %s' % direction)

Treepm = TreepmClass()
