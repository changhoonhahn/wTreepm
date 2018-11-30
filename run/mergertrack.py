#!/bin/python
'''
'''
import h5py 
import numpy as np 
from collections import Counter
from wtreepm import subhalo_io
from wtreepm import utility as wUT

"""
    def merger_track(mmid, dlogm=0.5): 
        ''' track the accreted halo masses 

        snap5   * * m3
                |/ 
        snap4   * 
                |
        snap3   * * m2
                |/ 
        snap2   * 
                | 
        snap1   * * m1 
                |/ 
        snap0   * M0 
        
        output [m1/M0, m2/M0, m3/M0...]  to file 

        '''
        snapshots = np.arange(35) 

        # read in treepm subhalos 
        sub = subhalo_io.Treepm.read('subhalo', zis=snapshots)
        nsub0 = len(sub[0]['ilk'])

        # central at snapshot 0 
        is_cen = (sub[0]['ilk'] == 1)
        print('%i out of %i subhalos are centrals' % (np.sum(is_cen), len(is_cen)))
        # within [mmid - 0.5 dlogm, mmid + 0.5 dlogm] at snapshot 0 
        in_mbin = (sub[0]['halo.m'] > mmid - 0.5*dlogm) & (sub[0]['halo.m'] < mmid + 0.5*dlogm)

        ihalos = np.arange(nsub0)[is_cen & in_mbin]
        m_M0 = [] 
        for isnap in range(snapshots[-1]): 
            # indices of the parent with the highest M_max (i.e. primaries) 
            i_primaries = sub[isnap]['par.i'][ihalos]
            # only keep halos that have parents
            # at snapshot=0 ~40 do not have parents
            has_primary = (i_primaries >= 0) 
            ihalos = ihalos[has_primary]
            i_primaries = i_primaries[has_primary]
            
            # make sure indices match across snapshot i and i+1 
            assert np.sum(ihalos - sub[isnap+1]['chi.i'][i_primaries]) == 0
            
            # identify halos with more than just a primary parent
            # the few lines below is just to speed things up a bit 
            counter = Counter(list(sub[isnap+1]['chi.i']))
            ip1_children_dupl = np.array([ii for ii, count in counter.items() if count >1])
            ihs = np.intersect1d(ihalos, np.array(ip1_children_dupl))
        
            for ih in ihs: 
                # loop through halos with parents and store non-primary halo masses 
                is_parent = np.where(sub[isnap+1]['chi.i'] == ih)[0]
                i_primary = sub[isnap]['par.i'][ih]
                notprimary = (is_parent != i_primary) 

                m_M0 += list(sub[isnap+1]['halo.m'][is_parent[notprimary]] - sub[isnap]['halo.m'][ih])
        
            print('snapshot %i, %i m_M0s' % (isnap, len(m_M0)))
            # keep searching through the primaries 
            ihalos = i_primaries
        print('%i halos have primaries in the last snapshot' % len(ihalos)) 
        m_M0 = np.array(m_M0)

        f = h5py.File('m_M0.h5', 'w') 
        f.create_dataset('m_M0', data=m_M0) 
        f.close() 
        return None 
"""

def merger_track_throughout(mmid, dlogm=0.5, m_type='m.200c'): 
    ''' track the accreted halo masses 

    snap5   * * m3
            |/ 
    snap4   * 
            |
    snap3   * * m2
            |/ 
    snap2   * 
            | 
    snap1   * * m1 
            |/ 
    snap0   * M0 
    
    output [m1/M0, m2/M0, m3/M0...]  to file 

    '''
    snapshots = np.arange(35) 

    # read in treepm **halos**
    hal = subhalo_io.Treepm.read('halo', zis=snapshots)
    nhalo0 = len(hal[0][m_type])
    
    # only keep halos within 
    # [mmid - 0.5 dlogm, mmid + 0.5 dlogm] at snapshot 0 
    in_mbin = (hal[0][m_type] > mmid - 0.5*dlogm) & (hal[0][m_type] < mmid + 0.5*dlogm)
    print('%i out of %i halos are within mass limit' % (np.sum(in_mbin), nhalo0))
    # and has prognitor until the last snapshot 
    prog_final = wUT.utility_catalog.indices_tree(hal, 0, snapshots[-1], np.arange(nhalo0))
    has_prog = (prog_final >= 0)
    i_halos = np.arange(nhalo0)[in_mbin & has_prog]
    print('%i out of %i halos are within the mass limit and tracked throughout the snapshots' % 
            (len(i_halos), nhalo0))
    halo_m0 = hal[0][m_type][i_halos] # halo mass at z = 0 

    m_M0s = [] 
    for isnap in range(snapshots[-1]): 
        print('--snapshot %i--' % isnap) 
        # find halos who have more than one parents 
        uniq_chi, n_chi = np.unique(hal[isnap+1]['chi.i'], return_counts=True) 
        has_chi = (n_chi > 1) & (uniq_chi != -1) 
        
        ih_wmerger, ii, _ = np.intersect1d(i_halos, uniq_chi[has_chi], return_indices=True) 
        
        # loop through them and get m/M0 of all secondary progenitors
        for ih, iii in zip(ih_wmerger, ii):   
            # all halos at isnap+1 that become the i_prog halos 
            sec_progs = (hal[isnap+1]['chi.i'] == ih) 
            if not sec_progs[hal[isnap]['par.i'][ih]]: 
                # primary progenitor should be one of the progenitors
                raise ValueError
            sec_progs[hal[isnap]['par.i'][ih]] = False # exclude primary progenitor 

            # halo mass of secondary progenitors
            m_sec = hal[isnap+1][m_type][sec_progs]
            # append m_sec / M0
            m_M0 = 10**(m_sec - halo_m0[iii])
            m_M0s += list(m_M0)

        i_halos = hal[isnap]['par.i'][i_halos] 
        if np.sum(i_halos < 0) > 0: 
            # all halos **should** have progenitors
            raise ValueError

    m_M0s = np.array(m_M0s)
    f = h5py.File('m_M0.h5', 'w') 
    f.create_dataset('m_M0', data=m_M0s) 
    f.close() 
    return None 


def merger_track_Mprimary(mmid, dlogm=0.5, nsnap_final=35): 
    ''' track the halo masses of the main branch
    '''
    snapshots = np.arange(nsnap_final) 

    # read in treepm subhalos 
    sub = subhalo_io.Treepm.read('subhalo', zis=snapshots)
    nsub0 = len(sub[0]['ilk'])

    # central at snapshot 0 
    is_cen = (sub[0]['ilk'] == 1)
    print('%i out of %i subhalos are centrals' % (np.sum(is_cen), nsub0))
    # within [mmid - 0.5 dlogm, mmid + 0.5 dlogm] at snapshot 0 
    in_mbin = (sub[0]['m.max'] > mmid - 0.5*dlogm) & (sub[0]['m.max'] < mmid + 0.5*dlogm)
    print('%i out of %i subhalos are centrals within mass limit' % (np.sum(is_cen & in_mbin), nsub0))
    # has prognitor until the last snapshot 
    prog_final = wUT.utility_catalog.indices_tree(sub, 0, snapshots[-1], np.arange(nsub0))
    has_prog = (prog_final >= 0)
    
    ihalos = np.arange(nsub0)[is_cen & in_mbin & has_prog]
    print('%i out of %i subhalos are tracked throughout the snapshots' % (len(ihalos), np.sum(is_cen & in_mbin)))
    
    mp_M0_dict = {} 
    for isnap in range(snapshots[-1]): 
        # indices of the parent with the highest M_max (i.e. primaries) 
        i_primaries = sub[isnap]['par.i'][ihalos]
        # only keep halos that have parents; at snapshot=0 ~40 do not have parents
        has_primary = (i_primaries >= 0) 
        ihalos = ihalos[has_primary]
        i_primaries = i_primaries[has_primary]
        
        Mp = sub[isnap+1]['m.max'][i_primaries] # log halo mass of primary halos 
        i_snap0 = wUT.utility_catalog.indices_tree(sub, isnap, 0, ihalos)
        Mp_M0 = Mp - sub[0]['m.max'][i_snap0]
        mp_M0_dict[isnap+1] = 10.**Mp_M0
        # keep searching through the primaries 
        ihalos = i_primaries
        
    print('%i halos have primaries in the last snapshot' % len(ihalos)) 

    f = h5py.File('m_Mprimary.nsnapf'+str(nsnap_final)+'.h5', 'w') 
    for k in mp_M0_dict.keys(): 
        f.create_dataset(str(k), data=mp_M0_dict[k]) 
    f.close() 
    return None 


def plotMzM0(nsnap_final): 
    ''' plot the 
    ''' 
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['axes.xmargin'] = 1
    mpl.rcParams['xtick.labelsize'] = 'x-large'
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.labelsize'] = 'x-large'
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['legend.frameon'] = False

    f = h5py.File('m_Mprimary.nsnapf'+str(nsnap_final)+'.h5', 'r')
    Mz_M0 = [1.] 
    for i in np.arange(1,nsnap_final): 
        Mz_M0.append(np.average(f[str(i)].value))
    tsnaps = np.array([13.8099 ,13.1328 ,12.4724 ,11.8271 ,11.1980 ,10.5893 , 9.9988 , 9.4289 , 8.8783 , 8.3525 , 7.8464 , 7.3635 , 6.9048 , 6.4665 , 6.0513 , 5.6597 , 5.2873 , 4.9378 , 4.6080 , 4.2980 , 4.0079 , 3.7343 , 3.4802 , 3.2408 , 3.0172 , 2.8078 , 2.6136 , 2.4315 , 2.2611 , 2.1035 , 1.9569 , 1.8198 , 1.6918 , 1.5726    , 1.4620]) 
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.plot(tsnaps[0]-tsnaps, Mz_M0)
    sub.legend(loc='upper right') 
    sub.set_xlabel(r'Look back Time', fontsize=25)
    sub.set_xlim([0, 13.]) 
    sub.set_ylabel(r'$M(z)/M_0$', fontsize=25)
    sub.set_ylim([0., 1.]) 
    fig.savefig('m_Mprimary.nsnapf'+str(nsnap_final)+'.png', bbox_inches='tight') 
    return None 


def plotMFaccreted(): 
    ''' plot the mass funciton
    ''' 
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['axes.xmargin'] = 1
    mpl.rcParams['xtick.labelsize'] = 'x-large'
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.labelsize'] = 'x-large'
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['legend.frameon'] = False

    m_M_stewart08 = np.array([0.009931501793687432, 0.013438729394117816, 0.018436209805733204, 0.024946804600076526, 0.033756562021813534, 0.045677412309913484, 0.06266353998445302, 0.08479265011700499, 0.1163246198130511,  0.1595824302693567,  0.212989507320644])
    n_M_stewart08 = np.array([7.690683028568886 , 6.347921959350477 , 5.0832369090372 , 4.111829402435831 , 3.22679911994581 , 2.481628922836825 , 1.908542144006686  , 1.3955224970987588 , 0.9604088212505378 , 0.6096600246370193 , 0.35338296804265507])

    f = h5py.File('m_M0.h5', 'r')
    m_M0 = f['m_M0'].value 
    
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    bins = np.logspace(-3., np.log10(0.3), 15) 
    sub.hist(m_M0, bins=bins, weights=np.repeat(1./7e4, len(m_M0)), histtype='step', cumulative=-1) #  
    sub.scatter(m_M_stewart08, n_M_stewart08, marker='x', s=20, label='Stewart+(2008)')
    sub.legend(loc='upper right') 
    sub.set_xlabel(r'$m/M_0$', fontsize=25)
    sub.set_xscale("log") 
    sub.set_xlim([1e-3, 0.3]) 
    sub.set_ylabel(r'$n(>m)$', fontsize=25)
    sub.set_yscale("log")
    #sub.set_ylim([0.2, 35]) 
    fig.savefig('m_M0.png', bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    #merger_track_throughout(12, dlogm=0.5)
    plotMFaccreted()
