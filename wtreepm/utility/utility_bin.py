'''
Utility functions for binning of array.

Author[s]: Andrew Wetzel.
'''

# system -----
from __future__ import division
import numpy as np
from numpy import log10, Inf
from scipy import ndimage, signal
# local -----
import utility_math as ut_math    # need to keep this way because of reverse import
from . import utility_array as ut_array
from . import utility_io as ut_io


#===================================================================================================
# binning
#===================================================================================================
def idigitize(vals, bin_mins, bin_max=None, round_kind='down', clip_to_bin_lim=False,
              scalarize=True, warn_outlier=True):
    '''
    Get bin indices of each value, using given rounding kind.
    Add extra bin to get outliers beyond bin_max & normalize so bin count starts at 0.
    If clip_to_bin_lim is false: if below min bin value, assign -1, if above max bin value,
    assign len(bin_mins).

    Import value[s], bin lower limits, bin upper limit, direction to round (up, down, near),
    whether to clip bin values to within input bin range (so all are counted),
    whether to convert to scalar if single bin,
    whether to warn if there are values beyond bin range.
    '''
    Say = ut_io.SayClass(idigitize)
    vals = ut_array.arrayize(vals)
    bin_mins = ut_array.arrayize(bin_mins)
    if bin_max is None:
        bin_max = 2 * bin_mins[-1] - bin_mins[-2]
    # add bin max to catch outliers & round properly
    bin_is = np.digitize(vals, np.append(bin_mins, bin_max)) - 1
    if warn_outlier:
        if bin_is.min() < 0 or bin_is.max() >= bin_mins.size:
            Say.say('! val lim = %s exceed bin lim = %s' %
                    (ut_array.get_limit(vals, digit_num=2),
                     ut_array.get_limit([bin_mins.min(), bin_max], digit_num=2)))
    if round_kind == 'up':
        bin_is[bin_is < bin_mins.size - 1] += 1
    elif round_kind == 'near':
        biis = ut_array.elements(bin_is, [0, bin_mins.size - 1.9])    # safely in bin limits
        biis_shift = biis[abs(bin_mins[bin_is[biis] + 1] - vals[biis]) <
                          abs(bin_mins[bin_is[biis]] - vals[biis])]
        # shift up if that is closer
        bin_is[biis_shift] += 1
    if clip_to_bin_lim:
        bin_is = bin_is.clip(0, bin_mins.size - 1)    # clip all values to within input bin range
    if scalarize and bin_is.size == 1:
        bin_is = bin_is[0]
    return bin_is


def filter_array(vals, filter_kind='triang', filter_size=3):
    '''
    Get array with smoothing filer applied.

    Import array of values, filter kind ('triang', 'boxcar'), filter size.
    '''
    window = signal.get_window(filter_kind, filter_size)
    window /= window.sum()
    return ndimage.convolve(vals, window)


class BinClass:
    '''
    Make & retrieve bin information for single array.
    '''
    def __init__(self, lim, wid=None, num=None, wid_kind='fix', vals=None, include_max=False):
        '''
        Assign bin information.

        Import value limits & bin width or number of bins, bin width kind (fix, vary),
        values to get bin widths if vary, whether to include limit maximum value in bin minima.
        '''
        if lim is None:
            self.lim = self.wid = self.wids = self.mins = self.mids = self.num = None
        elif wid_kind == 'fix':
            self.assign_bins_fix_width(lim, wid, num, include_max)
        elif wid_kind == 'vary':
            self.assign_bins_vary_width(vals, lim, num, include_max)
        else:
            raise ValueError('not recognize bin width kind = %s' % wid_kind)

    def get_bin_width(self, lim, wid, num):
        '''
        Import bin range, width, number of bins.
        If bin width is <= 0, use single bin across entire range.
        If limit is infinite, set width to infinity.
        If number defined, set width to give that number of bins.
        '''
        if num is None:
            if wid is None or wid <= 0:
                return lim[1] - lim[0]
            elif lim[0] == -Inf or lim[1] == Inf:
                return Inf
            else:
                return wid
        elif wid is None and num > 0:
            return (lim[1] - lim[0]) / num
        else:
            raise ValueError('bin wid = %s & num = %s, not sure how to parse' % (wid, num))

    def get_bin_num(self, lim, wid):
        '''
        Get interger number of values between min & max spaced by width.

        Import bin limits & width.
        '''
        return int(round(((lim[1] - lim[0]) / wid)))

    def assign_bins_fix_width(self, lim, wid, num, include_max=False):
        '''
        Import bin limits, number of bins, bin width,
        whether to include limit maximum in bin minnima.
        If limit is infinite, set width to infinity.
        If number defined, set width to give that number of bins.
        '''
        self.lim = lim
        self.wid = self.get_bin_width(lim, wid, num)
        self.mins = ut_array.arange_safe(self.lim, self.wid, include_max)
        if include_max:
            self.mids = self.mins[:-1] + 0.5 * self.wid
        else:
            if self.mins.size == 1 and np.isinf(self.mins):
                self.mids = np.abs(self.mins)
            else:
                self.mids = self.mins + 0.5 * self.wid
        self.num = self.mins.size
        self.wids = np.zeros(self.num) + self.wid

    def assign_bins_vary_width(self, vals, lim=[], num=30, include_max=False):
        '''
        Import value[s], limit to keep, number of bins, whether to include limit maximum in bin
        minima.
        If limit is infinite, set width to infinity.
        If number defined, set width to give that number of bins.
        '''
        self.num = num
        vals = np.array(vals)
        if lim:
            self.lim = lim
            vals = vals[vals >= lim[0]]
            if include_max:
                vals = vals[vals <= lim[1]]
            else:
                vals = vals[vals < lim[1]]
        else:
            # make sure bin goes just beyond value maximum
            lim_max = vals.max() * (1 + np.sign(vals.max()) * 1e-6)
            if lim_max == 0:
                lim_max += 1e-6
            self.lim = [vals.min(), lim_max]

        num_per_bin = int(vals.size / num)
        vals = np.sort(vals)
        self.mins, self.wids = np.zeros((2, num))

        # assign bin minima
        self.mins[0] = lim[0]
        for bini in xrange(1, num):
            # place bin minimum value half-way between adjacent points
            # unless equal, then place just above
            if vals[bini * num_per_bin - 1] == vals[bini * num_per_bin]:
                self.mins[bini] = vals[bini * num_per_bin] * (1 + 1e-5)
            else:
                self.mins[bini] = 0.5 * (vals[bini * num_per_bin - 1] + vals[bini * num_per_bin])
        # assign bin widths
        bin_is = np.arange(num - 1)
        self.wids[bin_is] = self.mins[bin_is + 1] - self.mins[bin_is]
        self.wids[-1] = self.lim[1] - self.mins[-1]

        # assign bin centers
        if include_max:
            self.mids = self.mins[:-1] + 0.5 * self.wids
        else:
            self.mids = self.mins + 0.5 * self.wids

    def get_bin_limit(self, ibin):
        '''
        Get limit of single bin.

        Import index of bin.
        '''
        return [self.mins[ibin], self.mins[ibin] + self.wids[ibin]]

    def binize(self, vals, round_kind='down', clip_to_bin_lim=False, scalarize=True,
               warn_outlier=True):
        '''
        Get bin indices of each value, using given rounding kind.

        Import value[s], direction to round (up, down, near),
        whether to clip bin values to input bin range (so all are counted),
        whether to convert to scalar if single bin,
        whether to warn if input values beyond bin range.
        '''
        return idigitize(vals, self.mins, self.lim[1], round_kind, clip_to_bin_lim, scalarize,
                         warn_outlier)

    def get_stat_dict_y(self, vals_x, vals_y):
        '''
        Get values & stats of y-array in each x-bin.
        Ignore values outside of bin range.

        Import x-array, y-array.
        '''
        stat = {'x': [], 'ys': [], 'indices': [], 'number': []}
        Stat = ut_math.StatClass()
        vals_y, vals_x = np.array(vals_y), np.array(vals_x)
        if vals_y.size != vals_x.size:
            raise ValueError('x-array size = %d but y-array size = %d' % (vals_x.size, vals_y.size))
        vis = ut_array.arange_length(vals_x)
        bin_is = self.binize(vals_x)
        stat['x'] = self.mids
        for bi in xrange(self.num):
            vals_y_in_bin = vals_y[bin_is == bi]
            stat['ys'].append(vals_y_in_bin)
            stat['indices'].append(vis[bin_is == bi])
            stat['number'].append(vals_y_in_bin.size)
            # if not vals_y_in_bin.size:
            #    for k in stat:
            #        if k not in ('x', 'ys', 'num'):
            #            stat[k].append(-1)
            # else:
            Stat.append_dictionary(vals_y_in_bin)
        stat.update(Stat.stat)
        for k in stat:
            if k != 'ys':
                stat[k] = np.array(stat[k])
        return stat


class MMbinClass(BinClass, ut_io.SayClass):
    '''
    Make & retrieve mass bin information for both galaxies/subhalos & halos.
    '''
    def __init__(self, g_kind, g_lim, g_wid, h_kind, h_lim, h_wid, vary_kind, include_max=False):
        '''
        Assign galaxy & halo mass bin information.

        Import subhalo/galaxy [m] kind & range & bin width, halo mass kind & range & bin width,
        which to vary (halo, galaxy), whether to include limiit max in bin values.
        '''
        self.gal = BinClass(g_lim, g_wid, include_max=include_max)
        self.hal = BinClass(h_lim, h_wid, include_max=include_max)
        self.gal.kind = g_kind
        self.hal.kind = h_kind
        if vary_kind in (h_kind, 'halo'):
            varym_kind = h_kind
            vary_kind = 'halo'
            fixm_kind = g_kind
            fixkind = 'galaxy'
        elif vary_kind in (g_kind, 'galaxy'):
            varym_kind = g_kind
            vary_kind = 'galaxy'
            fixm_kind = h_kind
            fixkind = 'halo'
        else:
            raise ValueError('not recognize vary_kind = %s' % vary_kind)
        if vary_kind == 'halo':
            self.vary = self.hal
            self.fix = self.gal
        elif vary_kind == 'galaxy':
            self.vary = self.gal
            self.fix = self.hal
        self.vary.m_kind = varym_kind
        self.vary.kind = vary_kind
        self.fix.m_kind = fixm_kind
        self.fix.kind = fixkind
        # copy varying parameters to self for shortcut
        self.wid = self.vary.wid
        self.wids = self.vary.wids
        self.lim = self.vary.lim
        self.mins = self.vary.mins
        self.mids = self.vary.mids
        self.num = self.vary.num

    def bins_limit(self, vary_i, fix_i=None, printm=False):
        '''
        Get limit of single vary & fix bin.

        Import vary value or index, fix value or index, whether to print mass bin.
        '''
        vary_lim = self.vary.get_bin_limit(vary_i)
        if fix_i is not None:
            fix_lim = self.fix.get_bin_limit(fix_i)
        else:
            fix_lim = self.fix.lim
        if printm:
            self.Say('%s [%.2f, %.2f]' % (self.vary.kind, vary_lim[0], vary_lim[1]))
            if fix_i is not None:
                self.Say('%s [%.2f, %.2f]' % (self.fix.kind, vary_lim[0], vary_lim[1]))
        if self.vary.kind == self.hal.kind:
            glim = fix_lim
            hlim = vary_lim
        elif self.vary.kind == self.gal.kind:
            glim = vary_lim
            hlim = fix_lim
        return glim, hlim


class RbinClass(BinClass, ut_io.SayClass):
    '''
    Radius / distance bin information.
    '''
    def __init__(self, scaling, lim, num=None, wid=None, dimen_num=3, include_max=False):
        '''
        Assign radius / distance bins, of fixed width in scaling units.

        Import bin scaling (lin, log), *linear* limits, number of bins *or* bin width
        (in scaling units), number of spatial dimensions,
        whether to inclue limits maximum in bin minima.
        '''
        self.scaling = scaling
        self.dimen_num = dimen_num
        self.lim = np.array(lim)
        self.log_lim = log10(self.lim)
        if 'log' in scaling:
            self.log_wid = self.get_bin_width(self.log_lim, wid, num)
            self.log_mins = ut_array.arange_safe(self.log_lim, self.log_wid, include_max)
            self.log_mids = self.log_mins + 0.5 * self.log_wid
            self.log_wids = np.zeros(self.log_mins.size) + self.log_wid
            self.mids = 10 ** self.log_mids
            self.mins = 10 ** self.log_mins
            self.wids = self.mins * (10 ** self.log_wid - 1)
        elif 'lin' in scaling:
            self.wid = self.get_bin_width(self.lim, wid, num)
            self.mins = ut_array.arange_safe(self.lim, self.wid, include_max)
            self.mids = self.mins + 0.5 * self.wid
            self.wids = np.zeros(self.mins.size) + self.wid
            self.log_mids = log10(self.mids)
            self.log_mins = log10(self.mins)
            self.log_wids = log10(self.wid / self.mins + 1)
        else:
            raise ValueError('not recognize scaling = %s' % scaling)
        if dimen_num > 0:
            if dimen_num == 1:
                self.vol_norm = 1
            elif dimen_num == 2:
                self.vol_norm = np.pi
            elif dimen_num == 3:
                self.vol_norm = 4 / 3 * np.pi
            self.vol_in_lim = self.vol_norm * (lim[1] ** dimen_num - lim[0] ** dimen_num)
            self.vols = self.vol_norm * ((self.mins + self.wids) ** dimen_num -
                                         self.mins ** dimen_num)
            self.vol_fracs = self.vols / self.vol_in_lim
        self.num = self.mins.size

    def get_bin_limit(self, scaling, ibin):
        '''
        Get distance limits (lin, log) of single bin.

        Import distance scaling (lin, log), bin index.
        '''
        if 'lin' in scaling:
            return [self.mins[ibin], (self.mins[ibin] + self.wids[ibin])]
        elif 'log' in scaling:
            return [self.log_mins[ibin], (self.log_mins[ibin] + self.log_wids[ibin])]
        else:
            raise ValueError('not recognize scaling = %s' % scaling)

    def get_profile(self, dists, host_num=None, normalize_lim=[]):
        '''
        Get dictionary of number & number density v distance.

        Import *linear* distances, number of host halos to normalize counts to,
        distance limit to normalized counts to.
        '''
        pro = ut_array.DictClass()
        if 'lin' in self.scaling:
            pro['dist'] = self.mids
        elif 'log' in self.scaling:
            pro['dist'] = self.log_mids
        # get numbers in bins
        if 'lin' in self.scaling:
            pro['num'] = np.histogram(dists, self.num, self.lim, False)[0]
        elif 'log' in self.scaling:
            pro['num'] = np.histogram(log10(dists), self.num, self.log_lim, False)[0]
        if normalize_lim:
            if 'lin' in self.scaling:
                wid = self.wid
            elif 'log' in self.scaling:
                wid = self.log_wid
            Rbin_normalize = RbinClass(self.scaling, self.lim, None, wid, self.dimen_num)
            num_in_lim = ut_array.elements(dists, normalize_lim).size
            vol_in_lim = Rbin_normalize.vol_in_lim
        else:
            num_in_lim = np.sum(pro['num'])
            vol_in_lim = self.vol_in_lim
        nonzeros = pro['num'] > 0
        pro['log-num'] = np.zeros(pro['num'].size) + np.nan
        pro['log-num'][nonzeros] = log10(pro['num'][nonzeros])
        pro['num.err'] = pro['num'] ** 0.5
        # get number densities in bins
        pro['den'] = pro['num'] / self.vols
        pro['log-den'] = np.zeros(pro['num'].size) + np.nan
        pro['log-den'][nonzeros] = log10(pro['den'][nonzeros])
        pro['den.err'] = pro['num.err'] / self.vols
        # get differential probability in bin (normalized to number within distance limits)
        pro['prob'] = (pro['num'] / self.vols) / (num_in_lim / vol_in_lim)
        pro['log-prob'] = np.zeros(pro['num'].size) + np.nan
        pro['log-prob'][nonzeros] = log10(pro['prob'][nonzeros])
        pro['prob.err'] = (pro['num.err'] / self.vols) / (num_in_lim / vol_in_lim)
        if host_num is not None:
            # get number per volume/area element (normalized to 1) per host
            pro['den-per-host'] = pro['num'] / host_num / self.vols
            pro['log-den-per-host'] = np.zeros(pro['num'].size) + np.nan
            pro['log-den-per-host'][nonzeros] = log10(pro['den-per-host'][nonzeros])
            pro['den-per-host.err'] = pro['num.err'] / host_num / self.vols
        self.say('input %d distances, of which %s are withim limits = %s' %
                 (len(dists), num_in_lim, ut_array.get_limit(self.lim, digit_num=4)))
        return pro


def get_indices_match_distr(vals_ref, vals_select, lim, bin_wid=None, bin_num=None):
    '''
    Get indices to sample from vals_select to give same relative distribution as in vals_ref.

    Import reference values, values to sample from, value limit & bin width or number of bins.
    '''
    Bin = BinClass(lim, bin_wid, bin_num)
    bin_is_ref = Bin.binize(vals_ref)
    bin_is_select = Bin.binize(vals_select)
    num_in_bins_ref = np.zeros(Bin.num)
    num_in_bins_select = np.zeros(Bin.num)
    for bi in xrange(Bin.num):
        num_in_bins_ref[bi] = np.sum(bin_is_ref == bi)
        num_in_bins_select[bi] = np.sum(bin_is_select == bi)
    frac_in_bins_ref = num_in_bins_ref / vals_ref.size
    ibin_mode = np.argmax(num_in_bins_ref)
    frac_in_bins_keep = frac_in_bins_ref / frac_in_bins_ref[ibin_mode]
    num_in_bins_keep = np.round(frac_in_bins_keep * num_in_bins_select[ibin_mode])
    vis_select = ut_array.arange_length(vals_select)
    vis_keep = []
    for bi in xrange(Bin.num):
        vis_bin = vis_select[bi == bin_is_select]
        if bi == ibin_mode:
            vis_keep.extend(vis_bin)
        else:
            vis_keep.extend(ut_array.sample_array(vis_bin, num_in_bins_keep[bi]))
    return np.array(vis_keep, vis_select.dtype)


def get_indices_sample_distr(vals, val_lim=[10, 14], val_wid=0.5, num_in_bin=5):
    '''
    Get indices that randomly sample value array with equal number in each bin.

    Import array of values, imposed limit, bin width, number to keep in each mass bin.
    '''
    vis = []
    Bin = BinClass(val_lim, val_wid)
    for bin_i in xrange(Bin.num):
        vis_bin = ut_array.elements(vals, Bin.get_bin_limit(bin_i))
        if len(vis_bin) < num_in_bin:
            num_in_bin_use = len(vis_bin)
        else:
            num_in_bin_use = num_in_bin
        vis.extend(ut_array.sample_array(vis_bin, num_in_bin_use))
    return np.array(vis, np.int32)
