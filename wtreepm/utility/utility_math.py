'''
Utility functions for math, function fitting.

Author[s]: Andrew Wetzel.
'''

# system -----
from __future__ import division
import numpy as np
from numpy import log10
from scipy import integrate, interpolate, ndimage, special, stats
# local -----
from .. import constants as const
import utility_bin as ut_bin    # need to keep this way because of reverse import
from . import utility_array as ut_array
from . import utility_io as ut_io
from . import utility_nmpfit as ut_nmpfit


#===================================================================================================
# fraction
#===================================================================================================
class FractionClass(dict, ut_io.SayClass):
    '''
    Compute fraction safely, convert from fraction to ratio, store fractions in self dictionary.
    '''
    def __init__(self, nums=None, err_kind='normal', dtype=np.float64):
        '''
        Initialize dictionary to store fraction values & uncertainties.

        Import number of values (int or list), fraction uncertainty kind (normal, beta).
        '''
        if nums is not None and len(nums):
            nums = ut_array.arrayize(nums)
            self.err_kind = err_kind
            if err_kind == 'normal':
                self['err'] = np.zeros(nums, dtype)
            elif err_kind == 'beta':
                self['err'] = np.zeros(np.append(nums, 2), dtype)
            else:
                raise ValueError('not recognize error kind: %s' % err_kind)
            self['val'] = np.zeros(nums, dtype)
            self['numer'] = np.zeros(nums, dtype)
            self['denom'] = np.zeros(nums, dtype)

    def fraction(self, nums, denoms, err_kind=''):
        '''
        Get m/num [& uncertainty if uncertainty kind defined].
        Assume m < n, & that m = 0 if n = 0.

        Import subset count[s], total count[s], uncertainty kind ('', normal, beta).
        '''
        if np.isscalar(nums):
            if nums == 0 and denoms == 0:
                # avoid dividing by 0
                if not err_kind:
                    return 0.0
                elif err_kind == 'normal':
                    return 0.0, 0.0
                elif err_kind == 'beta':
                    return 0.0, np.array([0.0, 0.0])
                else:
                    raise ValueError('not recognize uncertainty kind = %s' % err_kind)
            elif denoms == 0:
                raise ValueError('nums != 0, but denoms = 0')
        else:
            nums = np.array(nums)
            denoms = np.array(denoms).clip(1e-20)
        fracs = nums / denoms
        if err_kind:
            if err_kind == 'normal':
                fracs_err = ((nums / denoms * (1 - nums / denoms)) / denoms) ** 0.5
            elif err_kind == 'beta':
                # Cameron 2011
                conf_inter = 0.683    # 1 - sigma
                p_lo = (nums / denoms - stats.distributions.beta.ppf(0.5 * (1 - conf_inter),
                                                                     nums + 1, denoms - nums + 1))
                p_hi = stats.distributions.beta.ppf(
                    1 - 0.5 * (1 - conf_inter), nums + 1, denoms - nums + 1) - nums / denoms
                fracs_err = np.array([p_lo, p_hi]).clip(0)
            else:
                raise ValueError('not recognize uncertainty kind = %s' % err_kind)
            return fracs, fracs_err
        else:
            return fracs

    def fraction_volmax(self, volmaxs_sub, volmaxs_tot, err_kind='', jack_is_sub=None,
                        jack_is_tot=None):
        '''
        Get fraction, weighted by V_max, & use object counts to get error.

        Import subset V_maxs, all V_maxs, uncertainty kind ('', normal, beta).
        '''
        if volmaxs_sub.size == 0:
            return self.fraction(0, 0, err_kind)
        if np.min(volmaxs_sub) == 0 or np.min(volmaxs_tot) == 0:
            self.say('! V_max = 0')
        if err_kind == 'jackknife':
            if jack_is_sub is None or jack_is_tot is None:
                raise ValueError('no input jackknife bins')
            frac = self.fraction(np.sum(1 / volmaxs_sub), np.sum(1 / volmaxs_tot), '')
            fracs = []
            for jacki in xrange(jack_is_tot.max() + 1):
                volmaxs_sub_j = volmaxs_sub[jack_is_sub == jacki]
                volmaxs_tot_j = volmaxs_tot[jack_is_tot == jacki]
                if volmaxs_tot_j.size:
                    fracs.append(self.fraction(np.sum(1. / volmaxs_sub_j),
                                               np.sum(1. / volmaxs_tot_j), ''))
            fracs = np.array(fracs)
            #frac_err = ((fracs.size - 1) / fracs.size * np.sum((fracs - frac) ** 2)) ** 0.5
            frac_err = (1 / (fracs.size - 1) * np.sum((fracs - frac) ** 2)) ** 0.5
            return frac, frac_err
        else:
            frac = self.fraction(np.sum(1. / volmaxs_sub), np.sum(1. / volmaxs_tot), '')
            return self.fraction(frac * volmaxs_tot.size, volmaxs_tot.size, err_kind)

    def fraction_from_ratio(self, rat):
        '''
        Get x fraction of total, x / (x + y).

        Import ratio x / y.
        '''
        return 1 / (1 + 1 / rat)

    def ratio_from_fraction(self, frac):
        '''
        Get ratio x / y.

        Import fraction of total, x / (x + y).
        '''
        return frac / (1 - frac)

    def add_to_dict(self, iis, numer, denom):
        '''
        Get numer/denom & uncertainty.
        Assumes numer < denom, & that numer = 0 if denom = 0.

        Import index[s] to assign to, subset & total object count.
        '''
        if np.ndim(iis):
            iis = tuple(iis)
        self['val'][iis], self['err'][iis] = self.fraction(numer, denom, self.err_kind)
        self['numer'][iis] = numer
        self['denom'][iis] = denom

Frac = FractionClass()


#===================================================================================================
# statistics
#===================================================================================================
class StatClass(ut_io.SayClass):
    '''
    Store statistics & probability distribution of input array.
    '''
    def __init__(self, vals=None, lim=None, bin_num=10, bin_wid_kind='fix'):
        '''
        Import values, range to keep, number of bins, bin width kind (fix, vary).
        '''
        if vals is not None and len(vals):
            self.stat, self.prob = self.stat_prob_dict(vals, lim, bin_num, bin_wid_kind)
        else:
            self.stat = {}
            self.prob = {}

    def stat_dict(self, vals, lim=None):
        '''
        Get dicionaries for statistics within range.

        Import values, range to keep.
        '''
        stat = {
            'lim': [],
            'num': 0,
            'med': 0, 'med.16': 0, 'med.84': 0, 'med.02': 0, 'med.98': 0,    # values at confidence
            'med.err.02': 0, 'med.err.16': 0, 'med.err.84': 0, 'med.err.98': 0,
            'med.err': [0, 0],
            'ave': 0, 'std': 0, 'sem': 0,    # average, std dev, std dev of mean
            'std.lo': 0, 'std.hi': 0,    # values of std limits
            'min': 0, 'max': 0,    # minimum & maximum
            #'min.2': 0, 'min.3': 0, 'max.2': 0, 'max.3': 0,    # list of nth minimums & maximums

        }
        if vals is None or not len(vals):
            return stat
        vals = np.array(vals)
        if lim is not None and len(lim):
            if lim[0] == lim[1] or lim[1] == vals.max():
                lim[1] *= 1 + 1e-6    # make sure single value remains valid
                lim[1] += 1e-6
            vals = vals[ut_array.elements(vals, lim)]
        else:
            lim = ut_array.get_limit(vals)
            if lim[0] == lim[1]:
                lim[1] *= 1 + 1e-6    # make sure single value remains valid
                lim[1] += 1e-6
            if isinstance(lim[1], int):
                lim[1] *= 1 + 1e-6
        if not vals.size:
            self.say('! no values within bin limit: %s' % lim)
            return stat
        # scalar statistics
        stat['lim'] = lim
        stat['num'] = vals.size
        stat['med'] = np.median(vals)
        stat['med.16'] = stats.scoreatpercentile(vals, 16)
        stat['med.84'] = stats.scoreatpercentile(vals, 84)
        stat['med.02'] = stats.scoreatpercentile(vals, 2.275)
        stat['med.98'] = stats.scoreatpercentile(vals, 97.725)
        stat['med.err.02'] = stat['med'] - stat['med.02']
        stat['med.err.16'] = stat['med'] - stat['med.16']
        stat['med.err.84'] = stat['med.84'] - stat['med']
        stat['med.err.98'] = stat['med.98'] - stat['med']
        stat['med.err'] = [stat['med.err.16'], stat['med.err.84']]
        stat['ave'] = vals.mean()
        stat['std'] = vals.std(ddof=1)
        stat['sem'] = stats.sem(vals)
        stat['std.lo'] = stat['ave'] - stat['std']
        stat['std.hi'] = stat['ave'] + stat['std']
        stat['sem.lo'] = stat['ave'] - stat['sem']
        stat['sem.hi'] = stat['ave'] + stat['sem']
        stat['min'] = vals.min()
        stat['max'] = vals.max()
        """
        # make sure array has more than one value
        if stat['max'] != stat['min']:
            vals_sort = np.unique(vals)
            if vals_sort.size > 2:
                stat['min.2'] = vals_sort[1]
                stat['max.2'] = vals_sort[-2]
            if vals_sort.size > 4:
                stat['min.3'] = vals_sort[2]
                stat['max.3'] = vals_sort[-3]
        """
        return stat

    def prob_dict(self, vals, lim=None, bin_num=10, bin_wid_kind='fix'):
        '''
        Get dicionary for histogram/probability distribution, using number of input bins.

        Import values, limit to keep, number of bins, bin width kind (fix, vary).
        '''
        prob = {
            'lim': [],
            'val': np.array([]),
            'prob': np.array([]), 'prob.err': np.array([]), 'prob.cum': np.array([]),
            'num': np.array([]), 'num.err': np.array([]), 'log-num': np.array([])
        }
        if vals is None or not len(vals):
            return prob
        vals = np.array(vals)
        if lim is None or not len(lim):
            lim = ut_array.get_limit(vals)
            if lim[0] == lim[1]:
                lim[1] *= 1 + 1e-6    # make sure single value remains valid
                lim[1] += 1e-6
            if isinstance(lim[1], int):
                lim[1] *= 1 + 1e-6
        prob['lim'] = lim
        vis = ut_array.elements(vals, lim)
        if not vis.size:
            self.say('! no values within bin limit: %s' % lim)
            return prob
        vals_lim = vals[vis]
        if bin_num > 0:
            Bin = ut_bin.BinClass(lim=lim, num=bin_num, wid_kind=bin_wid_kind, vals=vals)
            prob['val'] = Bin.mids
            if bin_wid_kind == 'fix':
                prob['num'] = np.histogram(vals_lim, bin_num, lim, False)[0]
                prob['prob'] = np.histogram(vals_lim, bin_num, lim, True)[0]
            elif bin_wid_kind == 'vary':
                bin_mins = np.append(Bin.mins, Bin.lim[1])
                prob['num'] = np.histogram(vals_lim, bin_mins, lim, False)[0]
                prob['prob'] = prob['num'] / vals_lim.size / Bin.wids
            else:
                raise ValueError('not recognize bin width kind = %s' % bin_wid_kind)
            prob['log-num'] = log10(prob['num'] + 1e-10)
            val_lo_num = vals[vals < lim[0]].size
            prob['num.err'] = prob['num'] ** 0.5
            prob['prob.err'] = Frac.fraction(prob['prob'], prob['num.err'])
            prob['prob.cum'] = (np.cumsum(prob['num']) + val_lo_num) / vals.size
        return prob

    def stat_prob_dict(self, vals, lim=None, bin_num=10, bin_wid_kind='fix'):
        '''
        Get dicionaries for statistics & histogram/probability distribution, using number of input
        bins.

        Import values, limit to keep, number of bins, bin width kind (fix, vary).
        '''
        return self.stat_dict(vals, lim), self.prob_dict(vals, lim, bin_num, bin_wid_kind)

    def append_dictionary(self, vals, lim=None, bin_num=10, bin_wid_kind='fix'):
        '''
        Make dictionaries for statistics & histogram/probability distribution, append to self.

        Import values, range to keep, number of bins, bin width kind (fix, vary).
        '''
        # check if need to arrayize dictionaries
        if (self.prob and self.prob['prob'] and len(self.prob['prob']) and
            np.isscalar(self.prob['prob'][0])):
            for k in self.stat:
                self.stat[k] = [self.stat[k]]
            for k in self.prob:
                self.prob[k] = [self.prob[k]]
        stat, prob = self.stat_prob_dict(vals, lim, bin_num, bin_wid_kind)
        ut_array.append_dictionary(self.stat, stat)
        ut_array.append_dictionary(self.prob, prob)

    def append_dicts(self, StatIn):
        '''
        Append stat class dictionaries to self.

        Import another stat class.
        '''
        ut_array.append_dictionary(self.stat, StatIn.stat)
        ut_array.append_dictionary(self.prob, StatIn.prob)

    def arrayize(self):
        '''
        .
        '''
        self.stat = ut_array.arrayize(self.stat)
        self.prob = ut_array.arrayize(self.prob)

    def print_stat(self, i=None):
        '''
        Print statistics.

        Import bin index to get statistic of.
        '''
        stat_list = ['med', 'ave', 'std', 'med.16', 'med.84', 'med.02', 'med.98', 'min', 'max']
        #, 'min.2', 'min.3', 'max.2', 'max.3']
        if i is None and not np.isscalar(self.stat['med']):
            raise ValueError('no input index, but stat is multi-dimensional')
        if i is not None:
            val = self.stat['num'][i]
        else:
            val = self.stat['num']
        self.say('num = %d' % val)
        for k in stat_list:
            if i is not None:
                val = self.stat[k][i]
            else:
                val = self.stat[k]
            self.say('%s = %.4f' % (k, val))


def print_array_dif_stats(vals_1, vals_2):
    '''
    Print statistics of their absolute differences.

    Import two arrays.
    '''
    diffs = abs(vals_1 - vals_2)
    Stat = StatClass(diffs)
    Stat.print_stat()


def deconvolute(y_conv, scatter, x_wid, iter_num=10):
    '''
    Get deconvolved version via Lucy routine.

    Import gaussian convoluted function, scatter, bin width, number of iterations.
    '''
    yit = y_conv
    for _ in xrange(iter_num):
        ratio = y_conv / ndimage.filters.gaussian_filter1d(yit, scatter / x_wid)
        yit = yit * ndimage.filters.gaussian_filter1d(ratio, scatter / x_wid)
        # this is part of lucy's routine, but seems less stable
        #yit = yit * ratio
    return yit


def get_sample_variance(vals, poss, vol_num_per_dimen=2):
    '''
    Split volume by median position along each dimension.
    Get standard deviation from sample variance across volume bins.
    !!! NEED TO THINK MORE ABOUT HOW TO SUBDIVIDE VOLUME.

    Import values, positions (1, 2, 3D), number of volume bins per dimension,
    value bin range & number for histogram.
     '''
    scop = stats.scoreatpercentile
    if np.ndim(poss) == 1:
        dimen_num = 1
    else:
        dimen_num = poss.shape[1]
    if dimen_num < 1 or dimen_num > 3:
        raise ValueError('dimen num = %d' % dimen_num)
    bin_is = np.arange(vol_num_per_dimen)
    bin_sps_lo = 100 * (bin_is / vol_num_per_dimen)
    bin_sps_hi = 100 * ((bin_is + 1) / vol_num_per_dimen)
    vol_bin_stat = {'ave': [], 'med': [], 'num': []}
    pis_test = []
    if dimen_num == 1:
        for bi in bin_is:
            pos_lim_bin0 = [scop(poss, bin_sps_lo[bi]), scop(poss, bin_sps_hi[bi])]
            pis = ut_array.elements(poss[0], pos_lim_bin0)
            vol_bin_stat['ave'].append(np.mean(vals[pis]))
            vol_bin_stat['med'].append(np.median(vals[pis]))
    elif dimen_num == 2:
        for b0i in bin_is:
            pos_lim_bin0 = [scop(poss[:, 0], bin_sps_lo[b0i]), scop(poss[:, 0], bin_sps_hi[b0i])]
            pis0 = ut_array.elements(poss[:, 0], pos_lim_bin0)
            for b1i in bin_is:
                pos_lim_bin1 = [scop(poss[:, 1], bin_sps_lo[b1i]),
                                scop(poss[:, 1], bin_sps_hi[b1i])]
                pis1 = ut_array.elements(poss[:, 1], pos_lim_bin1, pis0)
                pis_test.extend(pis1)
                vol_bin_stat['ave'].append(np.mean(vals[pis1]))
                vol_bin_stat['med'].append(np.median(vals[pis1]))
    elif dimen_num == 3:
        for b0i in bin_is:
            pos_lim_bin0 = [scop(poss[:, 0], bin_sps_lo[b0i]), scop(poss[:, 0], bin_sps_hi[b0i])]
            pis0 = ut_array.elements(poss[:, 0], pos_lim_bin0)
            for b1i in bin_is:
                pos_lim_bin1 = [scop(poss[:, 1], bin_sps_lo[b1i]),
                                scop(poss[:, 1], bin_sps_hi[b1i])]
                pis1 = ut_array.elements(poss[:, 1], pos_lim_bin1, pis0)
                for b2i in bin_is:
                    pos_lim_bin2 = [scop(poss[:, 2], bin_sps_lo[b2i]),
                                    scop(poss[:, 2], bin_sps_hi[b2i])]
                    pis2 = ut_array.elements(poss[:, 2], pos_lim_bin2, pis1)
                    pis_test.extend(pis2)
                    vol_bin_stat['ave'].append(np.mean(vals[pis2]))
                    vol_bin_stat['med'].append(np.median(vals[pis2]))
    if np.unique(pis_test).size < len(pis_test):
        raise ValueError('values overlap in volume bins')
    vol_bin_stat['ave'] = np.std(vol_bin_stat['ave'], ddof=1)
    vol_bin_stat['med'] = np.std(vol_bin_stat['med'], ddof=1)
    return vol_bin_stat


#===================================================================================================
# spline fitting
#===================================================================================================
class SplineFunctionClass(ut_io.SayClass):
    '''
    Fit spline [& its inverse] to function, given input interval.
    '''
    def __init__(self, func, x_lim=[0, 1], num=100, dtype=np.float64, make_inverse=True,
                 **kwargs):
        '''
        Fit y(x) to spline, & fit x(y) if y is monotonic.

        Import function y(x), x range & number of spline points, data type to store,
        whether to make inverse spline.
        '''
        self.dtype = dtype
        self.xs = np.linspace(x_lim[0], x_lim[1], num).astype(dtype)
        self.ys = np.zeros(num, dtype)
        for xi in xrange(self.xs.size):
            self.ys[xi] = func(self.xs[xi], **kwargs)
        self.spline_y_from_x = interpolate.splrep(self.xs, self.ys)
        if make_inverse:
            self.make_spline_inverse()

    def make_spline_inverse(self):
        xs_temp = self.xs
        ys_temp = self.ys
        if ys_temp[1] < ys_temp[0]:
            ys_temp = ys_temp[::-1]
            xs_temp = xs_temp[::-1]
        yis = ut_array.arange_length(ys_temp.size - 1)
        if (ys_temp[yis] < ys_temp[yis + 1]).min():
            self.spline_x_from_y = interpolate.splrep(ys_temp, xs_temp)
        else:
            self.say('! unable to make inverse spline: y-values not monotonic')

    def val(self, x, ext=2):
        return interpolate.splev(x, self.spline_y_from_x, ext=ext).astype(self.dtype)

    def deriv(self, x, ext=2):
        return interpolate.splev(x, self.spline_y_from_x, der=1, ext=ext).astype(self.dtype)

    def val_inv(self, y, ext=2):
        return interpolate.splev(y, self.spline_x_from_y, ext=ext).astype(self.dtype)

    def deriv_inv(self, y, ext=2):
        return interpolate.splev(y, self.spline_x_from_y, der=1, ext=ext).astype(self.dtype)


class SplinePointClass(SplineFunctionClass):
    '''
    Fit spline [& its inverse] to input points.
    '''
    def __init__(self, xs, ys, dtype=np.float64, make_inverse=True):
        '''
        Fit y(x) to spline, & fit x(y) if y is monotonic.

        Import x & y points, data type to store, whether to make inverse spline.
        '''
        self.Say = ut_io.SayClass(SplineFunctionClass)
        self.dtype = dtype
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.spline_y_from_x = interpolate.splrep(self.xs, self.ys)
        if make_inverse:
            self.make_spline_inverse()


#===================================================================================================
# function fitting
#===================================================================================================
def fit(fit_func, params, xs, ys, ys_err=None):
    '''
    Fit via mpfit & return as Fit.params.

    Import function to fit to, inital parameter values & ranges [[value, lo, hi], etc.],
    x-values, y-values, y uncertainties.
    '''
    def test_fit(p, fit_func, x, y, y_err, fjac=None):    #@UnusedVariable
        '''
        Import parameter values, fit function, [fjac] if want partial derivs.
        '''
        model = fit_func(x, p)
        status = 0    # non-negative status value means MPFIT should continue
        return [status, (y - model) / y_err]

    Say = ut_io.SayClass(fit)
    xs, ys = np.array(xs, np.float64), np.array(ys, np.float64)    # need 64 bit for mpfit
    if ys_err is None or not len(ys_err):
        ys_err = np.zeros(ys.size) + 0.1 * ys.mean()
    else:
        ys_err = np.array(ys_err, np.float64)
    if np.NaN in ys:
        Say.say('! Nan values in y array')
    if np.NaN in ys_err:
        Say.say('! Nan values in y_er array')
    if np.min(ys_err) <= 0:
        Say.say('! 0 (or negative) values in y_err, excising these from fit')
        yis = ut_array.arange_length(ys_err)
        yis = yis[ys_err > 0]
        xs = xs[yis]
        ys = ys[yis]
        ys_err = ys_err[yis]

    pinfo = []
    for p in params:
        # if parameters limit are the same, interpret as fixed parameter
        if p[1] == p[2]:
            pinfo.append({'value': p[0], 'fixed': 1})
        else:
            pinfo.append({'value': p[0], 'fixed': 0, 'limited': [1, 1], 'limit': [p[1], p[2]]})
    fa = {'fit_func': fit_func, 'x': xs, 'y': ys, 'y_err': ys_err}
    Fit = ut_nmpfit.MpFit(test_fit, parinfo=pinfo, functkw=fa, quiet=1)
    if Fit.status >= 5:
        Say.say('! mpfit status %d, tolerances too small' % Fit.status)
    elif Fit.status <= 0:
        Say.say('! mpfit error: ', Fit.errmsg)
    for pi in xrange(Fit.params.size):
        if Fit.params[pi] == params[pi][1] or Fit.params[pi] == params[pi][2]:
            Say.say('! fit param %d = %.3f is at its input limit' % (pi, Fit.params[pi]))
    return Fit


def get_chisq_reduced(vals_test, vals_ref, vals_ref_err, param_num=1):
    '''
    Get reduced chi-squared.
    Excise reference values with 0 uncertainty.

    Import values to test, reference values, uncertainty in reference values (can be asymmetric),
    number of free parameters in getting test values.
    '''
    Say = ut_io.SayClass(get_chisq_reduced)
    vals_test = ut_array.arrayize(vals_test)
    vals_ref = ut_array.arrayize(vals_ref)
    vals_ref_err = ut_array.arrayize(vals_ref_err)
    if np.ndim(vals_ref_err) > 1:
        # get uncertainty on correct side of reference values
        if vals_ref_err.shape[0] != vals_ref.size:
            vals_ref_err = vals_ref_err.transpose()
        vals_ref_err_sided = np.zeros(vals_ref_err.shape[0])
        vals_ref_err_sided[vals_test <= vals_ref] = vals_ref_err[:, 0][vals_test <= vals_ref]
        vals_ref_err_sided[vals_test > vals_ref] = vals_ref_err[:, 1][vals_test > vals_ref]
    else:
        vals_ref_err_sided = vals_ref_err
    vis = ut_array.arange_length(vals_ref)[vals_ref_err_sided > 0]
    if vis.size != vals_ref.size:
        Say.say('excise %d reference values with uncertainty = 0' % (vals_ref.size - vis.size))
    chi2 = np.sum(((vals_test[vis] - vals_ref[vis]) / vals_ref_err_sided[vis]) ** 2)
    dof = vis.size - 1 - param_num
    return chi2 / dof


#===================================================================================================
# general functions
#===================================================================================================
class FunctionClass:
    '''
    Collection of functions, for fitting.
    '''
    def get_ave(self, func, p, x_lim=[0, 1]):
        def integrand_func_ave(x, func, p):
            return x * func(x, p)
        return integrate.quad(integrand_func_ave, x_lim[0], x_lim[1], (func, p))[0]

    def gaussian(self, x, p):
        return 1 / ((2 * np.pi) ** 0.5 * p[1]) * np.exp(-0.5 * ((x - p[0]) / p[1]) ** 2)

    def gaussian_normalized(self, x):
        return 1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * x ** 2)

    def gaussian_double(self, x, p):
        return (p[2] * np.exp(-0.5 * ((x - p[0]) / p[1]) ** 2) + (1 - p[2]) *
                np.exp(-0.5 * ((x - p[3]) / p[4]) ** 2))

    def gaussian_double_skew(self, x, p):
        return (p[3] * self.skew(x, p[0], p[1], p[2]) + (1 - p[3]) * self.skew(x, p[4], p[5], p[6]))

    def skew(self, x, e=0, wid=1, skew=0):
        t = (x - e) / wid
        return 2 * stats.norm.pdf(t) * stats.norm.cdf(skew * t) / wid

    def erf_0to1(self, x, p):
        '''
        Varies from 0 to 1.
        '''
        return 0.5 * (1 + special.erf((x - p[0]) / p[1]))

    def erf_AtoB(self, x, p):
        '''
        Varies from p[2] to p[3].
        '''
        return p[2] * 0.5 * (1 + special.erf((x - p[0]) / p[1])) + p[3]

    def line(self, x, p):
        return p[0] + x * p[1]

    def power_law(self, x, p):
            return p[0] + p[1] * x ** p[2]

    def line_exp(self, x, p):
            return p[0] + p[1] * x * np.exp(-x ** p[2])

    def m_function_schechter(self, m, params):
        '''
        Compute d(num-den) / d(log m) = ln(10) * amplitude * (10 ^ (m_star - m_char)) ^ slope *
        exp(-10**(m_star - m_char)).

        Import (stellar) mass, parameters: 0 = amplitude, 1 = m_char, 2 = slope.
        '''
        rats = 10 ** (m - params[1])
        return np.log(10) * params[0] * rats ** params[2] * np.exp(-rats)

    def numden_schechter(self, m, params, m_max=20):
        '''
        Get cumulative number density above m.

        Import (stellar) mass, parameters: 0 = amplitude, 1 = m_char, 2 = slope,
        maximum mass for integration.
        '''
        return integrate.quad(self.m_function_schechter, m, m_max, (params))[0]

Function = FunctionClass()


def sample_random_reject(func, params, x_lim, y_max, size):
    '''
    Use rejection method to sample distribution & return as array.
    Assumes minimum of func is 0.

    Import general function, its parameters, x-range, maximum value of func,
    number of values to sample.
    '''
    xs_rand = np.random.uniform(x_lim[0], x_lim[1], size)
    ys_rand = np.random.uniform(0, y_max, size)
    ys = func(xs_rand, params)
    xs_rand = xs_rand[ys_rand < ys]
    x_num = xs_rand.size
    if x_num < size:
        xs_rand = np.append(xs_rand, sample_random_reject(func, params, x_lim, y_max, size - x_num))
    return xs_rand


def mag_function_schechter(mag, amplitude, mag_char, slope):
    '''
    Schechter function, in magnitude form.
    '''
    mag_rat = 10 ** (slope * (mag_char - mag) / 2.5)
    return (np.log(10) / 2.5 * amplitude * mag_rat) * np.exp(-mag_rat)


def convert_lum(lum_kind, lum):
    '''
    Convert to other kind (lum, mag).

    Import luminosity kind (lum, mag), its value (if lum, in solar units).
    '''
    if lum_kind == 'lum':
        return const.sun_mag - 2.5 * log10(lum)
    elif lum_kind == 'mag.r':
        if lum > 0:
            lum = -lum
        return 10 ** ((const.sun_mag - lum) / 2.5)
    else:
        raise ValueError('not recognize luminosity kind: %s' % lum_kind)
