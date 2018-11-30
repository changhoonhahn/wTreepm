'''
Utility functions for array creation, manipulation, diagnostic.

Author[s]: Andrew Wetzel.
'''

# system -----
from __future__ import division
import numpy as np
from numpy import Inf
from scipy import stats


#===================================================================================================
# useful classes
#===================================================================================================
class DictClass(dict):
    pass


class ListClass(list):
    pass


#===================================================================================================
# initialization, manipulation, information
#===================================================================================================
def initialize_array(num, dtype=np.int32):
    '''
    Make array of data type with initial values that are negative & out of bounds.

    Import array size, data type.
    '''
    if np.isscalar(num):
        return np.zeros(num, dtype) - num - 1
    else:
        return np.zeros(num, dtype) - num[0] - 1


def arange_length(array_or_length_or_imin=None, imax=None, dtype=np.int32):
    '''
    Get arange corresponding to input limits or input array size.

    Import array or array length or starting value (if latter, also need ending value).
    '''
    if imax is None:
        if np.isscalar(array_or_length_or_imin):
            num = array_or_length_or_imin
        else:
            num = len(array_or_length_or_imin)
        return np.arange(num, dtype=dtype)
    else:
        return np.arange(array_or_length_or_imin, imax, dtype=dtype)


def arange_safe(lim, wid=None, include_max=False, dtype=np.float64):
    '''
    Get arange that safely does [not] reach limit maximum.
    If width < 0, return input limit.

    Import array limit, bin width, whether to include maximum value, data type.
    '''
    if lim is None:
        return lim
    lim = get_limit(lim)
    # ensure upper limit does not quite reach input limit
    lim_max = lim[1] * (1 - np.sign(lim[1]) * 1e-6)
    if lim_max == 0:
        lim_max -= 1e-6
    if include_max:
        if lim[0] == -Inf or lim[1] == Inf:
            return np.array(lim, dtype)
        elif wid == Inf:
            return np.array([lim[0], Inf], dtype)
        elif wid <= 0:
            return np.array(lim, dtype)
        else:
            return np.arange(lim[0], lim_max + wid, wid, dtype)
    else:
        if lim[0] == -Inf or lim_max == Inf:
            return np.array([lim[0]], dtype)
        elif wid == Inf:
            return np.array([lim[0]], dtype)
        elif wid <= 0:
            return np.array([lim[0]], dtype)
        else:
            return np.arange(lim[0], lim_max, wid, dtype)


def arrayize(vals, bit_num=64, repeat_num=1):
    '''
    Convert to array of given bit size.
    If vals is tuple, treat as independent arrays, else treat as one whole array.

    Import [tuple of] value[s]/list[s]/array[s], precision in bits to store,
    factor by which to repeat values periodically.
    '''
    def arrayize_one(val, repeat_num, bit_num):
        if np.isscalar(val):
            if repeat_num == 1:
                val = [val]
            elif repeat_num > 1:
                val = np.r_[repeat_num * [val]]
        val = np.array(val)
        if bit_num == 32:
            if val.dtype == 'float64':
                val = val.astype('float32')
            elif val.dtype == 'int64':
                val = val.astype('int32')
        return val

    if np.isscalar(vals):
        return arrayize_one(vals, repeat_num, bit_num)
    elif len(vals) == 1:
        return arrayize_one(vals, repeat_num, bit_num)
    elif isinstance(vals, tuple):
        arrays = []
        for val in vals:
            arrays.append(arrayize_one(val, repeat_num, bit_num))
        return arrays
    elif isinstance(vals, dict):
        for k in vals:
            vals[k] = arrayize_one(vals[k], repeat_num, bit_num)
        return vals
    else:
        return arrayize_one(vals, repeat_num, bit_num)


def scalarize(vals):
    '''
    If length is one, return as scalar, else, return as is.

    Import [tuple of] value[s]/list[s]/array[s].
    '''
    if vals is None or np.isscalar(vals):
        return vals
    elif len(vals) == 1:
        return vals[0]
    else:
        return vals


def get_limit(vals, cut_num=0, cut_percent=0, digit_num=None):
    '''
    Get tuple of minimum & maximum values, applying all cuts, to given number of digits.

    Import array/list/tuple, n'th unique value from minimum/maximum to keep,
    cut percent from mininum/maximum to keep, number of digits to keep (for printing).
    '''
    val_lim = [np.min(vals), np.max(vals)]
    if cut_num > 0:
        vals_unique = np.unique(vals)    # returns sorted values
        val_lim = [vals_unique[cut_num], vals_unique[-cut_num]]
    if cut_percent > 0:
        val_lim_temp = [stats.scoreatpercentile(vals, cut_percent),
                        stats.scoreatpercentile(vals, 100 - cut_percent)]
        val_lim = [max(val_lim[0], val_lim_temp[0]), min(val_lim[1], val_lim_temp[1])]
    if digit_num is None:
        return val_lim
    else:
        return [round(val_lim[0], digit_num), round(val_lim[1], digit_num)]


def expand_limit(vals, extra):
    '''
    Get expanded limits.

    Import array or limits, amoung to expand limits in both directions.
    '''
    if vals is None or extra is None:
        return None
    else:
        return [np.min(vals) - extra, np.max(vals) + extra]


def arrayize_dictionary(dic):
    '''
    Convert list entries to numpy arrays.

    Import dictionary of lists.
    '''
    for k in dic:
        if isinstance(dic[k], dict):
            dic[k] = arrayize_dictionary(dic[k])
        elif isinstance(dic[k], list) and len(dic[k]):
            dic[k] = np.array(dic[k])


def append_dictionary(dict_1, dict_2):
    '''
    Append elements of dict_2 that are in dict_1 to dict_1.
    If dict_1 is empty, append all elements of dict_2 to it.

    Import dictionaries (dict1 can be empty).
    '''
    # initialize dict1, if necessary
    if not dict_1:
        for k in dict_2:
            if isinstance(dict_2[k], dict):
                dict_1[k] = {}
                for kk in dict_2[k]:
                    dict_1[k][kk] = []
            else:
                dict_1[k] = []
    # append values to dict1
    for k in dict_1:
        if k in dict_2:
            if isinstance(dict_1[k], dict):
                for kk in dict_1[k]:
                    if kk in dict_2[k]:
                        if np.isscalar(dict_1[k][kk]):
                            dict_1[k][kk] = [dict_1[k][kk]]
                        dict_1[k][kk].append(dict_2[k][kk])
            else:
                if np.isscalar(dict_1[k]):
                    dict_1[k] = [dict_1[k]]
                dict_1[k].append(dict_2[k])


def print_extrema(vals, num=5):
    '''
    Print n'th unique extrema values in array.

    Import array / list / tuple, number of minimum / maximum values to print.
    '''
    vals_unique = np.unique(vals)    # returns sorted values
    print '# minima: ', vals_unique[:num]
    print '# maxima: ', vals_unique[-num:]


def print_list(vals, digit_num=3, print_vertical=False, print_comma=False):
    '''
    Print list in nice format.

    Import array / list / tuple, number of digits to print.
    '''
    string = '%.' + '%d' % digit_num + 'f'
    if print_comma:
        string += ','
    if print_vertical:
        for val in vals:
            print string % val
    else:
        for val in vals[:-1]:
            print string % val,
        print string.replace(',', '') % vals[-1]


#===================================================================================================
# comparison
#===================================================================================================
def compare_arrays(a1, a2, print_bad=True, tolerance=0.01):
    '''
    Check if values in arrays are the same (within tolerance percent if float).

    Import 2 arrays, whether to print values of mismatches, fractional difference tolerance.
    '''
    bad_num = 0
    if len(a1) != len(a2):
        print '! a1 len = %d, a2 len = %d' % (len(a1), len(a2))
        return
    if np.shape(a1) != np.shape(a2):
        print '! a1 shape =', np.shape(a1), 'a2 shape =', np.shape(a2)
        return
    if 'int' in a1.dtype.name:
        for a1i in xrange(len(a1)):
            if np.isscalar(a1[a1i]):
                if a1[a1i] != a2[a1i]:
                    if print_bad:
                        print '!', a1i, a1[a1i], a2[a1i]
                    bad_num += 1
            else:
                for a1ii in xrange(len(a1[a1i])):
                    if a1[a1i][a1ii] != a2[a1i][a1ii]:
                        if print_bad:
                            print '!', a1i, a1ii, a1[a1i][a1ii], a2[a1i][a1ii]
                        bad_num += 1
    elif 'float' in a1.dtype.name:
        for a1i in xrange(len(a1)):
            if np.isscalar(a1[a1i]):
                if a1[a1i] == a2[a1i]:
                    continue
                elif abs(np.max((a1[a1i] - a2[a1i]) / (a1[a1i] + 1e-10))) > tolerance:
                    if print_bad:
                        print '!', a1i, a1[a1i], a2[a1i]
                    bad_num += 1
            else:
                for a1ii in xrange(len(a1[a1i])):
                    if a1[a1i][a1ii] == a2[a1i][a1ii]:
                        continue
                    elif abs(a1[a1i][a1ii] - a2[a1i][a1ii]) / abs(a1[a1i][a1ii]) > tolerance:
                        if print_bad:
                            print('!', a1i, a1ii, a1[a1i][a1ii], a2[a1i][a1ii],
                                  abs(a1[a1i][a1ii] - a2[a1i][a1ii]) / abs(a1[a1i][a1ii]))
                        bad_num += 1
    else:
        print '! dtype = %s, not examined' % a1.dtype
    print 'bad num = %d' % bad_num


def compare_dictionaries(dic_1, dic_2, print_bad=True, tolerance=0.01):
    '''
    Check if values in dictionaries are the same (within tolerance percent if float).

    Import dictionaries, whether to print mismatches, fractional difference tolerance.
    '''
    bad_num = 0
    if len(dic_1) != len(dic_2):
        print '! dic1 len = %d, dic2 len = %d' % (len(dic_1), len(dic_2))
        return
    for k in dic_1:
        if k not in dic_2:
            print '! %s not in dic2' % k
        if np.isscalar(dic_1[k]):
            if dic_1[k] != dic_2[k]:
                if print_bad:
                    print '!', k, dic_1[k], dic_2[k]
                bad_num += 1
        elif len(dic_1[k]):
            print '%10s' % k,
            compare_arrays(dic_1[k], dic_2[k], print_bad, tolerance)
    for k in dic_2:
        if k not in dic_1:
            print '! %s not in dic1' % k


#===================================================================================================
# sub-sampling
#===================================================================================================
def elements(vals, lim=[-Inf, Inf], vis=None, vis_2=None, get_indices=False, dtype=np.int32):
    '''
    Get the indices of the input values that are within the input limit, that also are in input vis
    index array (if defined).
    Either of limits can have same range as vals.

    Import array, range to keep, prior indices of vals array to keep,
    other array to sub-sample in same way, whether to return selection indices of input vis array.
    '''
    if not isinstance(vals, np.ndarray):
        vals = np.array(vals)
    # check if input array
    if vis is None:
        vis = np.arange(vals.size, dtype=dtype)
    else:
        vals = vals[vis]
    vis_keep = vis
    # check if limit is just one value
    if np.isscalar(lim):
        keeps = (vals == lim)
    else:
        # sanity check - can delete this eventually
        if isinstance(lim[0], int) and isinstance(lim[1], int):
            if lim[0] == lim[1]:
                raise ValueError('input limit = %s, has same value' % lim)
            if lim[0] != lim[1] and 'int' in vals.dtype.name:
                print '! elements will not keep objects at lim[1] = %d' % lim[1]
        if not np.isscalar(lim[0]) or lim[0] > -Inf:
            keeps = (vals >= lim[0])
        else:
            keeps = None
        if not np.isscalar(lim[1]) or lim[1] < Inf:
            if keeps is None:
                keeps = (vals < lim[1])
            else:
                keeps *= (vals < lim[1])
        elif keeps is None:
            keeps = np.arange(vals.size, dtype=dtype)
    if get_indices:
        if vis_2 is not None:
            return vis_keep[keeps], vis_2[keeps], np.arange(vis.size, dtype=dtype)[keeps]
        else:
            return vis_keep[keeps], np.arange(vis.size, dtype=dtype)[keeps]
    else:
        if vis_2 is not None:
            return vis_keep[keeps], vis_2[keeps]
        else:
            return vis_keep[keeps]


def sample_array(vals, num):
    '''
    Get randomly sampled version of array.
    If num > vals.size, randomly sample (with repeat) from vals,
    if num <= vals.size, randomly sample without repeat.

    Import array of values, number of elements to sample.
    '''
    if not vals.size or num == 0:
        return np.array([])
    if num > vals.size:
        return vals[np.random.randint(0, vals.size, num)]
    else:
        vals_rand = vals.copy()
        np.random.shuffle(vals_rand)
        return vals_rand[:num]
