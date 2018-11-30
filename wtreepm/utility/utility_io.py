'''
Utility functions for reading, writing, printing during run time.

Author[s]: Andrew Wetzel.
'''

# system -----
from __future__ import division
import os
import glob
import numpy as np
import cPickle as pickle
# local -----
import utility_array as ut_array


#===================================================================================================
# print at run-time
#===================================================================================================
class ListPropClass:
    '''
    Print self & attributes in nice format.
    '''
    def __repr__(self):
        return '< Instance of %s:\n%s>' % (self.__module__ + '.' + self.__class__.__name__,
                                           self.get_attr_names())

    def get_attr_names(self):
        result = ''
        for attr in self.__dict__.keys():
            if attr[:2] == '__':
                result += '  self.%s = <built-in>\n' % attr
            else:
                result += '  self.%s = %s\n' % (attr, self.__dict__[attr])
        return result


class SayClass(ListPropClass):
    '''
    Print comments & diagnostics at runtime in nice format.
    '''
    is_first_print = True

    def __init__(self, func=None):
        '''
        Import function to get its name, if not using in class.
        '''
        if func is not None:
            self.func_name = func.__module__ + '.' + func.__name__
        elif 'SayClass' in str(self.__class__):
            raise ValueError('need to pass function to get name')

    def say(self, words):
        '''
        Print words in nice format.
        '''
        if words[0] != '!':
            words = '  ' + words
        if self.is_first_print:
            if 'SayClass' in str(self.__class__):
                print('# in %s():' % self.func_name)
            else:
                print('# in %s():' % (self.__module__ + '.' +
                                      self.__class__.__name__.replace('Class', '')))
            self.is_first_print = False
        print(words)
        os.sys.stdout.flush()


def print_flush(words):
    print(words)
    os.sys.stdout.flush()


def print_array(vals, form='%.3f', delimeter=','):
    '''
    .
    '''
    string = form + delimeter
    for val in vals[:-1]:
        print string % val,
    print form % vals[-1]
    os.sys.stdout.flush()


#===================================================================================================
# read/write file
#===================================================================================================
def get_safe_path(directory):
    '''
    Get name, safely including trailing /.

    Import directory path name, or list of names.
    '''
    if np.isscalar(directory):
        # single directory
        if directory[-1] != '/':
            directory += '/'
    else:
        # list of directories
        for di in xrange(len(directory)):
            if directory[di][-1] != '/':
                directory[di] += '/'
    return directory


def rename_files(directory, string_old='', string_new=''):
    '''
    For all file names containing string_old, rename string_old to string_old.

    Import directory, string of file name to replace (can use *), string with which to replace it
    (can also use *).
    '''
    directory = get_safe_path(directory)
    file_names = glob.os.listdir(directory)
    if not file_names:
        print 'found no files in directory: %s' % directory
    if '*' in string_old and '*' in string_new:
        strings_old = string_old.split('*')
        strings_new = string_new.split('*')
    else:
        strings_old = [string_old]
        strings_new = [string_new]
    if len(strings_old) != len(strings_new):
        raise ValueError('length of strings_old = %s not match strings_new = %s' %
                         (strings_old, strings_new))
    for file_name in file_names:
        file_name_new = file_name
        string_in_file = [False for _ in xrange(len(strings_old))]
        for si in xrange(len(strings_old)):
            if strings_old[si] in file_name:
                string_in_file[si] = True
                file_name_new = file_name_new.replace(strings_old[si], strings_new[si])
        if np.min(string_in_file) and file_name_new != file_name:
            print 'in', directory, 'rename', file_name, 'to', file_name_new
            file_name = directory + file_name
            file_name_new = directory + file_name_new
            glob.os.rename(file_name, file_name_new)


def get_numbers_in_string(string, scalarize=False):
    '''
    Get list of int & float numbers in string.

    Import string, whether to return scalar value if only one number.
    '''
    numbers = []
    number = ''
    for ci, char in enumerate(string):
        if char.isdigit():
            number += char
        elif char == '.':
            if (number and ci > 0 and string[ci - 1].isdigit() and
                len(string) > ci + 1 and string[ci + 1].isdigit()):
                number += char
        if number and ((not char.isdigit() and not char == '.') or ci == len(string) - 1):
            if '.' in number:
                numbers.append(float(number))
            else:
                numbers.append(int(number))
            number = ''
    if scalarize:
        numbers = ut_array.scalarize(numbers)
    return numbers


def get_file_names(file_name_base, number_type=None, sort_kind=''):
    '''
    Get all file name[s] (with full path) with given base [& number in each file name].

    Import file name base (with full path) using * as wildcard,
    what kind of number in file name to get (None, int, float) -
    get last one of given type in file name,
    whether to return list of file names & numbers in reverse order.
    '''
    # get all file names matching string in directory
    file_names = glob.glob(file_name_base)
    file_names.sort()
    if not file_names:
        raise ValueError('found no files with base name: %s' % file_name_base)
    if number_type:
        # for file names with number/aexp, get last number of given type in each file name
        file_numbers = []
        for file_name in file_names:
            file_number = None
            file_numbers_t = get_numbers_in_string(file_name, scalarize=False)
            if file_numbers_t == []:
                raise ValueError('cannot get number for file name: %s' % file_name)
            for file_number_t in reversed(file_numbers_t):
                if isinstance(file_number_t, number_type):
                    file_number = file_number_t
            if file_number:
                file_numbers.append(file_number)
            else:
                raise ValueError('no number of type %s in file name: %s' % (number_type, file_name))
        file_numbers = np.array(file_numbers)
        if sort_kind == 'reverse':
            file_names = file_names[::-1]
            file_numbers = file_numbers[::-1]
        return file_names, file_numbers
    else:
        if len(file_names) > 1 and sort_kind == 'reverse':
            file_names = file_names[::-1]
        return file_names


def get_file_names_nearest_number(file_name_base, numbers=None, number_type=float, sort_kind='',
                                  arrayize=False, dif_tolerance=0.1):
    '''
    Get file name (with full path) & number for each file whose number in its name is closest to
    input number[s].

    Import file name base (with full path), list of numbers (such as expansion scale factors),
    what kind of number in file name to get (None, int, float),
    sort kind ('' = none, 'forward', 'reverse'),
    whether to force return as list/array, even if single element,
    tolerance for warning flag in number rounding.
    '''
    numbers = ut_array.arrayize(numbers)
    if sort_kind:
        numbers = np.sort(numbers)
        if sort_kind == 'reverse':
            numbers = numbers[::-1]
    # get all file names & numbers matching string in directory
    file_names_read, file_numbers_read = get_file_names(file_name_base, number_type)
    file_names = []
    file_numbers = []
    for number in numbers:
        number_difs = abs(number_type(number) - file_numbers_read)
        inear = np.argmin(number_difs)
        # warn if number of file is too far from input value
        if number_difs[inear] > dif_tolerance:
            print ('! input number = %s, but nearest file number = %s' %
                   (number, file_numbers_read[inear]))
        file_names.append(file_names_read[inear])
        file_numbers.append(file_numbers_read[inear])
    if numbers.size == 1 and not arrayize:
        # if input scalar number, return as scalar
        file_names = file_names[0]
        file_numbers = file_numbers[0]
    else:
        file_names, file_numbers = np.array(file_names), np.array(file_numbers)
    return file_names, file_numbers


def get_file_names_intersect(file_name_bases=[], numbers_in=[], number_type=float, sort_kind='',
                             arrayize=False):
    '''
    Get number for each file that exists in all file names with given base whose number in its name
    is closest to input number[s].

    Import [list of] file name base[s], list of numbers (such as expansion scale factors),
    what kind of number in file name to get (None, int, float),
    sort kind ('' = none, 'forward', 'reverse'),
    whether to force return as list/array, even if single element.
    '''
    file_name_bases = ut_array.arrayize(file_name_bases)
    if numbers_in is None or numbers_in == []:
        file_numbers = get_file_names(file_name_bases[0], number_type)[1]
    else:
        file_numbers = get_file_names_nearest_number(file_name_bases[0], numbers_in, number_type,
                                                     arrayize=arrayize)[1]
    for file_i in xrange(1, file_name_bases.size):
        file_numbers_t = get_file_names(file_name_bases[file_i], number_type)[1]
        file_numbers = np.intersect1d(file_numbers, file_numbers_t)
    return get_file_names_nearest_number(file_name_bases[0], file_numbers, number_type, sort_kind,
                                         arrayize)


def pickle_object(file_name_base, direction='read', obj=None, print_name=True):
    '''
    Write or read pickle version of object.

    Import file name (without .pickle), pickle direction (read, write), general object,
    whether to print file name.
    '''
    file_name_base += '.pkl'
    if direction == 'write':
        if obj is None:
            raise ValueError('no object to pickle out')
        file_out = open(file_name_base, 'w')
        pickle.dump(obj, file_out)
        file_out.close()
        if print_name:
            print '# wrote %s' % file_name_base
    elif direction == 'read':
        file_out = open(file_name_base, 'r')
        obj = pickle.load(file_out)
        file_out.close()
        if print_name:
            print '# read %s' % file_name_base.split('/')[-1]
        return obj
    else:
        raise ValueError('not recognize pickle direction = %s' % direction)


#===================================================================================================
# command-line run in parallel
#===================================================================================================
def run_in_parallel(commands=[''], thread_num=4, print_diagnostic=False):
    '''
    Run list of commands in parallel.

    Import list of command strings, number of threads to run in parallel.
    '''
    commands = ut_array.arrayize(commands)
    if thread_num > commands.size:
        thread_num = commands.size    # wasteful to use more parallel threads than aexps

    if thread_num == 1:
        for ci in xrange(len(commands)):
            os.system(commands[ci])
            if print_diagnostic:
                print_flush('ran %s' % commands[ci])
    else:
        child_num = 0
        for ci in xrange(commands.size):
            proc_id = os.fork()    # return 0 to child process & proc id of child to parent process
            if proc_id:
                child_num += 1
                if child_num >= thread_num:
                    os.wait()    # wait for completion of child process
                    child_num -= 1
            else:
                os.system(commands[ci])    # if child process, execute command
                if print_diagnostic:
                    print_flush('ran %s' % commands[ci])
                os.sys.exit(0)

        while child_num > 0:
            os.wait()
            child_num -= 1
