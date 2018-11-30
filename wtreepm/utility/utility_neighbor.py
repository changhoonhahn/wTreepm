'''
Class for finding nearest neighbors.

Author[s]: Andrew Wetzel.
'''

# system -----
from __future__ import division
import numpy as np
from numpy import Inf
from scipy import spatial
# local -----
from . import utility_io as ut_io
from . import utility_array as ut_array
from . import utility_bin as ut_bin
from . import utility_coord as ut_coord


class NeigClass(ut_io.SayClass):
    '''
    Find neighbors using direct search, chaining mesh, or k-d tree.
    '''
    def get_neig(self, neig_props, cen_poss, neig_poss, num_max=50, dist_lim=[1e-6, 10],
                 periodic_len=None, find_kind='kd-tree', format_list=False, neig_is=None):
        '''
        Get neighbor distances, indices (in either neig_poss array or neig_is, if defined),
        &/or counts (up to num_max) within distance limit.

        Import neighbor properties to get (indices, distances, number),
        array of positions (object number x dimension number) around which to count neighbors,
        array of positions (object number x dimension number) of neighbors,
        neighbor max number & distance limit, periodicity length (if none, do not use periodic),
        neighbor find method (direct, mesh, kd-tree, kd-tree.ang),
        whether to return distances & indices as list instead of center_num x num_max array,
        ids of neighbors (to return instead of indices), whether to print diagnostic information.
        '''
        neig = {}
        if dist_lim[0] < 0:
            raise ValueError('distance_limit[0] = %f not valid' % dist_lim[0])
        elif periodic_len and dist_lim[1] >= 0.5 * periodic_len:
            raise ValueError('cannot support distance_limit[1] = %f > 0.5 * periodic_length = %f' %
                             (dist_lim[1], periodic_len))
        if dist_lim[0] == 0:
            self.say('keep neighbors at distance = 0')

        # make sure position arrays are 2-D
        if np.ndim(cen_poss) == 1:
            cen_poss = np.array([cen_poss])
        if np.ndim(neig_poss) == 1:
            neig_poss = np.array([neig_poss])
        if cen_poss.shape[1] != neig_poss.shape[1]:
            raise ValueError('cen_poss.shape[1] = %d != neig_poss.shape[1] %d' %
                             (cen_poss.shape[1], neig_poss.shape[1]))
        cen_num = cen_poss.shape[0]

        self.say('neighbor find via %s' % find_kind)
        self.say('center_num = %d, neighbor_num = %d, neighbor array size = %d' %
                 (cen_num, neig_poss.shape[0], cen_num * num_max))
        if find_kind == 'direct':
            neig['distances'], neig_iis = self.get_neig_direct(cen_poss, neig_poss, num_max,
                                                               dist_lim, periodic_len)
        elif find_kind == 'mesh':
            neig['distances'], neig_iis = self.get_neig_mesh(cen_poss, neig_poss, num_max,
                                                             dist_lim, periodic_len)
        elif find_kind == 'kd-tree':
            neig['distances'], neig_iis = self.get_neig_kd_tree(cen_poss, neig_poss, num_max,
                                                                dist_lim, periodic_len)
        elif find_kind == 'kd-tree.ang':
            neig['distances'], neig_iis = self.get_neig_kd_tree_ang(cen_poss, neig_poss, num_max,
                                                                    dist_lim, periodic_len)
        else:
            raise ValueError('not recognize find_kind = %s' % find_kind)

        if neig_is is not None:
            # fix neighbor raw indices to point to input neighbor indices
            neig_is = np.append(np.array([-neig_is.size - 1], np.int32), neig_is)
            neig_iis[neig_iis != -neig_poss.shape[0] - 1] += 1
            neig['indices'] = neig_is[neig_iis]
        else:
            neig['indices'] = neig_iis
        # count number of neighbors within maximum distance
        neig_props = neig_props.split()
        if 'number' in neig_props:
            neig['number'] = ut_array.initialize_array(cen_num)
            for ci in xrange(cen_num):
                neig['number'][ci] = neig['distances'][ci][neig['distances'][ci] < Inf].size
        self.say('minimum neighbor distance got = %.2e' % neig['distances'].min())
        # check if neighbor list is saturated at num_max
        if 0 < neig['distances'][:, -1].min() < Inf:
            self.say(('! reached neighbor num_max = %d: distance_max = %.2f, ' +
                      'minimum distance(num_max) got = %.2f | ') %
                     (num_max, dist_lim[1], neig['distances'][:, -1].min()))
        if 'indices' in neig_props or 'distances' in neig_props:
            if format_list:
                # return neighbor indices & distances as lists of extant objects
                neig_indices = neig['indices']
                neig_dists = neig['distances']
                neig['indices'] = []
                neig['distances'] = []
                for ci in xrange(cen_num):
                    neig_indices_t = neig_indices[ci][neig_indices[ci] >= 0]
                    neig['indices'].append(neig_indices_t)
                    neig_dists_t = neig_dists[ci][neig_dists[ci] < Inf]
                    neig['distances'].append(neig_dists_t)
                    if neig_indices_t.size != neig_dists_t.size:
                        raise ValueError('kept %d indices but kept %d distances for object %d' %
                                         (neig_indices_t.size, neig_dists_t.size, ci))
        # return as list of arrays
        if len(neig_props) == 1:
            neig = neig[neig_props[0]]
        else:
            neig = [neig[k] for k in neig_props]
        return ut_array.scalarize(neig)

    def get_neig_direct(self, cen_poss, neig_poss, num_max=50, dist_lim=[1e-6, 10],
                        periodic_len=None):
        '''
        Get neighbor distances & indices from input neig_poss array, using direct n ^ 2 calcuation.

        Import array of center positions (object number x dimension number) around which to count
        neighbors, array of neighbors positions, neighbor maximum number & distance range,
        periodicity length (if none, do not use periodic).
        '''
        if cen_poss.shape[1] == 2 and (periodic_len == 360 or periodic_len == 2 * np.pi):
            angular = True
            self.say('compute 2-D angular distance')
        else:
            angular = False
            dimen_is = ut_array.arange_length(cen_poss.shape[1])
        dist_min_2 = dist_lim[0] ** 2
        dist_max_2 = dist_lim[1] ** 2
        cis = ut_array.arange_length(cen_poss.shape[0])
        nis = ut_array.arange_length(neig_poss.shape[0])
        neig_indices = np.zeros((cen_poss.shape[0], num_max), np.int32) - neig_poss.shape[0] - 1
        dists = np.zeros((cen_poss.shape[0], num_max), np.float32)
        dists += Inf    # to keep as np.float32
        for ci in cis:
            if angular:
                # compute 2-D angular distance
                dist2s = ut_coord.distance_ang('scalar^2', cen_poss[ci], neig_poss, periodic_len)
            else:
                dist2s = ut_coord.distance('scalar^2', cen_poss[ci], neig_poss, periodic_len,
                                           dimen_is)
            # keep neighbors closer than dist_max that are not self
            nis_real = nis[dist2s < dist_max_2]
            nis_real = nis_real[dist2s[nis_real] >= dist_min_2]
            # sort neighbors by distance
            nis_real = nis_real[np.argsort(dist2s[nis_real]).astype(nis_real.dtype)[:num_max]]
            neig_indices[ci, :nis_real.size] = nis_real
            dists[ci, :nis_real.size] = dist2s[nis_real] ** 0.5
        return dists, neig_indices

    def get_neig_mesh(self, cen_poss, neig_poss, num_max=50, dist_lim=[0, 10], periodic_len=None):
        '''
        Get neighbor distances & indices from input neig_poss array, using chaining mesh.

        Import array of center positions (object number x dimension number) around which to count
        neighbors, array of neighbor positions, neighbor maximum number & distance range,
        periodicity length (if none, do not use periodic).
        '''
        def make_mesh(poss, mesh_num, periodic_len=None):
            '''
            Get dictionary list of position ids in each mesh cell, mesh size.

            Import object num x dimension num position array, number of mesh cells per dimension,
            periodicity length.
            Assign positions to mesh cells.
            '''
            dimen_num = poss.shape[1]
            if periodic_len:
                x_min, x_max = 0, periodic_len
            else:
                x_min, x_max = poss.min(), poss.max() * 1.001
            mesh_poss, mesh_wid = np.linspace(x_min, x_max, mesh_num + 1, True, True)
            mesh_bin_indices = ut_bin.idigitize(poss.flatten(), mesh_poss)
            mesh_bin_indices.shape = poss.shape
            mesh_pos_indices = {}
            mesh_range = xrange(mesh_num)
            if dimen_num == 3:
                for x in mesh_range:
                    for y in mesh_range:
                        for z in mesh_range:
                            mesh_pos_indices[(x, y, z)] = []
            elif dimen_num == 2:
                for x in mesh_range:
                    for y in mesh_range:
                        mesh_pos_indices[(x, y)] = []
            for mii in xrange(mesh_bin_indices.shape[0]):
                mesh_pos_indices[tuple(mesh_bin_indices[mii])].append(mii)
            return mesh_pos_indices, mesh_wid

        num_per_mesh = 8    # number of objects per mesh (assuming average density)

        if not periodic_len:
            raise ValueError('mesh neighbor finder not support non-periodic boundary')
        obj_num = cen_poss.shape[0]
        dimen_num = cen_poss.shape[1]
        if dimen_num == 2 and (periodic_len == 360 or periodic_len == 2 * np.pi):
            angular = True
        else:
            angular = False
        dist_min_2 = dist_lim[0] ** 2
        dist_max_2 = dist_lim[1] ** 2
        neig_indices = np.zeros((obj_num, num_max), np.int32) - neig_poss.shape[0] - 1
        dists = np.zeros((obj_num, num_max), np.float32)
        dists += Inf    # do this way to keep as np.float32
        # number of mesh points per dimen (make sure is even)
        mesh_num = 2 * int((obj_num / num_per_mesh / 2 ** dimen_num) ** (1 / dimen_num))
        self.say('using %d mesh cells per dimension' % mesh_num)
        mesh_pos_indices, _ = make_mesh(cen_poss, mesh_num, periodic_len)
        mesh_neig_indices, mesh_wid = make_mesh(neig_poss, mesh_num, periodic_len)
        loop_num = int(dist_lim[1] / mesh_wid) + 1
        mesh_loops = xrange(-loop_num, loop_num + 1)
        loop_num_2 = (loop_num + 1) ** 2
        for mesh_point in mesh_pos_indices.keys():
            if len(mesh_pos_indices[mesh_point]) > 0:
                neig_indices_mesh = []    # indices of position array
                if dimen_num == 3:
                    for x in mesh_loops:
                        xx = (x + mesh_point[0]) % mesh_num
                        for y in mesh_loops:
                            yy = (y + mesh_point[1]) % mesh_num
                            for z in mesh_loops:
                                if x ** 2 + y ** 2 + z ** 2 < loop_num_2:
                                    zz = (z + mesh_point[2]) % mesh_num
                                    neig_indices_mesh.extend(mesh_neig_indices[(xx, yy, zz)])
                elif dimen_num == 2:
                    for x in mesh_loops:
                        xx = (x + mesh_point[0]) % mesh_num
                        for y in mesh_loops:
                            if x ** 2 + y ** 2 < loop_num_2:
                                yy = (y + mesh_point[1]) % mesh_num
                                neig_indices_mesh.extend(mesh_neig_indices[(xx, yy)])
                if len(neig_indices_mesh) > 1:
                    # not just self in neighbor list
                    neig_indices_mesh = np.array(neig_indices_mesh, np.int32)
                    nis_mesh = ut_array.arange_length(neig_indices_mesh)
                    # loop over object indices in mesh point
                    for mi in mesh_pos_indices[mesh_point]:
                        if angular:
                            dist2s = ut_coord.distance_ang(
                                'scalar^2', cen_poss[mi], neig_poss[neig_indices_mesh],
                                periodic_len)
                        else:
                            dist2s = ut_coord.distance(
                                'scalar^2', cen_poss[mi], neig_poss[neig_indices_mesh],
                                periodic_len)
                        # keep neighbors closer than distmax that are not self
                        nis_real = nis_mesh[dist2s < dist_max_2]
                        nis_real = nis_real[dist2s[nis_real] >= dist_min_2]
                        # sort neighbors by distance
                        nis_sort = nis_real[
                            np.argsort(dist2s[nis_real]).astype(nis_real.dtype)[:num_max]]
                        neig_indices[mi][:nis_real.size] = neig_indices_mesh[nis_sort]
                        dists[mi][:nis_real.size] = dist2s[nis_sort] ** 0.5
        return dists, neig_indices

    def get_neig_kd_tree(self, cen_poss, neig_poss, num_max=50, dist_lim=[0, 10],
                         periodic_len=None):
        '''
        Get neighbor distances & indices from input neig_poss array, using kd-tree,
        creating buffer neighbors if periodic.

        Import array of center positions (object number x dimension number) around which to count
        neighbors, array of neighbor positions, neighbor maximum number & distance range,
        periodicity length (if none, do not use periodic), whether to print diagnostic information.
        '''
        num_max += 1    # accomodate selecting self as neighbor
        dimen_num = cen_poss.shape[1]
        neig_num = neig_poss.shape[0]
        cis = ut_array.arange_length(cen_poss.shape[0])
        dists = np.zeros((cis.size, num_max), np.float32)
        dists += Inf    # do this way to keep as np.float32
        if periodic_len:
            # create buffer neighbors beyond cube edge
            neig_poss_buffer = np.array(neig_poss)
            neig_poss_temp = np.array(neig_poss)
            if dimen_num == 3:
                for offset_0 in (0, -1, 1):
                    neig_poss_temp[:, 0] = neig_poss[:, 0] + offset_0 * periodic_len
                    for offset_1 in (0, -1, 1):
                        neig_poss_temp[:, 1] = neig_poss[:, 1] + offset_1 * periodic_len
                        for offset_2 in (0, -1, 1):
                            neig_poss_temp[:, 2] = neig_poss[:, 2] + offset_2 * periodic_len
                            if offset_0 == offset_1 == offset_2 == 0:
                                continue
                            neig_poss_buffer = np.append(neig_poss_buffer, neig_poss_temp, 0)
            elif dimen_num == 2:
                for offset_0 in (0, -1, 1):
                    neig_poss_temp[:, 0] = neig_poss[:, 1] + offset_0 * periodic_len
                    for offset_1 in (0, -1, 1):
                        neig_poss_temp[:, 1] = neig_poss[:, 1] + offset_1 * periodic_len
                        if offset_0 == offset_1 == 0:
                            continue
                        neig_poss_buffer = np.append(neig_poss_buffer, neig_poss_temp, 0)
        else:
            neig_poss_buffer = neig_poss
        # no periodic boundaries
        KDTree = spatial.cKDTree(neig_poss_buffer)
        dists, neig_indices = KDTree.query(cen_poss, num_max, distance_upper_bound=dist_lim[1])
        dists = dists.astype(np.float32)
        neig_indices = neig_indices.astype(np.int32)
        if periodic_len:
            # re-normalized indices of buffer neighbors
            neig_indices[neig_indices != neig_poss_buffer.shape[0]] = \
                neig_indices[neig_indices != neig_poss_buffer.shape[0]] % neig_num
        # make null results negative
        neig_indices[neig_indices == neig_poss_buffer.shape[0]] = -neig_poss.shape[0] - 1
        # remove neighbors below distance minimum (including self)
        cis = ut_array.arange_length(cen_poss.shape[0])[dists[:, 0] < dist_lim[0]]
        for ci in cis:
            dists[ci, :-1] = dists[ci, 1:]
            neig_indices[ci, :-1] = neig_indices[ci, 1:]
        dists = dists[:, :-1]
        neig_indices = neig_indices[:, :-1]
        return dists, neig_indices

    def get_neig_kd_tree_ang(self, cen_poss, neig_poss, num_max=50, dist_lim=[0, 10],
                             ang_sphere=360, recursion_i=0):
        '''
        Get neighbor distances & indices from input neig_poss array, using kd-tree for angular
        separations.

        Import array of angular positions (object number x dimension number) around which to
        count neighbors, array of angular positions of neighbors,
        neighbor maximum number & distance range,
        periodicity length of sphere (if none, do not use periodic),
        whether funcion is being called recursively.
        '''
        if cen_poss.shape[1] != 2:
            raise ValueError(
                'cen_poss has %d dimensions, but kd-tree_ang requires 2-D angular posisions' %
                cen_poss.shape[1])
        if neig_poss.shape[1] != 2:
            raise ValueError(
                'neig_poss has %d dimensions, but kd-tree_ang needs 2-D angular posisions' %
                neig_poss.shape[1])
        if not recursion_i:
            num_max += 1    # add one to neighbor list count to deal with self selection
        if ang_sphere == 360:
            ang_scale = np.pi / 180
        elif ang_sphere == 2 * np.pi:
            ang_scale = 1
        dec_max = abs(np.concatenate((cen_poss[:, 1], neig_poss[:, 1]))).max()
        # maximum distance, in RA units, scaled to maximum dec, so should get everything
        dist_max = dist_lim[1] / np.cos(ang_scale * dec_max)
        cis = ut_array.arange_length(cen_poss.shape[0])
        nis = ut_array.arange_length(neig_poss.shape[0])
        neig_indices = np.zeros((cis.size, num_max), np.int32) - neig_poss.shape[0] - 1
        dists = np.zeros((cis.size, num_max), np.float32) + np.array(Inf, np.float32)
        # deal with those far from box edge
        cis_safe = cis    # farther than dist_max from edge
        nis_safe = nis    # farther than 2 * dist_max from edge
        cis_safe = ut_array.elements(cen_poss[:, 0], [dist_max, ang_sphere - dist_max], cis_safe)
        nis_safe = ut_array.elements(neig_poss[:, 0], [2 * dist_max, ang_sphere - 2 * dist_max],
                                     nis_safe)
        KDTree = spatial.cKDTree(neig_poss)
        dists[cis_safe], neig_indices[cis_safe] = KDTree.query(cen_poss[cis_safe], num_max,
                                                               distance_upper_bound=dist_max)
        if cis_safe.size != cis.size and nis_safe.size != nis.size:
            # treat those near an edge
            cis_edge = np.setdiff1d(cis, cis_safe)
            nis_edge = np.setdiff1d(nis, nis_safe)
            self.say('%d total | %d safe | %d near edge' %
                     (cis.size, cis_safe.size, cis_edge.size))
            # deal with those at edge by shifting positions by 2 * dist_max
            cen_poss_edge = cen_poss[cis_edge].copy()
            neig_poss_edge = neig_poss[nis_edge].copy()
            cen_poss_edge[(cen_poss_edge[:, 0] + dist_max > ang_sphere), 0] -= ang_sphere
            cen_poss_edge[:, 0] += 2 * dist_max
            neig_poss_edge[(neig_poss_edge[:, 0] + 2 * dist_max > ang_sphere), 0] -= ang_sphere
            neig_poss_edge[:, 0] += 2 * dist_max
            dists[cis_edge], neig_iis = self.kd_tree_ang(
                cen_poss_edge, neig_poss_edge, num_max, dist_lim, ang_sphere,
                recursion_i=recursion_i + 1)
            # fix neighbor id pointers
            for cii in xrange(cis_edge.size):
                niis_real = neig_iis[cii][dists[cis_edge[cii]] < Inf]
                neig_indices[cis_edge[cii]][:niis_real.size] = nis_edge[niis_real]
        # make null results negative
        neig_indices[neig_indices == neig_poss.shape[0]] = -neig_poss.shape[0] - 1
        if not recursion_i:
            if dists[:, -1].min() < Inf:
                self.say('! reached neighbor num_max = %d before compute true ang distance\n' %
                         (num_max - 1) + '  minimum distance(num_max) = %.2f, dist_max = %.2f' %
                         (dists[:, -1].min(), dist_max))
            # fix angular separations & remove self entries
            neig_iis_all = ut_array.arange_length(neig_indices[0])
            for ci in cis:
                neig_iis = neig_iis_all[neig_indices[ci] >= 0]
                if neig_iis.size:
                    dists[ci, neig_iis] = ut_coord.distance_ang(
                        'scalar', cen_poss[ci], neig_poss[neig_indices[ci, neig_iis]], ang_sphere)
                    # now that have computed real distances, excise else, fix others
                    dists[ci][dists[ci] < dist_lim[0]] = Inf
                    neig_indices[ci][dists[ci] < dist_lim[0]] = -neig_poss.shape[0] - 1
                    dists[ci][dists[ci] >= dist_lim[1]] = Inf
                    neig_indices[ci][dists[ci] >= dist_lim[1]] = -neig_poss.shape[0] - 1
                    if np.max(neig_indices[ci] >= 0):
                        # if neighbors left in range, sort distances by real angular separations
                        neig_iis_sort = np.argsort(dists[ci])
                        neig_indices[ci] = neig_indices[ci, neig_iis_sort]
                        dists[ci] = dists[ci, neig_iis_sort]
            # excise last entry (can be self)
            dists = dists[:, :-1]
            neig_indices = neig_indices[:, :-1]
        return dists, neig_indices


Neig = NeigClass()
