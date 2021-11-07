# coding=utf-8
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from pickle import dump, load
import nibabel as nib
from sklearn.neighbors import KDTree
import sys
from skimage.morphology import skeletonize_3d, medial_axis, remove_small_holes
import time
import scipy
import csv
import os
from unet_core.io import ImageReader
import json
from unet_core.utils.data_utils import im_smooth
from itertools import product

class Point(object):
    def __init__(self, coords):
        self.x = int(coords[0])
        self.y = int(coords[1])
        self.z = int(coords[2])

        self.diameter = None

    def dist_to(self, point, pix_dim=None):
        if pix_dim is None:
            pix_dim = (1, 1, 1)

        dist = (pix_dim[0] * (self.x - point.x)) ** 2 + (pix_dim[1] * (self.y - point.y)) ** 2 + (pix_dim[2] * (
            self.z - point.z)) ** 2
        dist = np.sqrt(float(dist))
        return dist

    def to_ind(self, dims):
        return np.ravel_multi_index((self.x, self.y, self.z), dims)

    def __str__(self):
        return "({0}, {1}, {2})".format(self.x, self.y, self.z)

    def add_to_point(self, point):

        self.x += point.x
        self.x += point.y
        self.x += point.z

        return self

    def copy(self):
        p = Point([self.x, self.y, self.z])
        p.diameter = self.diameter
        return p

class Node(object):
    def __init__(self, location):
        self.point = location
        self.x = location.x
        self.y = location.y
        self.z = location.z

    def dist_to(self, x, pix_dim=None):
        return self.point.dist_to(x, pix_dim=pix_dim)


class Branch(object):
    def __init__(self, input_points=None):
        self.points = []
        self.performed_analysis = False
        self.length = None
        self.tortuosity = None
        self.parent = None
        self.children = []
        self.rejoin = False
        self.pix_dim = None
        self.diameter = None
        self.param_dict = {'diameter': self.diameter, 'length': self.length, 'tortuosity': self.tortuosity}

        if input_points is not None:
            if isinstance(input_points, list):
                self.append_points(input_points)
            else:
                self.append_point(input_points)

    def __getitem__(self, key):
        return self.param_dict[key]

    def append_point(self, point):
        self.points.append(point)

    def append_points(self, points, prepend=False, reverse=False):

        new_points = [p.copy() for p in points]

        if reverse:
            new_points.reverse()

        if prepend:
            new_points.extend(self.points)
            self.points = new_points
        else:
            self.points.extend(points)

        return self

    def append_coords(self, coords):
        self.append_point(Point(coords))

    def append_branch(self, branch, check_continuity=True, verbose=False, enforce=False):

        reverse = False
        prepend = False

        if check_continuity and len(self.points) > 0 and len(branch.points) > 0:
            dist_ac = self.points[0].dist_to(branch.points[0])
            dist_bc = self.points[-1].dist_to(branch.points[0])
            dist_ad = self.points[0].dist_to(branch.points[-1])
            dist_bd = self.points[-1].dist_to(branch.points[-1])

            dists = [dist_ac, dist_bc, dist_ad, dist_bd]

            if enforce is False:
                if np.argmin(dists) == 0:
                    reverse=True
                    prepend=True
                elif np.argmin(dists) == 1:
                    reverse=False
                    prepend=False
                elif np.argmin(dists) == 2:
                    reverse=False
                    prepend=True
                elif np.argmin(dists) == 3:
                    reverse=True
                    prepend=False

            else:
                assert dist_bc > 2, "ERROR: Appended branches are not connected"

        self.append_points(branch.points, reverse=reverse, prepend=prepend)

        return self

    def add_child(self, child):
        self.children.append(child)

    def analyse(self, pix_dim=None):
        if pix_dim is None:
            pix_dim = [1, 1, 1]

        self.pix_dim = pix_dim
        deltas = []
        for i, j in zip(self.points[:-1], self.points[1:]):
            d = i.dist_to(j, pix_dim=pix_dim)
            deltas.append(d)

        self.length = max(np.sum(deltas), 1.0)
        diameters = [p.diameter for p in self.points if p.diameter is not None]
        self.diameter = np.mean(diameters)
        self.soam = self.get_SOAM()
        self.soam_per_length = float(self.soam) / self.length
        self.chord = self.get_chord()
        self.clr = float(self.chord) / self.length
        self.neg_clr = 1-self.clr
        self.c2l = self.neg_clr / self.chord

        self.direction = self.get_direction()

        if self.chord - self.length > 1e-4:
            raise ValueError('Chord cannot exceed length')

        self.param_dict = {'diameter': self.diameter, 'length': self.length, 'tortuosity': self.tortuosity,
                           'soam': self.soam, 'soam_per_length': self.soam_per_length, 'chord': self.chord, 'clr': self.clr,
                           'neg_clr': self.neg_clr, 'c2l': self.c2l}

    def get_points(self):
        p = np.empty((len(self.points), 3))

        for i in range(len(self.points)):
            p[i, 0] = self.points[i].x
            p[i, 1] = self.points[i].y
            p[i, 2] = self.points[i].z

        return p

    def get_smoothed_points(self):
        p = np.empty((len(self.points), 3))

        for i in range(len(self.points)):
            p[i, 0] = self.points[i].x*self.pix_dim[0]
            p[i, 1] = self.points[i].y*self.pix_dim[1]
            p[i, 2] = self.points[i].z*self.pix_dim[2]

        border_mode = 'nearest'
        sigma = 3
        p_start = p[0,:].copy()
        p_end = p[-1,:].copy()

        p[:, 0] = scipy.ndimage.filters.gaussian_filter(p[:, 0], sigma=sigma, mode=border_mode)
        p[:, 1] = scipy.ndimage.filters.gaussian_filter(p[:, 1], sigma=sigma, mode=border_mode)
        p[:, 2] = scipy.ndimage.filters.gaussian_filter(p[:, 2], sigma=sigma, mode=border_mode)

        p = np.vstack((p_start, p, p_end))

        return p

    def get_SOAM(self):

        if len(self.points) < 5:
            return 0

        thetas = []
        points = self.get_smoothed_points()

        for i in range(len(points) - 2):
            p1 = points[i,:]
            p2 = points[i+1,:]
            p3 = points[i+2,:]

            v1 = p2-p1
            v2 = p3-p2

            if (v1 == [0, 0, 0]).all() or (v2 == [0, 0, 0]).all():
                continue

            cos_theta = np.dot(v1, v2)
            cos_theta /= np.linalg.norm(v1)
            cos_theta /= np.linalg.norm(v2)

            if np.abs(cos_theta - 1) < 1e-5:
                cos_theta = 1

            if np.abs(cos_theta + 1) < 1e-5:
                cos_theta = -1

            theta = np.arccos(cos_theta)
            if np.isnan(theta):
                raise ValueError("ERROR: Invalid theta value")

            thetas.append(theta)

        soam = np.sum(thetas)

        return soam

    def get_direction(self):
        p1 = self.points[0]
        p2 = self.points[-1]
        d = np.zeros(3)
        d[0] = p2.x - p1.x
        d[1] = p2.y - p1.y
        d[2] = p2.z - p1.z
        l = max(np.linalg.norm(d), 1e-5)
        d /= l
        return d

    def get_chord(self):
        chord = self.points[0].dist_to(self.points[-1], pix_dim=self.pix_dim)
        chord = max(chord, 1)
        return chord


class Skeleton(object):
    def __init__(self):
        self.components = []
        self.pix_dim = None

    def add_component(self, component):
        component.pix_dim = self.pix_dim
        self.components.append(component)

    def analyse(self):
        for c in self.components:
            c.analyse()

    @property
    def num_branches(self):
        n_branches = 0

        for c in self.components:
            n_branches += c.num_branches

        return n_branches

    @property
    def num_nodes(self):
        n_nodes = 0

        for c in self.components:
            n_nodes += c.num_nodes

        return n_nodes

    @property
    def num_components(self):
        return len(self.components)

    def skeleton_branch_iter(self):

        for c in self.components:
            for u,v,b in c.branch_iter():
                yield u,v,b

    def skeleton_node_iter(self):

        for c in self.components:
            for n in c.node_iter():
                yield n

class SkeletonComponent(object):
    def __init__(self):
        self.adjacency = np.zeros(1)
        self.graph = nx.MultiGraph()
        self.pix_dim = None

    def analyse(self):
        for n1, n2, b in self.graph.edges(data=True):
            b['branch'].analyse(pix_dim=self.pix_dim)

    def add_branch(self, node1, node2, branch=None):

        if branch is not None:
            if branch.points[0].dist_to(node1.point) < branch.points[-1].dist_to(node1.point):
                branch.points = [node1.point] + branch.points + [node2.point]
            else:
                branch.points = [node2.point] + branch.points + [node1.point]

        else:
            branch = Branch()
            npoints = 10
            pp = np.zeros((3, npoints))
            pp[0, :] = np.linspace(node1.point.x, node2.point.x, npoints)
            pp[1, :] = np.linspace(node1.point.y, node2.point.y, npoints)
            pp[2, :] = np.linspace(node1.point.z, node2.point.z, npoints)
            for i in range(pp.shape[1]):
                branch.append_coords(pp[:, i])

        if node1.dist_to(branch.points[0], pix_dim=self.pix_dim) > node1.dist_to(branch.points[-1], pix_dim=self.pix_dim):
            branch.points.reverse()

        self.graph.add_edge(node1, node2, attr_dict={"branch": branch})

    def add_node(self, node):
        self.graph.add_node(node)

    def remove_node(self, node, check_edges=True, enforce=False):

        if check_edges:
            n_edges = self.graph.degree(node)

            if n_edges > 1 and check_edges is True:
                print("WARNING: Attempting to remove node with more than 1 edge, this may cause discontinuity")

                if enforce:
                    raise AssertionError

        self.graph.remove_node(node)

    def merge_branches(self, branch1, branch2):
        branch1.append_branch(branch2)
        return branch1

    def expand_branch(self, u, v, b, max_length=10):

        if b.points[0].dist_to(v) < b.points[0].dist_to(u):
            b.points.reverse()

        self.graph.remove_edge(u, v)

        current_node = u
        j=0

        for i in range(0, len(b.points)-2*max_length, max_length):
            next_node = Node(b.points[i+max_length])
            branch = Branch(b.points[i:i+max_length])
            self.add_branch(current_node, next_node, branch=branch)
            current_node = next_node
            j = i + max_length

        next_node = v
        branch = Branch(b.points[j:])
        self.add_branch(current_node, next_node, branch=branch)


    def remove_node_merge_branches(self, node):

        neighbours = self.graph.neighbors(node)

        if len(neighbours) != 2:
            return

        n1 = neighbours[0]
        n2 = neighbours[1]

        e1 = list(self.graph.get_edge_data(node, n1).items())[0][1]['branch']
        e2 = list(self.graph.get_edge_data(node, n2).items())[0][1]['branch']
        """Determine ordering of neighbours for robust merging"""

        e1_start = e1.points[0]
        e1_end = e1.points[-1]
        e2_start = e2.points[0]
        e2_end = e2.points[-1]

        if e1_end.dist_to(e2_start) < 4:
            new_b = e1.append_points(e2.points)
        elif e2_end.dist_to(e1_start) < 4:
            new_b = e2.append_points(e1.points)
        elif e1_end.dist_to(e2_end) < 4:
            new_b = e1.append_points(e2.points[::-1])
        elif e1_start.dist_to(e2_start) < 4:
            new_b = e1
            new_b.points = new_b.points[::-1]
            new_b.append_points(e2.points)
        else:
            raise ValueError("Attempting to merge disjoint branches")

        # self.remove_node(node)
        self.graph.remove_edge(node, n1)
        self.graph.remove_edge(node, n2)
        self.add_branch(n1, n2, new_b)


    def collapse_branch(self, branch):

        n1_merge = None
        n2_merge = None

        for b in self.graph.edges_iter(data=True):
            if b[2]['branch'] == branch:
                n1_merge = b[0]
                n2_merge = b[1]
                break

        if n1_merge is None or n2_merge is None:
            raise ValueError("Attempted to collapse a branch which was not in the graph.")

        nodes = [n1_merge, n2_merge]

        new_point = Point([(n1_merge.point.x + n2_merge.point.x)/2,
                           (n1_merge.point.y + n2_merge.point.y)/2,
                           (n1_merge.point.z + n2_merge.point.z)/2])

        new_node = Node(new_point)

        self.graph.add_node(new_node)  # Add the 'merged' node

        for n1, n2, data in self.graph.edges(data=True):
            # For all edges related to one of the nodes to merge,
            # make an edge going to or coming from the `new gene`.
            if n1 in nodes:
                self.add_branch(new_node, n2, data['branch'])
            elif n2 in nodes:
                self.add_branch(n1, new_node, data['branch'])

        for n in nodes:  # remove the merged nodes
            if self.graph.has_node(n):
                self.graph.remove_node(n)


    def remove_self_loops(self):

        for n1, n2, data in self.graph.edges(data=True):
            if n1 is n2:
                while self.graph.has_edge(n1,n2):
                    self.graph.remove_edge(n1,n2)

    @property
    def num_branches(self):
        return self.graph.number_of_edges()

    @property
    def num_nodes(self):
        return self.graph.number_of_nodes()

    def branch_iter(self):
        for u,v,b in list(self.graph.edges_iter(data=True)):
            branch = b['branch']
            yield u,v,branch

    def node_iter(self):
        for n in self.graph.nodes_iter():
            yield n


class VesselTree(object):
    """Class to analyse an image containing vessel structures. Image will be preprocessed and passed to a
    SkeletonTracker object. """

    def __init__(self, input_image=None, do_skeletonize=True, skel_method='thinning',
                 min_branch_length=10, pix_dim=None, min_object_size=0, image_dimensions=None, skeleton_path=None):
        self.skeleton_tracker = SkeletonTracker(min_branch_length=min_branch_length,
                                                pix_dim=pix_dim, min_object_size=min_object_size)
        if input_image is not None:
            if input_image.ndim > 3:
                self.skeleton_image = np.squeeze(input_image)
            else:
                self.skeleton_image = input_image
            self.distance_image = self.skeleton_image
        else:
            self.skeleton_image = None
            self.distance_image = None

        self.skel_method = skel_method
        self.do_skeletonize = do_skeletonize
        self.image_dimensions = image_dimensions

        if pix_dim is None:
            self.pix_dim = [1, 1, 1]
        else:
            self.pix_dim = pix_dim

        if skeleton_path is not None:
            self.load_skeleton(skeleton_path)

    def skeletonize(self, input_image):

        self.skeletonize_time = time.time()
        assert self.skel_method in ["thinning", "medial_axis"], \
            "Invalid skeletonization method {0} [choose 'thinning' or 'medial_axis']".format(self.skel_method)

        if self.skel_method == "thinning":
            skeleton_image = skeletonize_3d(input_image > 0)
        elif self.skel_method == "medial_axis":
            skeleton_image = medial_axis(input_image)

        self.skeletonize_time = time.time() - self.skeletonize_time

        return skeleton_image

    def analyse_vessels(self, debug=None, verbose=True, compute_distance=True):

        self.start_time = time.time()

        if self.skeleton_image is not None:
            if self.skeleton_image.ndim > 2:
                self.max_intensity = np.sum(self.skeleton_image, axis=2)

        if debug is not None:
            self.skeleton_tracker.debug = debug

        self.skeleton_image = remove_small_holes(self.skeleton_image > 0, in_place=True, min_size=500)

        if compute_distance:
            self.distance_map = self.get_dist_map(self.distance_image, verbose=verbose)
        else:
            self.distance_map = self.skeleton_image
            self.distance_scaling = 1
            self.distance_map_time = 0

        if self.do_skeletonize is True:
            if verbose:
                print("Skeletonizing image...")
            self.skeleton_image = self.skeletonize(self.skeleton_image)
        else:
            self.skeletonize_time = 0

        self.tracking_time = time.time()
        self.skeleton = self.skeleton_tracker.analyse_skeleton(self.skeleton_image, self.distance_map,
                                                               self.distance_scaling, verbose=verbose)
        self.tracking_time = time.time() - self.tracking_time

        self.image_dimensions = self.skeleton_tracker.image_dimensions

        if verbose:
            self.post_analysis_report()

        return self.skeleton


    def make_skeleton_overlay(self, intensity_image=None):

        if intensity_image is not None:
            max_int_im = np.sum(intensity_image, axis=2)
        else:
            max_int_im = self.max_intensity

        skeleton_max_int = np.sum(self.make_skeleton_image(), axis=2)

        max_int_im = max_int_im.astype(float)

        r = max_int_im.copy()
        g = max_int_im.copy()
        b = max_int_im.copy()

        r /= (np.max(r))
        g /= (np.max(g))
        b /= (np.max(b))

        r *= 0

        g[skeleton_max_int > 0] = 0
        b[skeleton_max_int > 0] = 0
        r[skeleton_max_int > 0] = 1

        for c in self.skeleton.components:
            for n in c.node_iter():
                if len(c.graph.neighbors(n)) > 1:
                    p = n.point
                    g[p.x-1, p.y-1] = 1
                elif len(c.graph.neighbors(n)) == 1:
                    p = n.point
                    b[p.x-1, p.y-1] = 1

        im = np.stack([r, g, b], axis=2)

        return im


    def make_skeleton_image(self):
        skel_im = np.zeros(self.image_dimensions)

        for u,v,b in self.skeleton.skeleton_branch_iter():
            p = b.get_smoothed_points()
            for i in range(p.shape[0]):
                px = int(p[i, 0]/self.pix_dim[0] - 1)
                py = int(p[i, 1]/self.pix_dim[1] - 1)
                pz = int(p[i, 2]/self.pix_dim[2] - 1)
                skel_im[px, py, pz] = 1

        return skel_im

    def branch_iter(self):
        for u,v,b in self.skeleton.skeleton_branch_iter():
            yield u,v,b

    def node_iter(self):
        for n in self.skeleton.skeleton_node_iter():
            yield n

    def post_analysis_report(self):
        print("Vessel Analysis completed successfully...")
        print("Distance Map calculation took %ds" % self.distance_map_time)
        print("Skeletonization took %ds" % self.skeletonize_time)
        print("Vessel Tracking and analysis took %ds" % self.tracking_time)

    def get_dist_map(self, input_image, verbose=True):

        self.distance_map_time = time.time()
        if verbose:
            print("Computing distance map...")
        dist_map = np.zeros(input_image.shape, dtype=np.uint16)
        self.distance_scaling = 100
        for i in range(dist_map.shape[2]):
            dist_map[:, :, i] = (scipy.ndimage.morphology.distance_transform_edt(input_image[:, :, i], [self.pix_dim[0],
                                                                                                        self.pix_dim[
                                                                                                            1]]) * self.distance_scaling).astype(
                np.uint16)
        if verbose:
            print("Filtering distance map...")
        self.distance_map_time = time.time() - self.distance_map_time

        dist_map = scipy.ndimage.filters.maximum_filter(dist_map, [5, 5, 3])

        return dist_map

    def subdivide_skeleton(self, max_branch_length=10):

        for c in self.skeleton.components:
            for u, v, b in c.branch_iter():
                if len(b.points) > 20:
                    c.expand_branch(u, v, b, max_length=max_branch_length)

        self.skeleton.analyse()


    def plot_vessels(self, metric=None, metric_scaling=100, cmap='viridis', width_metric='diameter', width_scaling=0.5,
                     plot_nodes=True, write_location=None, subsample=1, threshold=None):

        c_m = plt.get_cmap(cmap).colors
        origin = np.array([0,0,0]).astype(float)
        node_count = 0

        if threshold is not None:
            metric_scaling = 255.0

        try:
            import pyqtgraph as pg
            import pyqtgraph.opengl as gl
            use_openGL = True
            q_app = pg.mkQApp()
        except ImportError:
            use_openGL = False

        use_openGL = False

        if use_openGL:
            w = gl.GLViewWidget()
            for c in self.skeleton.components:
                for u,v,b in c.branch_iter():
                    p = b.get_smoothed_points()
                    p[:, 0] /= self.image_dimensions[0] * 0.05
                    p[:, 1] /= self.image_dimensions[0] * 0.05
                    p[:, 2] /= self.image_dimensions[0] * 0.05
                    p -= 10

                    if len(p) > 10:
                        p = p[::subsample]

                    if metric is None:
                        C = pg.glColor('r')
                        width = 3
                    else:
                        assert metric in list(b.param_dict.keys()), "Invalid metric specified"

                        m = b.param_dict[metric] * metric_scaling
                        m = int(sorted((0, m, 255))[1])
                        C = c_m[m]
                        C = (C[0], C[1], C[2], 1.0)
                        # C = pg.glColor('r')
                        width = b.param_dict[width_metric] * width_scaling
                        width = int(sorted((1, width, 255))[1])

                    line = gl.GLLinePlotItem(pos=p, width=width, color=C)
                    w.addItem(line)

                for n in c.graph.nodes():
                    if c.graph.degree(n) > 0:
                        p = np.array([n.x, n.y, n.z]).astype(float)
                        p[0] /= self.image_dimensions[0] * 0.05
                        p[1] /= self.image_dimensions[0] * 0.05
                        p[2] /= self.image_dimensions[0] * 0.05
                        p -= 10
                        C = pg.glColor('w')
                        #print(p)
                        point = gl.GLScatterPlotItem(pos=p, color=C, size=1)
                        if plot_nodes:
                            w.addItem(point)

                        origin += p
                        node_count += 1

            g = gl.GLGridItem()
            w.setCameraPosition(pos=origin/node_count, distance=20, elevation=90, azimuth=0)
            w.addItem(g)
            w.show()

            if write_location is not None:
                d = w.renderToArray((3000, 3000))
                q_app.processEvents()
                pg.makeQImage(d).save(write_location)
            else:
                pg.QtGui.QApplication.exec_()

        else:
            fig = plt.figure()
            fig.patch.set_facecolor('black')
            Z = [[0,0],[0,0]]
            levels = np.linspace(0,255./metric_scaling,10)
            CS3 = plt.contourf(Z, levels, cmap=plt.get_cmap(cmap))
            plt.clf()
            ax = fig.add_subplot(111)
            # ax = fig.add_subplot(111, projection='3d')
            for c in self.skeleton.components:
                for n1, n2, b in c.branch_iter():

                    if threshold is not None:
                        m = b.param_dict[metric] > threshold
                        m *= 255.0
                    else:
                        m = b.param_dict[metric] * metric_scaling

                    m = int(sorted((0, m, 255))[1])
                    C = c_m[m]
                    C = (C[0], C[1], C[2], 1.0)
                    width = b.param_dict[width_metric] * width_scaling / 10
                    if width is not None and not np.isnan(width):
                        width = sorted((0.1, width, 255))[1]
                    else:
                        width = 5

                    p = b.get_smoothed_points()
                    #print(p)
                    pl = ax.plot(-p[:, 1], p[:, 0], color=C, linewidth=width, alpha=0.5) #changed by BJS
                    #pl = ax.plot(p[:, 0], p[:, 1], color=C, linewidth=width, alpha=0.5)
                    # pl = ax.plot(p[:, 0], p[:, 1], p[:, 2], color=C, linewidth=width, alpha=0.5)

                for n in c.graph.nodes():
                    if c.graph.degree(n) > 0:
                        # ax.scatter([n.x], [n.y], [n.z], marker='o')
                        pass

            # ax.set_aspect('equal')
            # ax.view_init(elev=90., azim=0)
            ax.patch.set_facecolor('black')
            ax.set_facecolor('black')
            ax.set_axis_off()
            cb = plt.colorbar(CS3)
            plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()
            cb.ax.yaxis.set_tick_params(labelcolor='w')


            if write_location is not None:
                plt.savefig(write_location, dpi=1000, facecolor=fig.get_facecolor())
            else:
                plt.show()

            plt.clf()

    def add_perfusion_information(self, image_path, perfusion_channel=None, image_series=1):

        with ImageReader(image_path, image_series=image_series, keep_vm_open=True) as reader:

            perf = reader.get_tile((0, 0), (reader.get_dims()[0], reader.get_dims()[1]), channels=perfusion_channel)
            sigma=3
            perf = scipy.ndimage.median_filter(np.squeeze(perf), [sigma,sigma,sigma])

            for u, v, b in self.skeleton.skeleton_branch_iter():

                branch_perf = []

                for p in b.points:
                    branch_perf.append(perf[p.x-1, p.y-1, p.z-1])

                b.perfusion_max = np.max(branch_perf)
                b.perfusion_mean = np.mean(branch_perf)
                b.perfusion_median = np.median(branch_perf)
                b.perfusion_min = np.min(branch_perf)
                b.param_dict['perfusion_max'] = b.perfusion_max
                b.param_dict['perfusion_mean'] = b.perfusion_mean
                b.param_dict['perfusion_median'] = b.perfusion_median
                b.param_dict['perfusion_min'] = b.perfusion_min

    def compute_node_density(self, output_name=None, normalize=False, vmin=None, vmax=None, save_data=False):

        from scipy.stats import gaussian_kde

        data_points = []

        for c in self.skeleton.components:
            for n in c.graph.nodes_iter():
                data_points.append([n.point.x, n.point.y])

        data_points = np.stack(data_points)
        kde = gaussian_kde(data_points.transpose())
        kde.set_bandwidth()

        nb_points = data_points.shape[0]

        x_d = self.image_dimensions[0]
        y_d = self.image_dimensions[1]
        x_step = int(float(x_d)/500)
        y_step = int(float(y_d)/500)
        x_step = max(x_step,0)
        y_step = max(y_step,0)

        x, y = np.mgrid[0:x_d:x_step, 0:y_d:y_step]
        test_data = np.vstack((x.ravel(), y.ravel()))

        density = kde(test_data)

        if normalize is False:
            density *= nb_points
            density /= self.pix_dim[0]
            density /= self.pix_dim[1]
            title = 'Branch point density (branchpoints/micron^2)'
        else:
            title = 'Branch point probability map'

        ax = sns.heatmap(density.reshape(x.shape), cbar=True, xticklabels=False, yticklabels=False, square=True,
                    cmap=mpl.colors.ListedColormap(sns.color_palette("Blues", n_colors=100)), vmin=vmin, vmax=vmax)

        plt.title(title)

        if output_name is False:
            plt.show()
        else:
            plt.savefig(output_name)

        plt.clf()


        if save_data:
            output_name = os.path.splitext(output_name)
            data_path = output_name[0] + '.csv'
            with open(data_path, 'w') as file:
                writer = csv.writer(file, delimiter=',')
                for r in list(density.flatten()):
                    writer.writerow([float(r),])

    def compute_directional_coherence(self, output_name, radius=200, save_data=False, nb_pixels=80):
        X = []
        y = []

        for c in self.skeleton.components:
            for u, v, b in c.branch_iter():
                for p in b.points[:]:
                    X.append([p.x, p.y])
                    y.append(b.direction[:2])

        X = np.stack(X)

        X = X.astype(float)
        kd = KDTree(X)

        x_d = self.image_dimensions[0]
        y_d = self.image_dimensions[1]

        nb_pixels = 80

        x_step = int(float(x_d) / nb_pixels)
        y_step = int(float(y_d) / nb_pixels)
        x_step = max(x_step, 0)
        y_step = max(y_step, 0)

        x_g, y_g = np.mgrid[0:x_d:x_step, 0:y_d:y_step]
        p_g = np.vstack((x_g.ravel(), y_g.ravel())).transpose()

        coherence = []
        direction = []
        sigma = radius/2

        for i in range(p_g.shape[0]):

            dyad = np.zeros((2,2))

            idx, dist = kd.query_radius(p_g[i, :].reshape((1, -1)), r=radius, return_distance=True)

            values = [y[i] for i in idx[0]]

            if len(values) == 0:
                value = 0
            else:
                for v, d in zip(values, dist[0]):
                    weight = np.exp(-d ** 2 / (2 * sigma ** 2))
                    dyad += (weight * np.outer(v, v))

            e, v = np.linalg.eig(dyad)
            e_max = max(np.abs(e))
            e_min = min(np.abs(e))
            c = (e_max - e_min)/max(e_max + e_min, 1e-5)
            theta = np.arctan2(v[0][0], v[0][1])
            if theta < 0:
                theta += np.pi
            if theta > np.pi:
                theta -= np.pi

            coherence.append(c)
            direction.append(theta)

        direction = np.array(direction)
        coherence = np.array(coherence)

        output_name = os.path.splitext(output_name)

        f1 = sns.heatmap(direction.reshape(x_g.shape),
                         cmap=mpl.colors.ListedColormap(sns.husl_palette(100, l=0.5, h=1.0, s=1.0)),
                         cbar=True, xticklabels=False, yticklabels=False, square=True, vmin=0, vmax=np.pi)

        plt.title("Local vessel direction (Theta)")

        if output_name is False:
            plt.show()
        else:
            plt.savefig(output_name[0] + '_theta' + output_name[1])
            if save_data:
                data_path = output_name[0] + '_theta' + '.csv'
                with open(data_path, 'w') as file:
                    writer = csv.writer(file, delimiter=',')
                    for r in list(direction.flatten()):
                        writer.writerow([float(r),])

        plt.clf()

        f2 = sns.heatmap(coherence.reshape(x_g.shape), cbar=True, xticklabels=False, yticklabels=False, vmin=0, vmax=1,
                         square=True, cmap=mpl.colors.ListedColormap(sns.color_palette("Blues", n_colors=100)) )
        plt.title("Local directional coherence")

        if output_name is False:
            plt.show()
        else:
            plt.savefig(output_name[0] + '_coherence' + output_name[1])
            if save_data:
                data_path = output_name[0] + '_coherence' + '.csv'
                with open(data_path, 'w') as file:
                    writer = csv.writer(file, delimiter=',')
                    for r in list(coherence.flatten()):
                        writer.writerow([float(r),])

        plt.clf()

        return direction.reshape(x_g.shape), coherence.reshape(x_g.shape)


    def compute_metric_average(self, metric, output_name=None, units=None, radius=200, threshold=None, vmin=None, vmax=None, save_data=False):

        X = []
        y = []

        for c in self.skeleton.components:
            for u, v, b in c.branch_iter():
                for p in b.points[::]:
                    X.append([p.x, p.y])
                    v = b.param_dict[metric]
                    if threshold is not None:
                        v = int(v > threshold)
                    y.append(v)

        str_dict = {}

        if units is not None:
            str_dict['units'] = '(' + units + ')'
        else:
            str_dict['units'] = ''

        str_dict['metric'] = metric

        X = np.stack(X)
        y = np.array(y)

        X = X.astype(float)
        kd = KDTree(X)

        nb_pixels = 100

        x_d = self.image_dimensions[0]
        y_d = self.image_dimensions[1]
        x_step = int(float(x_d) / nb_pixels)
        y_step = int(float(y_d) / nb_pixels)
        x_step = max(x_step, 0)
        y_step = max(y_step, 0)

        x_g, y_g = np.mgrid[0:x_d:x_step, 0:y_d:y_step]
        p_g = np.vstack((x_g.ravel(), y_g.ravel())).transpose()

        density = []
        variance = []
        r = radius
        sigma = radius/4

        for i in range(p_g.shape[0]):

            idx, dist = kd.query_radius(p_g[i, :].reshape((1, -1)), r=r, return_distance=True)

            values = y[idx[0]]

            if len(values) < 3:
                value = 0
                var_value = 0
            else:
                weights = np.exp(-dist[0]**2/(2*sigma**2))
                value = np.sum(values*weights)
                value /= np.sum(weights)

                if threshold is not None:
                    value = np.mean(values)
                else:
                    value = np.median(values)

                var_value = np.var(values)

            density.append(value)
            variance.append(var_value)

        output_name = os.path.splitext(output_name)

        density = np.array(density)
        f = sns.heatmap(density.reshape(x_g.shape),
                        cbar=True,
                        xticklabels=False,
                        yticklabels=False,
                        square=True,
                        cmap=mpl.colors.ListedColormap(sns.color_palette("Blues", n_colors=100)),
                        vmin=vmin,
                        vmax=vmax)
        plt.title("Local median of {metric} {units}".format(**str_dict))


        if output_name is False:
            plt.show()
        else:
            plt.savefig(output_name[0] + '_median' + output_name[1])

        plt.clf()
        plt.close()

        variance = np.array(variance)
        f = sns.heatmap(variance.reshape(x_g.shape),
                        cbar=True,
                        xticklabels=False,
                        yticklabels=False,
                        square=True,
                        cmap=mpl.colors.ListedColormap(sns.color_palette("Blues", n_colors=100)),
                        vmin=vmin,
                        vmax=vmax)

        plt.title("Local variance of {metric} {units}".format(**str_dict))


        if output_name is False:
            plt.show()
        else:
            plt.savefig(output_name[0] + '_variance' + output_name[1])

        plt.clf()
        plt.close()

        if save_data:
            data_path = output_name[0] + '.csv'
            with open(data_path, 'w') as file:
                writer = csv.writer(file, delimiter=',')
                for r in list(density.flatten()):
                    writer.writerow([float(r),])


    def plot_assignment(self, v2, assignment, show=False, output_name=None):

        fig = plt.figure()
        fig.patch.set_facecolor('black')
        plt.clf()
        # ax = fig.add_subplot(111)
        ax = fig.add_subplot(111, projection='3d')
        for c in self.skeleton.components:
            for n1, n2, b in c.branch_iter():

                C = (0, 0, 1, 1.0)
                width = b.param_dict['diameter']
                if width is not None and not np.isnan(width):
                    width = sorted((0.1, width, 255))[1]
                else:
                    width = 5

                p = b.get_smoothed_points()
                # Flipped axes for alignment
                pl = ax.plot(p[:, 0], p[:, 1], p[:, 2], color=C, linewidth=2*width, alpha=0.5)


        for c in v2.skeleton.components:
            for n1, n2, b in c.branch_iter():

                C = (0, 1, 0, 1.0)
                width = b.param_dict['diameter']
                if width is not None and not np.isnan(width):
                    width = sorted((0.1, width, 255))[1]
                else:
                    width = 5

                p = b.get_smoothed_points()
                # Flipped axes for alignment
                pl = ax.plot(p[:, 0], p[:, 1], p[:, 2], color=C, linewidth=width, alpha=1)

        for i, n1 in enumerate(self.skeleton.skeleton_node_iter()):
            if n1.color > 0 and assignment is not None:
                n2 = n1.match
                plt.plot([n1.x, n2.x], [n1.y, n2.y], [n1.z, n2.z], 'r')


        #add invisible corner points
        x = [0,self.image_dimensions[0]]
        y = [0,self.image_dimensions[1]]
        z = [0,self.image_dimensions[-1]]

        for _x, _y, _z in product(x, y, z):
            ax.plot([_x,], [_y,], [_z,], 'w')

        # ax.set_aspect('equal')
        ax.view_init(elev=90., azim=0)
        ax.patch.set_facecolor('black')
        # ax.set_axis_bgcolor('black')
        ax.set_axis_off()
        plt.gca().invert_yaxis()

        if output_name is not None:
            plt.savefig(output_name)

        if show:
            plt.show()


    def edge_agreement(self, v2, plot_agreement=False, iterations=1, assignment=None):

        if self.skeleton.num_nodes == 0 or v2.skeleton.num_nodes == 0:
            return 0

        if assignment is None:
            assignment, acc = self.distance_to_fgm_cpp(v2, symmetric=False, iterations=iterations)

        if len(assignment) == 0:
            return 0

        n_dict_1 = {}
        n_dict_2 = {}
        i_dict_1 = {}
        i_dict_2 = {}

        graphs = [c.graph for c in v2.skeleton.components]

        graph = nx.MultiGraph()

        for g in graphs:
            graph = nx.compose(graph, g)

        for i, n in enumerate(self.skeleton.skeleton_node_iter()):
            n_dict_1[n] = i
            i_dict_1[i] = n

        for i, n in enumerate(v2.skeleton.skeleton_node_iter()):
            n_dict_2[n] = i
            i_dict_2[i] = n

        iter = 0
        success = 0

        for u, v, b in self.skeleton.skeleton_branch_iter():

            a_1 = n_dict_1[u]
            a_1 = np.where(assignment[a_1])
            a_1 = a_1[0]
            iter += 1
            if len(a_1) > 0:
                a_1 = i_dict_2[int(a_1[0])]

                a_2 = n_dict_1[v]
                a_2 = np.where(assignment[a_2])
                a_2 = a_2[0]
                if len(a_2) > 0:
                    a_2 = i_dict_2[int(a_2[0])]
                    if graph.has_edge(a_1, a_2):
                        success += 1

        return float(success)/float(iter)

    def topology_compare(self, v2, nodes_per_subset=5, n_repeats = 10, assignment=None, fgm_iterations=1, symmetric=False):

        if self.skeleton.num_nodes == 0 or v2.skeleton.num_nodes == 0:
            return 0

        if assignment is None:
            assignment, acc = self.distance_to_fgm_cpp(v2, symmetric=False, iterations=fgm_iterations)

        if len(assignment) == 0:
            return 0

        n_dict_1 = {}
        n_dict_2 = {}
        i_dict_1 = {}
        i_dict_2 = {}

        graphs_1 = [c.graph for c in self.skeleton.components]
        graphs_2 = [c.graph for c in v2.skeleton.components]

        graph_1 = nx.MultiGraph()
        graph_2 = nx.MultiGraph()

        for g in graphs_1:
            graph_1 = nx.compose(graph_1, g)

        for g in graphs_2:
            graph_2 = nx.compose(graph_2, g)

        for i, n in enumerate(self.skeleton.skeleton_node_iter()):
            n_dict_1[n] = i
            i_dict_1[i] = n

        for i, n in enumerate(v2.skeleton.skeleton_node_iter()):
            n_dict_2[n] = i
            i_dict_2[i] = n

        iter = 0
        success = 0

        # if assignment.shape[0] > assignment.shape[1]:
        #     graph_1, graph_2 = graph_2, graph_1
        #     n_dict_1, n_dict_2 = n_dict_2, n_dict_1
        #     i_dict_1, i_dict_2 = i_dict_2, i_dict_1
        #     assignment = assignment.T

        for i in range(n_repeats):

            iter += 1

            subset_1 = random_node_subset(graph_1, nodes_per_subset)
            subset_2 = []
            for s_1 in subset_1:
                s_1 = n_dict_1[s_1]
                s_1 = np.where(assignment[s_1])
                s_1 = s_1[0]
                if len(s_1) > 0:
                    subset_2.append(i_dict_2[s_1[0]])

            subgraph_1 = graph_1.subgraph(subset_1)
            subgraph_2 = graph_2.subgraph(subset_2)

            if nx.is_isomorphic(subgraph_1, subgraph_2):
                success += 1

        rate = float(success)/float(iter)

        if symmetric:
            rate_sym = v2.topology_compare(self,
                                           nodes_per_subset=nodes_per_subset,
                                           n_repeats=n_repeats,
                                           assignment=None,
                                           fgm_iterations=fgm_iterations,
                                           symmetric=False)

            rate = 0.5*(rate + rate_sym)

        return rate

    def compose_graphs(self):

        graphs = [c.graph for c in self.skeleton.components]
        composite_graph = nx.MultiGraph()

        for g in graphs:
            composite_graph = nx.compose(composite_graph, g)

        return composite_graph

    def construct_color_graph(self, graph1, graph2, max_d=50):

        def add_edge(graph, node1, node2, branch=None):
            if branch is None:
                branch = Branch()
                npoints = 100
                branch.diameter = 1
                branch.length = npoints
                pp = np.zeros((3, npoints))
                pp[0, :] = np.linspace(node1.point.x, node2.point.x, npoints)
                pp[1, :] = np.linspace(node1.point.y, node2.point.y, npoints)
                pp[2, :] = np.linspace(node1.point.z, node2.point.z, npoints)
                for i in range(pp.shape[1]):
                    branch.append_coords(pp[:, i])

                branch.pix_dim = [5,5,5]
                branch.analyse()
            else:
                branch.pix_dim = [5,5,5]
                branch.analyse()

            graph.add_edge(node1, node2, attr_dict={"branch": branch})

            return graph

        def test_path(graph, node1, node2):
            try:
                path = nx.shortest_path(graph, node1, node2, weight='length')
                colored = [x.color == -1 for x in path[1:-1]]
                done = [x.done == False for x in path[1:-1]]
                valid_path = all(colored)

                branch = None
                if valid_path:
                    branch = Branch()

                    for (p1, p2) in zip(path[:-1], path[1:]):
                        b = graph.get_edge_data(p1, p2)
                        if not isinstance(b, Branch):
                            b = list(b.values())[0]
                        b['branch'].used = True
                        branch.append_branch(b['branch'])

            except nx.exception.NetworkXNoPath as e:
                valid_path = False
                branch = None
                path = []

            return valid_path, branch, path

        n_dict_1 = {}
        n_dict_2 = {}
        i_dict_1 = {}
        i_dict_2 = {}

        for i, n in enumerate(graph1.nodes()):
            n.color = -1
            n.done = False
            n_dict_1[n] = i
            i_dict_1[i] = n

        for i,n in enumerate(graph2.nodes()):
            n.color = -1
            n.done = False
            n_dict_2[n] = i
            i_dict_2[i] = n

        for u, v, b in graph1.edges(data=True):
            branch = b['branch']
            b['length'] = branch.length

        for u, v, b in graph2.edges(data=True):
            branch = b['branch']
            b['length'] = branch.length

        dists = np.zeros((nx.number_of_nodes(graph1), nx.number_of_nodes(graph2)))

        for n1 in graph1.nodes():
            for n2 in graph2.nodes():
                d = n1.dist_to(n2, pix_dim=[5,5,5])
                if d > max_d:
                    d = np.nan
                dists[n_dict_1[n1], n_dict_2[n2]] = d

        for u, v, b in graph1.edges(data=True):
            b['branch'].used = False

        for u, v, b in graph2.edges(data=True):
            b['branch'].used = False

        find_pairs = True
        color = 1
        color_dict_1 = {}
        color_dict_2 = {}

        while find_pairs:
            try:
                min_dist = np.nanargmin(dists)
                min_dist = np.unravel_index(min_dist, dists.shape)

                n1 = i_dict_1[min_dist[0]]
                n2 = i_dict_2[min_dist[1]]

                n1.color = color
                n2.color = color
                n1.match = n2
                n2.match = n1

                color_dict_1[color] = n1
                color_dict_2[color] = n2

                dists[min_dist[0], :] = np.nan
                dists[:, min_dist[1]] = np.nan

                color += 1
            except ValueError as e:
                find_pairs = False

        core_graph_1 = nx.MultiGraph()
        core_graph_2 = nx.MultiGraph()

        for n in graph1.nodes():
            if n.color > 0:
                core_graph_1.add_node(n)

        for n1 in core_graph_1.nodes():
            for n2 in core_graph_1.nodes():
                if n1 != n2:
                    local_result, local_branch, local_path = test_path(graph1, n1, n2)
                    remote_result, remote_branch, remote_path = test_path(graph2, n1.match, n2.match)
                    if remote_result and local_branch is not None:
                        for p in remote_path:
                            p.done = True

                        core_graph_1 = add_edge(core_graph_1, n1, n2, local_branch)


        for n in graph2.nodes():
            if n.color > 0:
                core_graph_2.add_node(n)

        for n1 in core_graph_2.nodes():
            for n2 in core_graph_2.nodes():
                if n1 != n2:
                    local_result, local_branch, local_path = test_path(graph2, n1, n2)
                    remote_result, remote_branch, remote_path = test_path(graph1, n1.match, n2.match)
                    if remote_result and local_branch is not None:
                        for p in remote_path:
                            p.done = True

                        core_graph_2 = add_edge(core_graph_2, n1, n2, local_branch)



        # Construct FN
        fn = 0
        tp = 0
        l_fn = 0
        l_fn_total = 0
        for n in graph1.nodes():
            if n.done == False:
                fn += 1

        for u,v,b in graph1.edges(data=True):
            b = b['branch']
            l_fn_total += b.length
            if not b.used:
                fn += 1
                l_fn += b.length
            else:
                tp += 1

        # Construct FP
        fp = 0
        l_fp = 0
        l_fp_total = 0

        for n in graph2.nodes():
            if n.done==False:
                fp += 1

        for u,v,b in graph2.edges(data=True):
            b = b['branch']
            l_fp_total += b.length
            if not b.used:
                fp += 1
                l_fp += b.length
            else:
                tp += 1

        T_fn = fn / max(fn + tp, 0.01)
        T_fp = fp / max(fp + tp, 0.01)

        G_fn = l_fn / max(l_fn_total, 0.01)
        G_fp = l_fp / max(l_fp_total, 0.01)

        # Construct Assignment
        # assignment = np.zeros((nx.number_of_nodes(graph1),nx.number_of_nodes(graph2)))
        # for i in range(1, color):
        #     i1 = n_dict_1[color_dict_1[i]]
        #     i2 = n_dict_2[color_dict_2[i]]
        #     assignment[i1, i2] = 1

        return T_fn, T_fp, G_fn, G_fp, core_graph_1, core_graph_2


    def to_obj(self, location):

        with open(location,'w') as f:
            graph = self.compose_graphs()
            n_dict_i = {}

            for i, n in enumerate(graph.nodes()):
                f.write('v {} {} {} \n'.format(n.x, n.y, n.z))
                n_dict_i[n] = i + 1

            for u,v,b in graph.edges(data=True):
                b = b['branch']

                f.write('l {} {} \n'.format(n_dict_i[u], n_dict_i[v]))



    def connectivity_compare(self, v2, max_d=50):
        g1 = self.compose_graphs()
        g2 = v2.compose_graphs()

        T_fn, T_fp, G_fn, G_fp, core_graph_1, core_graph_2 = self.construct_color_graph(g1,g2, max_d=max_d)

        return T_fn, T_fp, G_fn, G_fp, core_graph_1, core_graph_2

    def get_metric_average(self, metric, function=np.mean):

        metrics = []

        for u,v,b in self.skeleton.skeleton_branch_iter():
            metrics.append(b.param_dict[metric])

        metrics = np.array(metrics)
        return function(metrics)

    def random_node_subset(graph, n_nodes):
        node_list = set()
        neighbours_set = set()
        init_node = graph.nodes()[np.random.randint(graph.number_of_nodes())]

        node_list.add(init_node)
        for n in graph.neighbors(init_node):
            neighbours_set.add(n)

        while len(node_list) < n_nodes and len(neighbours_set) > 0:
            for n in node_list:
                n2 = graph.neighbors(n)
                for n3 in n2:
                    if n3 not in node_list:
                        neighbours_set.add(n3)
            node_list.add(neighbours_set.pop())

        return node_list


    def get_subtree(self, nodes):

        sub_v = VesselTree()
        sub_v.image_dimensions = self.image_dimensions
        sub_v.skeleton = Skeleton()
        for c in self.skeleton.components:
            comp = SkeletonComponent()
            comp.graph = c.graph.subgraph(nodes)
            sub_v.skeleton.components.append(comp)
        sub_v.skeleton.analyse()

        return sub_v


    def compute_metric_histogram(self, metric, bins=20, output_name=None, units=None, range=None):

        X = []
        plt.figure()
        for u, v, b in self.skeleton.skeleton_branch_iter():
            X.append(b.param_dict[metric])

        str_dict = {}

        if units is not None:
            str_dict['units'] = '(' + units + ')'
        else:
            str_dict['units'] = ''

        str_dict['metric'] = metric

        plt.hist(X, bins=bins, range=range)
        plt.title("Histogram of {} values".format(metric))
        plt.ylabel("Count")
        plt.xlabel("{metric} {units}".format(**str_dict))

        if output_name is not None:
            plt.savefig(output_name)
            plt.close()
        else:
            plt.show()

    def metric_compare_histogram(self, metric1, metric2, bins=20, output_name=None, units1=None, units2=None, xlim=None,
                                 ylim=None):
        X1 = []
        X2 = []
        plt.figure()
        for u, v, b in self.skeleton.skeleton_branch_iter():
            X1.append(b.param_dict[metric1])
            X2.append(b.param_dict[metric2])

        str_dict = {}

        if units1 is not None:
            str_dict['units1'] = '(' + units1 + ')'
        else:
            str_dict['units1'] = ''

        if units2 is not None:
            str_dict['units2'] = '(' + units2 + ')'
        else:
            str_dict['units2'] = ''

        str_dict['metric1'] = metric1
        str_dict['metric2'] = metric2

        X1 = np.array(X1)
        X2 = np.array(X2)

        data = {}
        data[metric1] = X1
        data[metric2] = X2
        data = pd.DataFrame(data)

        sns.jointplot(x=metric1, y=metric2, data=data, kind='hex', xlim=xlim, ylim=ylim)
        plt.title("Comparison of {metric1} and {metric2} values".format(**str_dict))

        if output_name is not None:
            plt.savefig(output_name)
            plt.close()
        else:
            plt.show()


    def write_metric_csv(self,  metric, output_name=None):

        if not isinstance(metric, list):
            metric = [metric, ]

        metric_dict = {}

        for m in metric:
            metric_dict[m] = []

        for u, v, b in self.skeleton.skeleton_branch_iter():
            for m in metric:
                metric_dict[m].append(b.param_dict[m])

        zd = zip(*metric_dict.values())
        with open(output_name, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(list(metric_dict.keys()))
            writer.writerows(zd)

    def draw_graph(self, component_id=0, use_locations=False, output_name=None):

        from networkx import draw
        import matplotlib.pyplot as plt
        graph = self.skeleton.components[0].graph

        if use_locations:
            locs = []
            for n in graph.nodes_iter():
                locs.append(np.array([n.x, n.y]))

            locs = np.transpose(np.array(locs))
            loc_dict = dict(zip(graph, locs))
            draw(graph, loc=loc_dict)
        else:
            draw(graph, node_size=50)

        if output_name is None:
            plt.show()
        else:
            plt.imsave(output_name)

    def distance_to(self, vessel_tree, symmetric=False):

        sk1 = self.skeleton
        sk2 = vessel_tree.skeleton

        distance = 0
        iter = 0

        for n1 in sk1.skeleton_node_iter():
            best_dist = np.inf
            for n2 in sk2.skeleton_node_iter():
                dist = n2.point.dist_to(n1.point)
                if dist < best_dist:
                    best_dist = dist
            iter += 1
            distance += best_dist

        distance /= max(iter,1)
        if symmetric:
            iter = 0
            dist2 = 0
            sk1 = vessel_tree.skeleton
            sk2 = self.skeleton
            for n1 in sk1.skeleton_node_iter():
                best_dist = np.inf
                for n2 in sk2.skeleton_node_iter():
                    dist = n2.point.dist_to(n1.point)
                    if dist < best_dist:
                        best_dist = dist

                dist2 += best_dist
                iter += 1

            dist2 /= max(iter,1)
            distance += dist2
            distance /= 2

        return distance

    @staticmethod
    def remove_duplicate_points(a, b=None):

        if len(a) == 1:
            return a

        new_p = [a[0]]

        if b is not None:
            new_b = b[0]

        for p1, p2 in zip(a[:-1], a[1:]):
            if (p1 == p2).all() == False:
                new_p.append(p2)
                new_b.append()

        return np.array(new_p)

    def add_skeleton(self, skeleton):
        self.skeleton = skeleton
        self.skeleton_tracker.skeleton = skeleton

    def save_skeleton(self, write_location):
        f = open(write_location, "wb")
        dump(self.skeleton, f)

    def load_skeleton(self, read_location):
        f = open(read_location, "rb")
        skeleton = load(f, encoding='latin1')
        self.add_skeleton(skeleton)



class SkeletonTracker(object):
    """Class to populate a Skeleton object from a skeleton image"""

    def __init__(self, min_branch_length=0, pix_dim=None, debug=False, min_object_size=0):
        self.skeleton = Skeleton()
        self.current_idx = None
        self.branch_dict = {}
        self.current_component = None
        self.min_branch_length = min_branch_length
        self.debug = False
        self.total_skel_points = None
        self.skel_points_tracked = 0
        self.min_object_size = min_object_size
        self.props = None
        self.image_dimensions = None
        self.min_branch_clr = 0.1
        self.verbose = True

        if pix_dim is None:
            self.pix_dim = [1, 1, 1]
        else:
            self.pix_dim = pix_dim

        self.skeleton.pix_dim = self.pix_dim

    def analyse_skeleton(self, skeleton_image, distance_map=None, distance_scaling=1, verbose=True):

        self.image_dimensions = skeleton_image.shape
        self.branch_counter = 0
        self.verbose = verbose

        if np.sum(skeleton_image) == 0:
            if verbose:
                print("Skeleton image is empty...")
            return self.skeleton

        skeleton_image = np.pad(skeleton_image, 1, 'constant')
        if distance_map is not None:
            self.distance_map = np.pad(distance_map, 1, 'constant')
        else:
            self.distance_map = None

        self.distance_scaling = distance_scaling

        self.label_image, self.components, self.skeleton_image = self.split_components(skeleton_image)

        self.total_skel_points = int(np.sum(self.skeleton_image))

        if self.verbose:
            # self.pre_tracking_summary()
            print("Analysing skeleton...")

        self.start_point_dict = self.find_all_start_points(self.label_image)

        self.start_time = time.time()

        for comp_id in self.components:
            if len(self.start_point_dict[comp_id]) > 0:
                comp = self.track_component(comp_id)
                self.skeleton.add_component(comp)

        self.tracking_time = time.time() - self.start_time
        self.status_update(draw_progress_bar=self.verbose)
        self.prune_skeleton()
        self.skeleton.analyse()

        return self.skeleton

    def find_all_start_points(self, comp_image):

        start_point_dict = {}

        for c in self.components:
            start_point_dict[c] = []

        end_kernel = np.ones((3, 3, 3))
        end_kernel[1, 1, 1] = 0

        use_convolution = False

        if use_convolution:
            n_neighbours = scipy.ndimage.filters.convolve(self.skeleton_image, end_kernel)
            start_points = np.transpose(np.where(n_neighbours == 1))
            for sk in start_points:
                id = comp_image[sk[0],sk[1],sk[2]]
                start_point_dict[id].append(Point(sk))
        else:
            skel_points = np.transpose(np.where(self.skeleton_image))
            for sk in skel_points:
                box = self.get_box(comp_image, sk)
                id = comp_image[sk[0],sk[1],sk[2]]
                n_neighbours = np.sum(box * end_kernel) / id
                if n_neighbours == 1:
                    start_point_dict[id].append(Point(sk))

        return start_point_dict


    def track_component(self, component_id):
        self.branch_dict = {}
        comp_image = self.skeleton_image

        skel_component = SkeletonComponent()
        self.current_component = skel_component
        self.current_idx = 2

        if self.debug:
            print("Analysing component: {}".format(component_id))

        start_point = self.start_point_dict[component_id][0]

        if start_point is not None:
            start_node = Node(start_point)
            skel_component.add_node(start_node)

            tracker = PointTracker()
            tracker.add_point(start_point, start_node)

            for t, parent_node in tracker.generator():
                branch, new_tracking_points = self.track_vessel_from_point(t, comp_image)
                self.branch_dict[self.current_idx] = branch
                end_node = self.finalize_branch(branch, parent_node, new_tracking_points)

                for new_t in new_tracking_points:
                    tracker.add_point(new_t, end_node)

                self.current_idx += 1
                self.branch_counter += 1

                if self.branch_counter % 10 == 0:
                    self.status_update(draw_progress_bar=self.verbose)

        return skel_component

    def split_components(self, skeleton_image):

        skeleton_image = (skeleton_image > 0).astype(np.uint8)

        labels, n_comp = scipy.ndimage.measurements.label(skeleton_image > 0, structure=np.ones((3, 3, 3)))

        components = [c + 1 for c in range(n_comp)]
        return labels, components, skeleton_image

    def find_start_points(self, comp_image, component_id):
        end_kernel = np.ones((3, 3, 3))
        end_kernel[1, 1, 1] = 0

        use_convolution = False

        if use_convolution:
            n_neighbours = scipy.ndimage.filters.convolve(comp_image, end_kernel)
            start_points = np.transpose(np.where(n_neighbours == 1))

        else:
            skel_points = np.transpose(np.where(comp_image == component_id))
            start_points = []
            for sk in skel_points:
                box = self.get_box(comp_image, sk)
                n_neighbours = np.sum(box * end_kernel) / component_id
                if n_neighbours == 1:
                    start_points.append(sk)

        if len(start_points) > 0:
            return Point(start_points[0])
        else:
            if len(skel_points) > 3:
                print("WARNING: No valid start points detected, check skeletonization algorithm")
            return None

    def finalize_branch(self, branch, parent, new_tracking_points):
        """Decide whether to add tracked branch to tree or not"""
        end_node = None

        if len(new_tracking_points) > 0 or len(branch.points) > self.min_branch_length or self.current_component.num_nodes <= 2:
            end_node = self.find_node(branch.points[-1])

            if end_node is None:
                end_node = Node(branch.points[-1])
                self.current_component.add_node(end_node)

            self.current_component.add_branch(parent, end_node, branch)

        return end_node

    def track_vessel_from_point(self, point, comp_image):
        """Traverse vessel image until branch of end point is reached.
        Initializes a Branch object and appends each new point

        Returns the tracked Branch object and a list of new initialization Point objects to begin tracking from"""

        if self.debug:
            print("Tracking branch: {}".format(self.current_idx))

        dims = comp_image.shape
        current_coords = np.array([point.x, point.y, point.z])
        branch = Branch(point)
        comp_image.flat[point.to_ind(dims)] = self.current_idx
        self.skel_points_tracked += 1

        reached_end = False

        while reached_end is False:

            neighbours = self.get_box(comp_image, current_coords)
            neighbours[1, 1, 1] = -1
            idx_neighbours = np.transpose(np.where(neighbours == 1))
            n_neighbours = len(idx_neighbours)

            if n_neighbours == 1:
                """Branch continuation"""

                current_coords += (idx_neighbours[0] - 1)
                p = Point(current_coords)
                comp_image[p.x, p.y, p.z] = self.current_idx

                if self.distance_map is not None:
                    p.diameter = self.get_diameter_at(p)

                branch.append_point(p)
                self.skel_points_tracked += 1

            if n_neighbours == 0:
                """Branch end point reached. Check for reconnection with previously tracked branches"""

                if self.debug:
                    print("Reached end point, Branch: {}".format(self.current_idx))

                return branch, []

            if n_neighbours > 1:
                """Branching point reached. Return new branch start points"""

                if self.debug:
                    print("Found branching point, Branch: {}".format(self.current_idx))

                new_tracking_points = []
                for n in idx_neighbours:
                    p_n = Point(current_coords + (n - 1))

                    if self.distance_map is not None:
                        p_n.diameter = self.get_diameter_at(p_n)

                    new_tracking_points.append(p_n)
                    comp_image[p_n.x, p_n.y, p_n.z] = 0

                return branch, new_tracking_points

        return Branch

    def get_box(self, image, center):
        return image[center[0] - 1:center[0] + 2, center[1] - 1:center[1] + 2, center[2] - 1:center[2] + 2]

    def prune_skeleton(self):

        self.skeleton.analyse()

        for c in self.skeleton.components:

            for i in range(3):
                c = self.merge_nodes(c)
                c.graph = self.clean_up_nodes(c.graph)
                c.analyse()
                c = self.prune_branches(c)
                c.graph = self.clean_up_nodes(c.graph)

        keep_list = []
        for c in self.skeleton.components:
            c.graph = self.clean_up_nodes(c.graph)
            if c.num_nodes > 0 and c.num_branches > 0:
                keep_list.append(c)

        self.skeleton.components = keep_list

    def clean_up_nodes(self, g):
        nodes_to_remove = []
        for n in g.nodes_iter():
            if g.degree(n) == 0:
                nodes_to_remove.append(n)

        g.remove_nodes_from(nodes_to_remove)

        return g

    def merge_nodes(self, c):
        for n in c.graph.nodes_iter():
            if c.graph.degree(n) in [2]:
                c.remove_node_merge_branches(n)

        return c

    def prune_branches(self, c):

        for n in c.graph.nodes():
            nb = c.graph.neighbors(n)
            if len(nb) == 1:
                nb = nb[0]
                e = c.graph.get_edge_data(n, nb)

                for _e in list(e.items()):
                    min_length = max(self.min_branch_length, 2*np.max(_e[1]['branch'].diameter))

                    if _e[1]['branch'].length < min_length and c.num_nodes > 2:
                        while c.graph.has_edge(n, nb):
                            c.graph.remove_edge(n, nb)

        change = True
        # while change:
        #     change = False
        #     for u, v, b in c.branch_iter():
        #         if b.length < 5:
        #             c.collapse_branch(b)
        #             change = True
        #             break
        #
        #         if b.c2l > 200./5000:
        #             c.graph.remove_edge(u, v)
        #             change = True
        #             break

        c.remove_self_loops()
        return c


    def find_node(self, location):
        best_dist = np.inf
        best_node = None
        for n in self.current_component.graph.nodes_iter():
            dist = location.dist_to(n.point)
            if dist < 3 and dist < best_dist:
                best_node = n

        return best_node

    def get_diameter_at(self, p):
        d = 2 * float(self.distance_map[p.x, p.y, p.z]) / self.distance_scaling
        return d

    def status_update(self, draw_progress_bar=True):
        if draw_progress_bar and self.debug is False:
            self.progress_bar(self.skel_points_tracked, self.total_skel_points, show_eta=True)

    def pre_tracking_summary(self):

        min_comp = np.inf
        max_comp = 0

        for p in self.props:
            if p.area > max_comp:
                max_comp = p.area
            if p.area < min_comp:
                min_comp = p.area

        print("Summary:")
        print("Image size: {}".format(self.image_dimensions))
        print("Number of components: {}".format(len(self.props)))
        print("Largest component: {}       Smallest component: {}".format(max_comp, min_comp))

    def remove_bundle_artefacts(self):
        for c in self.skeleton.components:
            g = c.graph
            remove_nodes = []
            for n in g.nodes():
                if g.degree(n) > 7:
                    for u,v,edge in c.graph:
                        pass



            g.remove_nodes_from(remove_nodes)



    def progress_bar(self, iteration, total, prefix='', suffix='', decimals=1, barLength=100, fill='█', show_eta=False):
        """
            Call in a loop to create terminal progress bar
            @params:
                iteration   - Required  : current iteration (Int)
                total       - Required  : total iterations (Int)
                prefix      - Optional  : prefix string (Str)
                suffix      - Optional  : suffix string (Str)
                decimals    - Optional  : positive number of decimals in percent complete (Int)
                barLength   - Optional  : character length of bar (Int)
            """
        if 'linux' not in sys.platform:
            fill = '#'

        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(barLength * iteration // total)
        bar = fill * filledLength + '-' * (barLength - filledLength)

        if show_eta is False:
            eta = ''
        else:
            current_time = time.time()
            delta_t = current_time - self.start_time
            completed = float(iteration) / total
            if completed == 0:
                eta_s = 10000000
            else:
                eta_s = delta_t * (1 - completed) / completed

            eta = ' ETA: %ds' % eta_s

        sys.stdout.write('\r%s |%s| %s%s %s %s ' % (prefix, bar, percent, '%', eta, suffix)),
        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()


class PointTracker():
    def __init__(self):
        """Maintains a list of tracking starting points (Point objects) and the corresponding parent branch
        (Branch objects) and returns them via a generator"""
        self.tracking_points = []
        self.parent_branches = []

    def add_point(self, point, parent):
        """adds point (Point) and parent (Branch) to the tracking points list"""
        self.tracking_points.append(point)
        self.parent_branches.append(parent)

    def generator(self):
        for t, i in enumerate(self.tracking_points):
            yield i, self.parent_branches[t]


""" Helper Functions """

def random_node_subset(graph, n_nodes):
    node_list = set()
    neighbours_set = set()
    init_node = graph.nodes()[np.random.randint(graph.number_of_nodes())]

    node_list.add(init_node)
    for n in graph.neighbors(init_node):
        neighbours_set.add(n)

    while len(node_list) < n_nodes and len(neighbours_set) > 0:
        for n in node_list:
            n2 = graph.neighbors(n)
            for n3 in n2:
                if n3 not in node_list:
                    neighbours_set.add(n3)
        node_list.add(neighbours_set.pop())

    return node_list


def analyse_image_file(image_path, min_object_size=100, min_branch_length=20):

    image = nib.load(image_path).get_data

    v = VesselTree(image, min_object_size=min_object_size, min_branch_length=min_branch_length)
    v.analyse_vessels()

    return v

def analyse_image(image, min_object_size=100, min_branch_length=20):

    v = VesselTree(image, min_object_size=min_object_size, min_branch_length=min_branch_length)
    v.analyse_vessels()

    return v

def _get_ranges_from_vessels(vessels):
    x_max = 0
    x_min = np.inf

    y_max = 0
    y_min = np.inf

    max_length = 0
    min_length = np.inf
    max_diameter = 0
    min_diameter = np.inf
    max_soam = 0
    min_soam = np.inf
    max_soam_per_length = 0
    min_soam_per_length = np.inf
    max_clr = 0
    min_clr = np.inf

    for v in vessels:

        for u, v, b in v.skeleton.skeleton_branch_iter():
            if b.length < min_length:
                min_length = b.length
            if b.length > max_length:
                max_length = b.length

            if b.diameter < min_diameter:
                min_diameter = b.diameter
            if b.diameter > max_diameter:
                max_diameter = b.diameter

            if b.soam < min_soam:
                min_soam = b.soam
            if b.soam > max_soam:
                max_soam = b.soam

            if b.soam_per_length < min_soam_per_length:
                min_soam_per_length = b.soam_per_length
            if b.soam_per_length > max_soam_per_length:
                max_soam_per_length = b.soam_per_length

            if b.clr < min_clr:
                min_clr = b.clr
            if b.clr > max_clr:
                max_clr = b.clr

            for p in b.points:
                if p.x > x_max:
                    x_max = p.x
                if p.x < x_min:
                    x_min = p.x
                if p.y > y_max:
                    y_max = p.y
                if p.y < y_min:
                    y_min = p.y

    ranges = {}
    ranges['diameter'] = (min_diameter, max_diameter)
    ranges['length'] = (min_length, max_length)
    ranges['soam'] = (min_soam, max_soam)
    ranges['soam_per_length'] = (min_soam_per_length, max_soam_per_length)
    ranges['clr'] = (min_clr, max_clr)
    ranges['x'] = (x_min, x_max)
    ranges['y'] = (y_min, y_max)

    ranges['node_density'] = (0, 0.003)

    return ranges


def _parse_ranges(params, vessels):

    ranges = {}
    ranges['diameter'] = params['diameter_range']
    ranges['length'] = params['length_range']
    ranges['soam'] = params['soam_range']
    ranges['soam_per_length'] = params['soam_per_length_range']
    ranges['clr'] = params['clr_range']
    ranges['density'] = params['node_density_range']
    ranges['x'] = params['x_range']
    ranges['y'] = params['y_range']

    auto_ranges = _get_ranges_from_vessels(vessels)

    for r in ranges.keys():
        if ranges[r] == 'auto':
            ranges[r] = auto_ranges[r]

    return ranges


def whole_image_analysis(image_path, seg_path, output_path, params):
    print("Computing whole image measurements...")
    total_vd, vessel_mass, tumour_mass = compute_vessel_density(image_path, seg_path, params)

    with open(output_path, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["Observed Vessel Density", "Observed Vessel Mass (microns^3)", "Observed Tumour Mass (microns^3)"])
        writer.writerow([total_vd, vessel_mass, tumour_mass])


def compute_vessel_density(image_path, seg_path, params):
    from unet_core.segmentation import _split_into_tiles

    with ImageReader(image_path, image_series=1, keep_vm_open=False) as r_im, ImageReader(seg_path) as r_seg:
        image_dims = r_im.get_dims().astype(int)

        tile_size = [256, 256]
        stride = [256, 256]

        tiles = _split_into_tiles(image_dims, tile_size, stride)
        nb_total = 0
        nb_ves = 0

        for i, t in enumerate(tiles):
            im_tile = r_im.get_tile(t, tile_size)
            seg_tile = r_seg.get_tile(t, tile_size)
            n_t, n_v = vessel_density_single_tile(im_tile[:, :, :, params['channels'][1]], seg_tile, 5)
            nb_total += n_t
            nb_ves += n_v

        ratio = float(nb_ves) / float(nb_total)
        vox_size = np.prod(r_seg.get_pixdims())
        nb_total *= vox_size
        nb_ves *= vox_size

        return ratio, nb_ves, nb_total


def vessel_density_single_tile(tumour_im, seg_im, tumour_thresh):

    tumour_im = im_smooth(tumour_im, sigma=5)

    tumour_seg = tumour_im > tumour_thresh
    ves_seg = seg_im > 0

    nb_tumour = np.sum(tumour_seg)
    nb_ves = np.sum(ves_seg)

    nb_total = nb_tumour + nb_ves

    return nb_total, nb_ves


def compare_vessels(comparison_file_path):

    with open(comparison_file_path) as f:
        params = json.load(f)

    output_path = params['output_path']
    skeleton_name = params['skeleton_name']
    names = params['names']

    vessels = []

    for p in params['input_paths']:
        v = VesselTree()
        v.load_skeleton(p + skeleton_name)
        v.skeleton.analyse()
        vessels.append(v)

    ranges = _parse_ranges(params, vessels)

    for v, n in zip(vessels, names):
        outputs = os.path.join(output_path, n) + "/"
        heatmaps = outputs + 'heatmaps/'
        perfusion = outputs + 'perfusion/'
        histograms = outputs + "histograms/"
        directional_coherence = outputs + "directional_coherence/"

        os.makedirs(outputs, exist_ok=True)
        v.image_dimensions = [ranges['x'][1] - ranges['x'][0], ranges['y'][1] - ranges['y'][0]]

        if params['diameter_heatmap']:
            os.makedirs(heatmaps, exist_ok=True)
            v.compute_metric_average('diameter',
                                     output_name=heatmaps + "kernelDiameterPlot_radius100.png",
                                     radius=100,
                                     vmin=ranges['diameter'][0],
                                     vmax=ranges['diameter'][1],
                                     save_data=True)
            v.compute_metric_average('diameter',
                                     output_name=heatmaps + "kernelDiameterPlot_radius200.png",
                                     radius=200,
                                     vmin=ranges['diameter'][0],
                                     vmax=ranges['diameter'][1],
                                     save_data=True)
        if params['length_heatmap']:
            os.makedirs(heatmaps, exist_ok=True)
            v.compute_metric_average('length',
                                     output_name=heatmaps + "kernelLengthPlot_radius100.png",
                                     radius=100,
                                     vmin=ranges['length'][0],
                                     vmax=ranges['length'][1],
                                     save_data=True)
            v.compute_metric_average('length',
                                     output_name=heatmaps + "kernelLengthPlot_radius200.png",
                                     radius=200,
                                     vmin=ranges['length'][0],
                                     vmax=ranges['length'][1],
                                     save_data=True)

        if params['clr_heatmap']:
            os.makedirs(heatmaps, exist_ok=True)
            v.compute_metric_average('clr',
                                     output_name=heatmaps + "kernelClrPlot_radius100.png",
                                     radius=100,
                                     vmin=ranges['clr'][0],
                                     vmax=ranges['clr'][1],
                                     save_data=True)
            v.compute_metric_average('clr',
                                     output_name=heatmaps + "kernelClrPlot_radius200.png",
                                     radius=200,
                                     vmin=ranges['clr'][0],
                                     vmax=ranges['clr'][1],
                                     save_data=True)

        if params['soam_per_length_heatmap']:
            os.makedirs(heatmaps, exist_ok=True)
            v.compute_metric_average('soam_per_length',
                                     output_name=heatmaps + "kernelSOAMPerLengthPlot_radius100.png",
                                     radius=100,
                                     vmin=ranges['soam_per_length'][0],
                                     vmax=ranges['soam_per_length'][1],
                                     save_data=True)
            v.compute_metric_average('soam_per_length',
                                     output_name=heatmaps + "kernelSOAMPerLengthPlot_radius200.png",
                                     radius=200,
                                     vmin=ranges['soam_per_length'][0],
                                     vmax=ranges['soam_per_length'][1],
                                     save_data=True)

        if params['density_heatmap']:
            os.makedirs(heatmaps, exist_ok=True)
            v.compute_node_density(output_name=heatmaps + "nodeDensity.png",
                                   normalize=False,
                                   vmin=ranges['density'][0],
                                   vmax=ranges['density'][1],
                                   save_data=True)

        if params['diameter_histogram']:
            os.makedirs(histograms, exist_ok=True)
            v.compute_metric_histogram('diameter',
                                       output_name=histograms + "histogramDiameter.png",
                                       range=ranges['diameter'])
            v.metric_compare_histogram(metric1='diameter',
                                       metric2='length',
                                       output_name=histograms + "lengthDiameterCompare.png",
                                       xlim=ranges['diameter'],
                                       ylim=ranges['length'])
            v.metric_compare_histogram(metric1='diameter',
                                       metric2='clr',
                                       output_name=histograms + "diameterClrCompare.png",
                                       xlim=ranges['diameter'],
                                       ylim=ranges['clr'])
            v.metric_compare_histogram(metric1='diameter',
                                       metric2='soam_per_length',
                                       output_name=histograms + "diameterSOAMPerLengthCompare.png",
                                       xlim=ranges['diameter'],
                                       ylim=ranges['soam_per_length'])

        if params['length_histogram']:
            os.makedirs(histograms, exist_ok=True)
            v.compute_metric_histogram('length',
                                       output_name=histograms + "histogramLength.png",
                                       range=ranges['length'])
            v.metric_compare_histogram(metric1='length',
                                       metric2='clr',
                                       output_name=histograms + "lengthClrCompare.png",
                                       xlim=ranges['length'],
                                       ylim=ranges['clr'])
            v.metric_compare_histogram(metric1='length',
                                       metric2='soam_per_length',
                                       output_name=histograms + "lengthSOAMPerLengthCompare.png",
                                       xlim=ranges['length'],
                                       ylim=ranges['soam_per_length'])

        if params['clr_histogram']:
            os.makedirs(histograms, exist_ok=True)
            v.compute_metric_histogram('clr',
                                       output_name=histograms + "histogramClr.png",
                                       range=ranges['clr'])

        if params['soam_per_length_histogram']:
            os.makedirs(histograms, exist_ok=True)
            v.compute_metric_histogram('soam_per_length',
                                       output_name=histograms + "histogramSOAMPerLength.png",
                                       range=ranges['soam_per_length'])

        if params['directional_coherence']:
            os.makedirs(directional_coherence, exist_ok=True)
            v.compute_directional_coherence(output_name=directional_coherence + "vesselDirection_radius200.png", radius=200)
            v.compute_directional_coherence(output_name=directional_coherence + "vesselDirection_radius400.png", radius=400)

        if params['perfusion']:
            os.makedirs(perfusion, exist_ok=True)
            v.plot_vessels(metric='perfusion_max', metric_scaling=1, width_scaling=0.2,
                           write_location=perfusion + "perfusionMaxPlot.png")
            v.plot_vessels(metric='perfusion_min', metric_scaling=1, width_scaling=0.2,
                           write_location=perfusion + "perfusionMinPlot.png")
            v.plot_vessels(metric='perfusion_median', metric_scaling=1, width_scaling=0.2,
                           write_location=perfusion + "perfusionMedianPlot.png")
            v.plot_vessels(metric='perfusion_mean', metric_scaling=1, width_scaling=0.2,
                           write_location=perfusion + "perfusionMeanPlot.png")

            v.plot_vessels(metric='perfusion_median', metric_scaling=1, width_scaling=0.2,
                           threshold=2, write_location=perfusion + "perfusionMedianPlot_threshold2.png")
            v.plot_vessels(metric='perfusion_median', metric_scaling=1, width_scaling=0.2,
                           threshold=3, write_location=perfusion + "perfusionMedianPlot_threshold3.png")
            v.plot_vessels(metric='perfusion_median', metric_scaling=1, width_scaling=0.2,
                           threshold=5, write_location=perfusion + "perfusionMedianPlot_threshold5.png")
            v.plot_vessels(metric='perfusion_median', metric_scaling=1, width_scaling=0.2,
                           threshold=10, write_location=perfusion + "perfusionMedianPlot_threshold10.png")


def swc_to_volume(swc_path, image_dims):
    data = np.genfromtxt(swc_path, delimiter=' ')
    image = np.zeros(image_dims)
    nb_interp = 50
    data[:, 2] = np.clip(data[:, 2], 0, image_dims[0] - 1)
    data[:, 3] = np.clip(data[:, 3], 0, image_dims[1] - 1)
    data[:, 4] = np.clip(data[:, 4], 0, image_dims[2] - 1)

    for i in range(data.shape[0]):
        idx = int(data[i, 0])
        parent = int(data[i, -1])
        type = int(data[i, 1])

        if parent > -1:
            x_c, y_c, z_c = data[i, 2:5].astype(int)
            x_p, y_p, z_p = data[parent - 1, 2:5]
            x_interp = np.linspace(x_c, x_p, nb_interp)
            y_interp = np.linspace(y_c, y_p, nb_interp)
            z_interp = np.linspace(z_c, z_p, nb_interp)

            for x, y, z in zip(x_interp, y_interp, z_interp):
                image[int(x), int(y), int(z)] = 1

    return image

def skeleton_from_graph(graph):
    sk = Skeleton()
    c = SkeletonComponent()
    c.graph = graph
    sk.components.append(c)

    return sk
