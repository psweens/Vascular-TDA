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
