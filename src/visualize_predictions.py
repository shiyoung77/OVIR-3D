import copy
from collections import defaultdict

import numpy as np
import networkx as nx
import open3d as o3d

from utils import vis_pcd


def compute_hierarchy_graph(scene_graph: nx.DiGraph, precision_thresh=0.75):
    G = nx.DiGraph()
    G.add_nodes_from(scene_graph.nodes)
    for i, j in scene_graph.edges:
        if scene_graph.edges[i, j]['precision'] > precision_thresh:
            G.add_edge(j, i)

    hierarchy = {node: node for node in G.nodes}
    out_degrees = dict(G.out_degree)
    sorted_nodes = list(reversed(list(nx.topological_sort(G))))
    for node in sorted_nodes:
        for predecessor in G.predecessors(node):
            out_degrees[predecessor] -= 1
            if out_degrees[predecessor] == 0:
                hierarchy[predecessor] = node

    G = nx.DiGraph()
    G.add_nodes_from(hierarchy)
    for child_id, parent_id in hierarchy.items():
        if child_id != parent_id:
            G.add_edge(parent_id, child_id)
    return G


def compute_node_level_mapping(scene_graph: nx.DiGraph, precision_thresh=0.75):
    hierarchy_graph = compute_hierarchy_graph(scene_graph, precision_thresh)
    level_to_nodes: defaultdict[int, list] = defaultdict(list)
    node_to_level = dict()
    for node in hierarchy_graph.nodes:
        level = 0
        node_ptr = node
        predecessors = list(hierarchy_graph.predecessors(node_ptr))
        while predecessors:
            assert len(predecessors) == 1, "Node in the hierarchy tree should not have multiple parents."
            level += 1
            node_ptr = predecessors[0]
            predecessors = list(hierarchy_graph.predecessors(node_ptr))
        level_to_nodes[level].append(node)
        node_to_level[node] = level
    return level_to_nodes, node_to_level


def visualize_scene_graph_complex(scene_pcd, scene_graph, show_center=False):
    geometries = []
    pcd_vis = copy.deepcopy(scene_pcd)
    vis_colors = np.asarray(pcd_vis.colors)
    vis_colors[:] = (0, 0, 0)
    level_to_nodes, node_to_level = compute_node_level_mapping(scene_graph, precision_thresh=0.99)
    level_to_color = {0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1), 3: (1, 1, 0), 4: (1, 0, 1)}
    colors = np.random.random((len(scene_graph.nodes), 3))
    for level in range(len(level_to_nodes)):
        for node_id in level_to_nodes[level]:
            node = scene_graph.nodes[node_id]
            pt_indices = node['pt_indices']
            vis_colors[pt_indices] = colors[node_id]
            if show_center:
                mesh_sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.02)
                mesh_sphere.compute_vertex_normals()
                mesh_sphere.paint_uniform_color(level_to_color[level])
                mesh_sphere.translate(node['center'])
                geometries.append(mesh_sphere)
    geometries.append(pcd_vis)
    vis_pcd(geometries)


def visualize_scene_graph(scene_pcd, scene_graph, show_center=False):
    geometries = []
    pcd_vis = copy.deepcopy(scene_pcd)
    vis_colors = np.asarray(pcd_vis.colors)
    vis_colors[:] = (0, 0, 0)
    colors = np.random.random((len(scene_graph.nodes), 3))

    # naive visualization (sort by segment size and visualize large instances first)
    node_ids = list(scene_graph.nodes)
    print(f"{len(node_ids) = }")
    node_ids.sort(key=lambda x: len(scene_graph.nodes[x]['pt_indices']), reverse=True)
    for node_id in node_ids:
        node = scene_graph.nodes[node_id]
        pt_indices = node['pt_indices']
        vis_colors[pt_indices] = colors[node_id]
        if show_center:
            mesh_sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.02)
            mesh_sphere.compute_vertex_normals()
            mesh_sphere.paint_uniform_color((1, 0, 0))
            mesh_sphere.translate(node['center'])
            geometries.append(mesh_sphere)
    geometries.append(pcd_vis)
    vis_pcd(geometries)
