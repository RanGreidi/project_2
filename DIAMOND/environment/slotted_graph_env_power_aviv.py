import networkx as nx
from matplotlib import pyplot as plt
import copy
import sys

sys.path.insert(0, 'DIAMOND')
import numpy as np
from pprint import pprint
from collections import Counter

import warnings
from environment.utils import *  # get_k_paths, one_link_transmission, plot_graph, init_seed
from names_generator import generate_name


# import torch
# from torch_geometric.utils.convert import from_networkx


class SlottedGraphEnvPower:
    """
    """

    def __init__(self,
                 adjacency_matrix,
                 bandwidth_matrix,
                 node_positions,
                 flows,
                 k,
                 received_interference_map=None,
                 normalize_capacity=True,
                 render_mode=True,
                 seed=42,
                 slot_duration=60,  # [SEC]
                 Tot_num_of_timeslots=60, # [Minutes]
                 simulate_residuals=False,
                 Simulation_Time_Resolution=1e-3,
                 is_slotted=True,
                 arrival_matrix=None,
                 slotted=True,
                 **kwargs):

        # seed
        self.seed = seed
        init_seed(seed)

        # received
        self.flows = flows
        self.initial_flows = flows
        self.residual_flows = []
        self.next_slot_flows = None
        self.adjacency_matrix = adjacency_matrix
        self.bandwidth_matrix = bandwidth_matrix
        self.node_positions = node_positions
        self.k = k  # action size (k paths to choose from)
        self.received_interference_map = received_interference_map
        self.normalize_capacity = normalize_capacity
        self.seed = seed
        self.kwargs = kwargs

        # attributes
        self.graph: nx.DiGraph
        self.graph_pos = None
        self.nodes = None
        self.num_nodes = None
        self.num_edges = None
        self.num_flows = len(self.flows)
        self.initial_num_of_flows = len(self.flows) # TODO: num_flows can change between timeslots, this one doesnt

        self.max_capacity = np.max(self.bandwidth_matrix)
        self.demands = np.array([f["packets"] for f in self.flows])
        self.max_demand = np.max(self.demands)

        self.possible_actions = [[] for _ in range(len(self.flows))]
        self.allocated = []
        self.residual_allocated = []
        self.prev_reward = None

        # rate is a matrix with rows as flows and columns as time step
        self.routing_metrics = dict(rate=dict(rate_per_flow=np.full([self.num_flows,int(slot_duration/Simulation_Time_Resolution)],np.inf).astype(np.float64)),
                                    delay=dict(end_to_end_delay_per_flow_for_a_defined_packet_size=np.zeros(self.initial_num_of_flows), slot_delay=[]))

        self.path_bank = dict()

        # graph data
        self.interference_map = None
        self.current_link_interference = None
        self.links_length = None
        self.cumulative_link_interference = None
        self.current_link_interference_list_4EachTimeStep = None
        self.current_link_capacity_list_4EachTimeStep = None
        self.current_flow_delay_list_4EachTimeStep = None
        self.current_link_capacity = None
        self.current_link_queue = None
        self.bandwidth_edge_list = None
        self.last_action = np.zeros_like(self.adjacency_matrix)
        self.trx_power = None

        self.eids = dict()
        self.id_to_edge = []

        self.slot_duration = slot_duration
        self.Tot_num_of_timeslots = Tot_num_of_timeslots
        self.slot_num = 0
        self.active_links_after_time_slot = []

        self.render_mode = render_mode
        self.simulate_residuals = simulate_residuals
        self.Simulation_Time_Resolution = Simulation_Time_Resolution
        self.is_slotted = is_slotted
        # initialization once
        self.__create_graph()
        self.__calc_possible_actions()

        # Todo: My adding, arrival_matrix
        self.arrival_matrix = arrival_matrix
        self.alive_flow_indices = list(np.arange(self.initial_num_of_flows))
        self.print_index = 0  # For saving images if render mode is true
        self.tm_reading_idx = 0
        self.slotted = slotted
        self.original_flows = copy.deepcopy(self.flows)

        # for tf env
        self.tf_env = False
        self.num_sps = None
        self.firsts = []
        self.seconds = []
        self.gen_first_second()

    def gen_first_second(self):
        for i, u in enumerate(self.adjacency_matrix):
            for j, v in enumerate(u):
                if v != 0:
                    self.firsts.append(i)
                    self.seconds.append(j)

    def __create_graph(self):
        """
        Create communication graph
        Edges contains metadata of the network's state
        """
        # create graph
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
        # assign attributes
        self.graph = G
        self.nodes = list(G.nodes)
        self.num_nodes = G.number_of_nodes()
        self.num_edges = G.number_of_edges()

        # calc interference_map
        self.gen_edge_data()
        self.init_edge_data()

    def plot_raw_graph(self, save_path=None):
        """ draw global graph"""
        plt.figure()
        nx.draw_networkx(self.graph, self.graph_pos, with_labels=False, node_color="tab:blue",node_size=100)
        plt.axis('off')
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

    def show_graph(self, active_links, plot_rate, total_time_stemp_in_single_slot, show_fig=True):
        """ draw global graph"""
        label_dict = {}
        residual_dict = {}
        if plot_rate:
            # add flows to graph
            for a in active_links:
                if 'residual_name' not in a:
                    u, v = a['link']
                    if (u, v) in label_dict:
                        flow_name = f"\n flow_idx: {a['constant_flow_name']}"
                        label_dict[(u, v)] += (
                                    flow_name + f"\n remaining packet: {round(a['packets'], 2)} \n Avg. Rate: {round(self.routing_metrics['rate']['rate_per_flow'][a['flow_idx']][total_time_stemp_in_single_slot - 1], 2)}")
                    else:
                        flow_name = f"\n flow_idx: {a['constant_flow_name']}"
                        label_dict.update({(u,
                                            v): flow_name + f"\n remaining packet: {round(a['packets'], 2)} \n Avg. Rate: {round(self.routing_metrics['rate']['rate_per_flow'][a['flow_idx']][total_time_stemp_in_single_slot - 1], 2)}"})
                else:
                    u, v = a['link']
                    if (u, v) in residual_dict:
                        flow_name = f"\n res: {a['residual_name']}"
                        residual_dict[(u, v)] += flow_name + f"\n remaining packet: {round(a['packets'], 2)}"
                    else:
                        flow_name = f"\n res: {a['residual_name']}"
                        residual_dict.update({(u, v): flow_name + f"\n remaining packet: {round(a['packets'], 2)}"})
            # add rates table to graph
            flow_rates = self.routing_metrics['rate']['rate_per_flow'][:, total_time_stemp_in_single_slot - 1].tolist()
            table_data = [[f"Flow {i}", f"{rate:.3f}"] for i, rate in enumerate(flow_rates)]
        else:
            # add flows to graph
            for a in active_links:
                if 'residual_name' not in a:
                    u, v = a['link']
                    if (u, v) in label_dict:
                        flow_name = f"\n flow_idx: {a['constant_flow_name']}"
                        label_dict[(u, v)] += flow_name + f"\n remaining packet: {round(a['packets'], 2)}"
                    else:
                        flow_name = f"\n flow_idx: {a['constant_flow_name']}"
                        label_dict.update({(u, v): flow_name + f"\n remaining packet: {round(a['packets'], 2)}"})
                else:
                    u, v = a['link']
                    if (u, v) in residual_dict:
                        flow_name = f"\n res: {a['residual_name']}"
                        residual_dict[(u, v)] += flow_name + f"\n remaining packet: {round(a['packets'], 2)}"
                    else:
                        flow_name = f"\n res: {a['residual_name']}"
                        residual_dict.update({(u, v): flow_name + f"\n remaining packet: {round(a['packets'], 2)}"})
            # add rates table to graph
            flow_rates = self.routing_metrics['rate']['rate_per_flow'][:, total_time_stemp_in_single_slot - 1].tolist()
            table_data = [[f"Flow {i}", '--'] for i, rate in enumerate(flow_rates)]

            # add capacity matrix to graph
        current_link_capacity_mat = self.edge_list_to_adj_mat(self.current_link_capacity)
        # for u in range(self.num_nodes):
        #     for v in range(self.num_nodes):
        #         if u != v:
        #             if (u,v) in label_dict:
        #                 label_dict[(u,v)] += (f"\n Capacity [bps]: {round(current_link_capacity_mat[u,v],2)/self.Simulation_Time_Resolution}")

        # #add bandwidth matrix to graph
        # bandwidth = self.edge_list_to_adj_mat(self.bandwidth_edge_list)
        # for u in range(self.num_nodes):
        #     for v in range(self.num_nodes):
        #         if u != v:
        #             if (u,v) in label_dict:
        #                 label_dict[(u,v)] += (f"\n Total channel Bandwidth: {bandwidth[u,v]}")
        if not (label_dict == {} and residual_dict == {}):
            plot_graph(self.graph, self.graph_pos, label_dict, residual_dict, total_time_stemp_in_single_slot, table_data, self.Simulation_Time_Resolution, self.is_slotted, self.slot_num)

    def gen_edge_data(self):
        self.eids = dict()
        self.id_to_edge = []
        self.bandwidth_edge_list = []
        self.link_pos = []
        self.links_length = []
        self.graph_pos = dict()
        # self.link_metadata = dict()
        spring_pos = nx.spring_layout(self.graph)
        id = 0
        for u in range(self.num_nodes):
            for v in range(u + 1, self.num_nodes):
                if self.adjacency_matrix[u, v]:
                    # edge id
                    self.eids[(u, v)] = id
                    self.eids[(v, u)] = id
                    id += 1

                    # node position
                    if self.node_positions is not None:
                        self.graph_pos[u] = np.array(self.node_positions[u])
                        self.graph_pos[v] = np.array(self.node_positions[v])
                    else:
                        self.graph_pos[u] = np.array(spring_pos[u])
                        self.graph_pos[v] = np.array(spring_pos[v])

                    # edge position
                    self.link_pos.append(np.mean([self.graph_pos[u], self.graph_pos[v]], axis=0))
                    self.id_to_edge.append((u, v))
                    self.links_length.append(np.linalg.norm(self.graph_pos[u] - self.graph_pos[v]))

                    # capacity matrix
                    self.bandwidth_edge_list.append(self.bandwidth_matrix[u, v])

        self.link_pos = np.array(self.link_pos)
        self.links_length = np.array(self.links_length)
        self.bandwidth_edge_list = np.array(self.bandwidth_edge_list)
        self.current_link_capacity = self.bandwidth_edge_list.copy()

    def init_edge_data(self):
        L = self.num_edges // 2
        self.interference_map = np.zeros((L, L))
        self.current_link_interference = np.zeros(L)
        self.cumulative_link_interference = np.zeros(L)
        self.current_link_capacity = self.bandwidth_edge_list.copy()
        self.trx_power = self._init_transmission_power()

        # ---- raz new implementaion, interference map not caculated according to link pos---
        # if self.received_interference_map is not None:
        #     raise NotImplementedError("TODO: implement recieved_interference_map")
        # else:
        #     for l1 in range(L):
        #         for l2 in range(l1 + 1, L):
        #             trans_1, rec_1 = self.id_to_edge[l1]
        #             trans_2, rec_2 = self.id_to_edge[l2]
        #             r12 = np.linalg.norm(self.graph_pos[trans_1] - self.graph_pos[rec_2]) * 1e1  # distance [km]
        #             r21 = np.linalg.norm(self.graph_pos[trans_2] - self.graph_pos[rec_1]) * 1e1  # distance [km]
        #             # I_{i,j} = interference power at the receiver of node j
        #             # TODO: currently only spherical interference is implemented.
        #             #     need to extent do directional antennas
        #             if r12 > sys.float_info.epsilon:
        #                 self.interference_map[l1, l2] = self.trx_power[l1] / (r12 ** 2)
        #             if r21 > sys.float_info.epsilon:
        #                 self.interference_map[l2, l1] = self.trx_power[l2] / (r21 ** 2)

        if self.received_interference_map is not None:
            for l1 in self.received_interference_map:
                for l2 in self.received_interference_map[l1]:
                    self.interference_map[self.eids[l1], self.eids[l2]] = self.received_interference_map[l1][l2]
                    self.interference_map[self.eids[l2], self.eids[l1]] = self.interference_map[
                        self.eids[l1], self.eids[l2]]

        else:
            for l1 in range(L):
                for l2 in range(l1 + 1, L):
                    r = np.linalg.norm(self.link_pos[l1] - self.link_pos[l2]) * 1e1  # distance [km]
                    if r > sys.float_info.epsilon:
                        self.interference_map[l1, l2] = self.trx_power[l1] / (r ** 2)
                        self.interference_map[l2, l1] = self.trx_power[l2] / (r ** 2)
                    else:  # in case r = 0, we do want to have very large interference, not zero!
                        self.interference_map[l1, l2] = 0  # self.trx_power[l1] / ((r + 1e-10) ** 2)
                        self.interference_map[l2, l1] = 0  # self.trx_power[l2] / ((r + 1e-10) ** 2)

    def _init_transmission_power(self):
        """
        set the transmission power
        """
        L = self.num_edges // 2
        power_mode = self.kwargs.get('trx_power_mode', 'equal')
        assert power_mode in ('equal', 'rayleigh', 'rayleigh_gain', 'steps', 'gain', 'manual_gain'), f'Invalid power mode. got {power_mode}'
        channel_coeff = np.ones(L)
        channel_gain = np.ones(L)
        if 'rayleigh' in power_mode:
            channel_coeff = np.random.rayleigh(scale=self.kwargs.get('rayleigh_scale', 1), size=L)
        if 'gain' in power_mode:
            channel_gain = self.kwargs.get('channel_gain', np.random.uniform(low=1, high=10, size=L)) * np.ones(L)
        if 'manual_gain' in power_mode:
            channel_gain = self.kwargs.get('channel_manual_gain',
                                           np.random.uniform(low=0.01, high=1000, size=L)) * np.ones(L)
        p_max = self.kwargs.get('max_trx_power', 1) * np.ones(L)
        trx_power = channel_gain * np.minimum(p_max, 1 / channel_coeff)  # P_l
        if power_mode == 'steps':
            rng = np.max(self.links_length) - np.min(self.links_length)

            trx_power = np.ones(L)
            trx_power[np.where(self.links_length < rng * 1 / 3)] = 1 / 3
            trx_power[np.where((self.links_length >= rng * 1 / 3) & (self.links_length < rng * 2 / 3))] = 2 / 3
        return trx_power

    def __calc_possible_actions(self):
        """ store possible routs into self.possible_actions for each flow"""
        for i, flow in enumerate(self.flows):
            # check if not already calculated:
            if (flow["source"], flow["destination"]) in self.path_bank:
                self.possible_actions[i] = self.path_bank[(flow["source"], flow["destination"])]
            elif (flow["destination"], flow["source"]) in self.path_bank:
                paths = self.path_bank[(flow["destination"], flow["source"])]
                reversed_paths = [list(reversed(p)) for p in paths]
                self.possible_actions[i] = reversed_paths
            else:
                # 1. create new undirected graph with unit weights
                new_graph = self.graph.to_undirected()
                for u, v, attr in new_graph.edges(data=True):
                    attr['weight'] = 1

                # 2. calc k possible routs
                paths = get_k_paths(G=new_graph,
                                    s=flow["source"],
                                    d=flow["destination"],
                                    k=self.k)

                # 3.
                while len(paths) < self.k:
                    paths.append(paths[-1])

                # 4. store valid paths to
                self.possible_actions[i] = paths

                # 5. add to path bank
                self.path_bank[(flow["source"], flow["destination"])] = paths

    def __init_links(self):
        """init links at env.reset"""
        self.current_link_interference = np.zeros_like(self.current_link_interference)
        self.cumulative_link_interference = np.zeros_like(self.cumulative_link_interference)
        self.current_link_capacity = self.bandwidth_edge_list.copy()
        self.last_action = np.zeros_like(self.last_action)
        self.trx_power = self._init_transmission_power()
        self.current_link_interference_list_4EachTimeStep = []
        self.current_link_capacity_list_4EachTimeStep = []
        self.current_flow_delay_list_4EachTimeStep = []

    def __update_interference(self, s, d):
        """ update interference due to transmission s->d
            Capacity of each link over the same channel is effected:  capacity = bandwidth*log2(1+SNR)

            update interference due to transmission s->d, effects all edges except (s->*) and (d->s)
            Interference is calculated by 1/(r**2), where r is the distance between two *links*
            Capacity of each link is effected:  capacity = bandwidth*log2(1+SNR) assuming unit transmission power

            {self.current_link_interference is a vector with the total interference from all link for each link.
            example: link 0 tot interfernece is at self.current_link_interference[0]
            same goes for self.current_link_capacity}
        """

        trx_power = self.trx_power[self.eids[s, d]]  # P_l
        self.current_link_interference += self.interference_map[self.eids[s, d]]  # I_l
        sinr = trx_power / (self.current_link_interference + 1)  # SINR_l
        # self.current_link_capacity = np.maximum(1, self.bandwidth_edge_list * np.log2(1 + sinr)) # np.minimum(self.bandwidth_edge_list, np.maximum(1, np.floor(self.bandwidth_edge_list * np.log2(1 + sinr))))
        self.current_link_capacity = self.bandwidth_edge_list * np.log2(1 + sinr)  # np.minimum(self.bandwidth_edge_list, np.maximum(1, np.floor(self.bandwidth_edge_list * np.log2(1 + sinr))))

    def _transmit_singe_timestep(self, active_links, total_time_slots):
        """ send packet over all links in active_links for a single time-step """

        ## 0. produce action dict
        next_flows_list = []
        action_dict = {}
        for l1 in range(len(active_links)):
            if 'residual_name' in active_links[l1]:
                shared_resource = dict(link=active_links[l1]['link'],
                                       packets=[active_links[l1]['packets']],
                                       flows_idxs=[(active_links[l1]['flow_idx'], active_links[l1]['residual_name'])],
                                       constant_flow_name=[active_links[l1]['constant_flow_name']])
            else:
                shared_resource = dict(link=active_links[l1]['link'],
                                       packets=[active_links[l1]['packets']],
                                       flows_idxs=[(active_links[l1]['flow_idx'], None)],
                                       constant_flow_name=[active_links[l1]['constant_flow_name']])

            for l2 in range(l1 + 1, len(active_links)):
                # find transmission over the same link
                if active_links[l2]['link'] == active_links[l1]['link']:
                    shared_resource['packets'].append(active_links[l2]['packets'])
                    if 'residual_name' in active_links[l2]:
                        shared_resource['flows_idxs'].append((active_links[l2]['flow_idx'], active_links[l2]['residual_name']))
                        shared_resource['constant_flow_name'].append((active_links[l2]['constant_flow_name'], active_links[l2]['residual_name']))
                    else:
                        shared_resource['flows_idxs'].append((active_links[l2]['flow_idx'], None))
                        shared_resource['constant_flow_name'].append(active_links[l2]['constant_flow_name'])
            if str(shared_resource['link']) not in action_dict.keys():
                action_dict[str(shared_resource['link'])] = shared_resource


        # 1. update interference on the link
        self.current_link_interference = np.zeros_like(self.current_link_interference)  # reset current_link_interference for __update_interference
        self.current_link_capacity = self.bandwidth_edge_list.copy()  # reset current_link_capacity for __update_interference

        for a in action_dict.values():
            u = a["link"][0]
            v = a["link"][1]

            # update links interference due to transmission u->v
            self.__update_interference(u, v)

        # 2. calculate rewards (influence on others, want to *minimize*)
        next_active_links = []
        metadata = dict(capacity_reduction=[], interference=[], delay=[])
        for a in action_dict.values():
            # metadata
            capacity = self.current_link_capacity[self.eids[a['link']]] / len(a['flows_idxs'])
            bandwidth = self.bandwidth_edge_list[self.eids[a['link']]]
            capacity_reduction = (bandwidth - capacity) / bandwidth
            metadata['capacity_reduction'].append(capacity_reduction)
            metadata['interference'].append(self.current_link_interference[self.eids[a['link']]])

            # share link's resource
            c = calc_indevidual_minimum_capacity(self, a, action_dict, self.current_link_capacity)  # c is a list of minmum capacity for each flow (in their oreder)
            remaining_packets = one_link_transmission(c, a['packets'], self.Simulation_Time_Resolution)      # transmit for a time_resolution duration. remaining packets is a list of remining packets for each flow (in their oreder)
            # remaining_packets = one_link_transmission(capacity, a['packets'])  # packets remained at transmit for the next time-step over (u, v)
            advanced_packets = [p - r for p, r in
                                zip(a['packets'], remaining_packets)]  # packets to transmit over (v, w)

            # flows advancing to the next hop
            for idx, pkt in enumerate(advanced_packets):  # this loop is for the case two flows share the same link, than advanced_packets is a list
                flow_idx = a['flows_idxs'][idx][0]
                residual_name = a['flows_idxs'][idx][1]  # name or None

                constant_flow_name = a['constant_flow_name'][idx]
                if not residual_name:
                    flow = self.flows[flow_idx]

                    u, v = a['link']  # current hop
                    v_pos = flow['path'].index(v)
                    if v_pos < len(flow['path']) - 1:
                        next_hop = (flow['path'][v_pos], flow['path'][v_pos + 1])
                        # exist_flow = list(filter(lambda x: x[1]['flow_idx'] == flow_idx and x[1]['link'] == next_hop,
                        #                          enumerate(next_active_links)))  # filters next_active_links according to conditions in lambda
                        exist_flow = list(filter(lambda x: x[1]['constant_flow_name'] == constant_flow_name and x[1]['link'] == next_hop,
                                          enumerate(next_active_links)))
                        if exist_flow:
                            next_active_links[exist_flow[0][0]]['packets'] += pkt
                        else:
                            next_active_links.append(dict(flow_idx=flow_idx,
                                                          link=next_hop,
                                                          packets=pkt,
                                                          constant_flow_name=constant_flow_name))
                    else:
                        # only when pkt reached its destination
                        next(d for d in self.flows if d["constant_flow_name"] == constant_flow_name)["delivered_in_a_slot"] += pkt
                else:
                    # search for the spesific resiudal flow at self.residual_flows
                    flow = next((dict for dict in self.residual_flows if dict.get('residual_name') == residual_name),
                                None)

                    u, v = a['link']  # current hop
                    v_pos = flow['path'].index(v)
                    if v_pos < len(flow['path']) - 1:
                        next_hop = (flow['path'][v_pos], flow['path'][v_pos + 1])
                        exist_flow = list(filter(lambda x: x[1]['flow_idx'] == flow_idx and x[1]['link'] == next_hop and ('residual_name' in x[1]) and x[1]['residual_name'] == residual_name,
                                          enumerate(next_active_links)))  # filters next_active_links according to conditions in lambda
                        if exist_flow:
                            next_active_links[exist_flow[0][0]]['packets'] += pkt
                        else:
                            next_active_links.append(dict(flow_idx=flow_idx,
                                                          link=next_hop,
                                                          packets=pkt,
                                                          residual_name=residual_name,
                                                          constant_flow_name=constant_flow_name))



            # flows staying in the current hop for the next time stamp
            for idx, pkt in enumerate(remaining_packets):
                flow_idx = a['flows_idxs'][idx][0]
                residual_name = a['flows_idxs'][idx][1]  # name or None

                constant_flow_name = a['constant_flow_name'][idx]
                if not residual_name:
                    flow_path = self.flows[flow_idx]['path']
                    if pkt > 0:
                        # exist_flow = list(filter(lambda x: x[1]['flow_idx'] == flow_idx and x[1]['link'] == a['link'],
                        #                   enumerate(next_active_links)))
                        exist_flow = list(
                            filter(lambda x: x[1]['constant_flow_name'] == flow_idx and x[1]['link'] == a['link'],
                                   enumerate(next_active_links)))
                        if exist_flow:
                            next_active_links[exist_flow[0][0]]['packets'] += pkt
                        else:
                            next_active_links.append(dict(flow_idx=flow_idx,
                                                          link=a['link'],
                                                          packets=pkt,
                                                          constant_flow_name=constant_flow_name))
                    elif a['link'][0] == flow_path[0]:
                        # only when there are no pkt at source (flow finishes during slot)
                        next(d for d in self.flows if d["constant_flow_name"] == constant_flow_name)["delivered_in_a_slot"] = np.inf
                else:
                    flow = next((dict for dict in self.residual_flows if dict.get('residual_name') == residual_name), None)
                    if pkt > 0:
                        # exist_flow = list(filter(lambda x: x[1]['flow_idx'] == flow_idx and x[1]['link'] == a['link'] and ('residual_name' in x[1]) and x[1]['residual_name'] == residual_name,
                        #     enumerate(next_active_links)))
                        exist_flow = list(filter(lambda x: x[1]['constant_flow_name'] == constant_flow_name and x[1]['link'] == a['link'] and ('residual_name' in x[1]) and x[1]['residual_name'] == residual_name,
                                          enumerate(next_active_links)))
                        if exist_flow:
                            next_active_links[exist_flow[0][0]]['packets'] += pkt
                        else:
                            next_active_links.append(dict(flow_idx=flow_idx,
                                                          link=a['link'],
                                                          packets=pkt,
                                                          residual_name=residual_name,
                                                          constant_flow_name=constant_flow_name))

        if len(self.allocated) == len(self.flows):

            if total_time_slots == int(self.slot_duration/self.Simulation_Time_Resolution)-1: # last time step in time slot
                self.residual_flows = next_flows_list

            # update rate metric - find bottleneck rate for each flow
            for a in active_links:
                if 'residual_name' not in a:
                    flow_idx = a['flow_idx']
                    # if not flow_idx[1]: #TODO is this currect?
                    link_capacity = self.current_link_capacity[self.eids[a['link']]]
                    how_many_share_this_link_with_flow_idx = Counter(link['link'] for link in active_links if 'link' in link)[a['link']]
                    # rate_in_bps = (link_capacity / self.Simulation_Time_Resolution) / how_many_share_this_link_with_flow_idx
                    rate_in_bps = link_capacity / how_many_share_this_link_with_flow_idx
                    if self.routing_metrics['rate']['rate_per_flow'][flow_idx][self.total_time_stemp_in_single_slot] > rate_in_bps:

                        # Todo : my adding, rate is minimum between demand and bottleneck
                        # rate_in_bps = np.minimum(rate_in_bps, a["packets"])

                        self.routing_metrics['rate']['rate_per_flow'][flow_idx][self.total_time_stemp_in_single_slot] = rate_in_bps  # this is the rate in a single time step (which how much packet were deliverd in a single time resolution)

        counter_for_same_link = Counter(self.eids[link['link']] for link in active_links if 'link' in link)
        counter_for_same_link = Counter({key: value + 1 for key, value in counter_for_same_link.items()})  # becaouse: if one line start to use the link, we want to next capcaity list to be already divided by 2
        current_link_sharing_list = [counter_for_same_link.get(i, 1) for i in range(len(self.current_link_capacity))]  # list of how many flows share each link
        current_link_capacity_after_sharing_devision = self.current_link_capacity / current_link_sharing_list  # interference after sharing the link
        # update list for all links interefences in order to avarge later
        self.current_link_interference_list_4EachTimeStep.append(self.current_link_interference)
        self.current_link_capacity_list_4EachTimeStep.append(current_link_capacity_after_sharing_devision)

        packet_size = 1e5
        # flow_ready_target = len(self.allocated)
        flow_ready_target = sum(1 for flow in self.flows if 'path' in flow and flow.get('delivered_in_a_slot') != np.inf)
        flow_ready_counter = 0
        # update delay metric
        for ii, flow in enumerate(self.flows):
            # flow_idx = flow['constant_flow_name']
            flow_constant_name = flow['constant_flow_name']
            if 'path' in flow and active_links: # if flow is allocated and active_links is not empty (simulation is still running)
                # stop counting when a full packet reached the flow dst - when last link in flow path is active and has delivered more then packet_size
                if flow['delivered_in_a_slot'] < packet_size:
                    self.routing_metrics['delay']['end_to_end_delay_per_flow_for_a_defined_packet_size'][flow_constant_name] += self.Simulation_Time_Resolution
                elif flow['delivered_in_a_slot'] > packet_size and flow['delivered_in_a_slot'] < np.inf and self.routing_metrics['delay']['end_to_end_delay_per_flow_for_a_defined_packet_size'][flow_constant_name] == 0:
                    # in case a larger amount of data was delivered in the first time step
                    self.routing_metrics['delay']['end_to_end_delay_per_flow_for_a_defined_packet_size'][flow_constant_name] = self.Simulation_Time_Resolution
                elif flow['delivered_in_a_slot'] < np.inf:
                    flow_ready_counter += 1

        # if (delay_ready[0] if delay_ready else 0) & (delay_ready[0] if len(delay_ready) > 1 else 0):
        if (flow_ready_counter == flow_ready_target) and (flow_ready_target > 0) and (active_links):
            # if all element in delay_ready list are True
            any(d.update({"delivered_in_a_slot": 0}) for d in self.flows if d['delivered_in_a_slot'] != np.inf) # zero all delivered in a slot to calulate delay all over again
            self.current_flow_delay_list_4EachTimeStep.append(self.routing_metrics['delay']['end_to_end_delay_per_flow_for_a_defined_packet_size'])
            self.routing_metrics['delay']['end_to_end_delay_per_flow_for_a_defined_packet_size'] = np.zeros(self.initial_num_of_flows)

        # 3. plot & reset & save interferences for next global step
        # if self.render_mode:
        #     self.show_graph(active_links, total_time_slots, plot_rate,self.total_time_stemp_in_single_slot)
        if next_active_links:
            # self.cumulative_link_interference += self.current_link_interference
            if np.mean(self.current_link_interference) > np.mean(self.cumulative_link_interference):
                self.cumulative_link_interference = self.current_link_interference
            self.current_link_interference = np.zeros_like(self.current_link_interference)
            self.current_link_capacity = self.bandwidth_edge_list.copy()

        return next_active_links, metadata

    def __simulate_single_slot(self, action, eval_path=False):
        """ simulate transmission of all flows from src->dst and get reward
        @input: action - [[rout],[rout],[rout]] example:[[0,3],[0,1,3]]
        @output:
        """

        ###### ---active links first generation--- ######
        # This part converts the entring flows into the first active_link list in the time slot #
        if self.flows:
            self.allocated.append(action)
            # Todo: This is not correct!! possible_actions is always a list of length num_original_flows so if some flows finished flow and path will not be attached correctly
            if not eval_path:
                allocated_paths = [self.possible_actions[a[0]][a[1]] for a in
                                   sorted(self.allocated, key=lambda x: x[0])]
            else:
                allocated_paths = [a[1] for a in sorted(self.allocated, key=lambda x: x[0])]

            # update flows in env.flows that were allocated so far

            # Todo: My change. self.allocated is not necessarily sorted with flow indices so must go through it and not with enumerate that assumes flows added by flow id.
            for allocated_action, allocated_path in zip(self.allocated, allocated_paths):
                self.flows[allocated_action[0]].update({'path': allocated_path})

            # for i, allocated_path in enumerate(allocated_paths):
            #     self.flows[i].update({'path': allocated_path})

            #   active_flows is a list of tuples with (flow_indx,rout) of all active flows.
            #   it does not change throughut a step. each episode it appends one more flow
            active_flows = [(a[0], self.flows[a[0]]) for a in self.allocated]

            # active_links is a list of dicts with srs,dst,rout,packet. it changes throughut an episode
            active_links = [dict(flow_idx=f_idx,
                                 link=(f['path'][0], f['path'][1]),
                                 packets=f['packets'], constant_flow_name=f['constant_flow_name']) for i, (f_idx, f) in enumerate(active_flows)]
        else:
            active_links = []

        if self.residual_flows:
            active_residual_flows = [a for a in self.residual_flows]
            residual_active_links = [dict(flow_idx=f['flow_idx'],
                                          link=(f['path'][0], f['path'][1]),
                                          packets=f['packets'],
                                          residual_name=f['residual_name'],
                                          constant_flow_name=f['constant_flow_name']) for f in active_residual_flows]
        else:
            residual_active_links = []

        active_links += residual_active_links
        ##########################  end of active links first generation  ##################################

        self.total_time_stemp_in_single_slot = 0
        any(d.update({"delivered_in_a_slot": 0}) for d in self.flows)  # zero all delivered in a slot

        metadata = []
        # if self.render_mode:
        #     self.show_graph(active_links, self.total_time_stemp_in_single_slot, 0, self.total_time_stemp_in_single_slot)
        #     self.print_index += 1  # for image saving
        # # if self.render_mode and self.total_time_stemp_in_single_slot == 0 and self.slot_num == 0: # initial plot
        # if self.render_mode: # initial plot
        #     self.show_graph(active_links, 0, self.total_time_stemp_in_single_slot)

        while True:
            # transmit single hop for all flows
            if self.total_time_stemp_in_single_slot < self.slot_duration / self.Simulation_Time_Resolution:


                # Todo: My adding, plot changing only when all flows in action
                if self.render_mode and len(self.allocated) == len(self.flows):
                    # plot rate only when all flows are allocated
                    plot_rate = 1 if len(self.allocated) == len(self.flows) else 0
                    self.show_graph(active_links, plot_rate, self.total_time_stemp_in_single_slot)
                    self.print_index += 1  # For image saving

                active_links, hop_metadata = self._transmit_singe_timestep(active_links, self.total_time_stemp_in_single_slot)

                metadata.append(hop_metadata)
                self.total_time_stemp_in_single_slot += 1

                # Todo: My adding, insert packets here to handle slotted\unslotted problem
                # if self.total_time_stemp_in_single_slot % self.slot_duration == 0 and len(self.allocated) == len(self.flows):  # insert demand only when all flows in play
                #     if self.arrival_matrix is not None:
                #         self.insert_traffic_demand(row_idx=self.tm_reading_idx)
                #         self.tm_reading_idx += 1
                #     else:
                #         pass

            else:
                if len(self.allocated) == len(self.flows):  # if finish all flow at a time slot
                    self.update_flows(active_links, action)
                    self.slot_num += 1
                break

            # active_links, hop_metadata = self._transmit_singe_timestep(active_links, total_time_slots)
            # metadata.append(hop_metadata)
            # total_time_slots += 1

        # introduce selected action into the graph
        p = self.possible_actions[action[0]][action[1]] if not eval_path else action[1]
        for i in range(len(p) - 1):
            self.last_action[p[i], p[i + 1]] = 1

        # calc reward
        reward = self.calc_reward(metadata, action)

        # routing metrics are re-calculated with each new flow allocation
        # if len(self.allocated) < self.num_flows:
        #     self.routing_metrics['delay']['end_to_end_delay_per_flow_for_a_defined_packet_size'] = np.zeros(self.num_flows)

        return reward

    def insert_traffic_demand(self, row_idx):
        incoming_demands = self.arrival_matrix[row_idx, :]
        for flow_id, flow in enumerate(self.flows):
            flow['packets'] += incoming_demands[flow_id]

    def update_flows(self, active_links, action):
        ''' --- update seld.flows and self.residual flows ---
            update residual flows:
        pckates that have left thier original flows, stays in thier determenied rout
        in the simulation we treat that as new flows to be in active links. This part needs to update self.residual_flows
        when new packet leave thier original flows
            update flows:
        needs to update original flows packets quantitiy
        '''
        active_flows = [(a[0], self.flows[a[0]]) for a in self.allocated]
        sorted_active_links = sorted(active_links, key=lambda x: x.get('flow_idx', 0))
        list_of_new_residuals = []
        list_of_2flows = []
        ii = 0
        res_name_index = 0  # TODO: My adding, if a lot of flows, generate_name() can repeat itself so need to make distinction
        for a in self.flows:
            flow_idx = a['flow_idx']
            constant_flow_name = a['constant_flow_name']
            # next we looks only at links that is not residual and belong to flow a (there can be only 2)
            list_of_links_4flow_a = [d for d in sorted_active_links if
                                     d.get('flow_idx') == flow_idx and not d.get('residual_name')]
            _2flows = next((d for d in list_of_links_4flow_a if d.get('link')[0] == a['source']), {})
            if self.simulate_residuals:
                _2res = [d for d in list_of_links_4flow_a if d.get('link')[0] != a['source']]
            else:
                _2res = []

            list_of_links_4flow_a.remove(_2flows) if _2flows else None
            if _2flows:
                _2flows = dict(source=a['source'],
                               destination=a['destination'],
                               packets=_2flows['packets'],   # + self.arrival_matrix[self.slot_num, a['constant_flow_name']] if self.arrival_matrix is not None else _2flows['packets'], # Todo: my adding incoming packets
                               time_constrain=10,
                               flow_idx=ii,
                               constant_flow_name=a['constant_flow_name'],
                               path=a['path'],
                               )
                ii += 1
                list_of_2flows.append(_2flows)

            for res in _2res:
                if res and res['packets'] > 0:
                    res = dict(source=res['link'][0],
                               destination=a['path'][-1],
                               packets=res['packets'],
                               time_constrain=10,
                               flow_idx=flow_idx,
                               constant_flow_name=a['constant_flow_name'],
                               path=a['path'][a['path'].index(res['link'][0]):],
                               residual_name=f'{generate_name()}_{res_name_index}')
                    # self.residual_flows.append(_2res)
                    list_of_new_residuals.append(res)
                    res_name_index += 1

        # delete finished flows from self.allocated:
        in_flows = [fl['flow_idx'] for fl in list_of_2flows]
        new_allocated = []
        for flw in self.allocated:
            if flw[0] in in_flows:
                new_allocated.append(flw)
        self.allocated = new_allocated

        # update old residuals or make new residuals:
        # if a residual flow finishes with the first node, than change him to be routed as [srs+1 -> dst]
        # if a residual flow didnot finish, update its packet quantity, and make a new residual flow for what have moved
        # to the next node till dst

        self.flows = list_of_2flows
        # Todo: can update flows with incoming demand here
        # if self.arrival_matrix is not None:
        #     pass
            # for flow_id, flow in enumerate(self.flows):
            #     flow['packets'] += self.arrival_matrix[self.slot_num][flow_id]
        # ------------------------- #
        self.num_flows = len(self.flows)
        self.residual_flows += list_of_new_residuals
        return

    def calc_reward(self, metadata, action):
        curent_flow_idx = action[0]
        current_flow_action = action[1]
        curent_flow_path = self.possible_actions[curent_flow_idx][current_flow_action]
        idx_for_interferance_and_capacity = next((i for i in range(len(self.current_link_interference_list_4EachTimeStep) - 1) if np.all(self.current_link_interference_list_4EachTimeStep[i + 1] == 0)), -1)
        idx_for_interferance_and_capacity -= 1

        # rate reward
        rate_for_all_link_of_current_flow = self.current_link_capacity_list_4EachTimeStep[idx_for_interferance_and_capacity]
        rates_list_for_current_flow = []

        for i in range(len(curent_flow_path) - 1):
            u = curent_flow_path[i]
            v = curent_flow_path[i + 1]
            link_idx = self.eids[(u, v)]
            rates_list_for_current_flow.append(rate_for_all_link_of_current_flow[link_idx])
        rate_reward = np.min(rates_list_for_current_flow)

        # influence reward
        Tot_interence_for_current_flow = sum(metadata[idx_for_interferance_and_capacity]['interference'])
        capacity_reduction = sum(metadata[idx_for_interferance_and_capacity]['capacity_reduction'])
        avg_excess_delay = np.max(self.routing_metrics['delay']['end_to_end_delay_per_flow_for_a_defined_packet_size'])
        influence_reward = Tot_interence_for_current_flow + avg_excess_delay + capacity_reduction

        # rate_weight = self.kwargs.get('reward_weights', dict()).get('rate_weight', 1)
        reward = 0.8 * rate_reward + 0.2 * influence_reward

        # if self.kwargs.get('direction', 'maximize') == 'minimize':
        #     reward *= -1
        #
        # if self.kwargs.get('telescopic_reward'):
        #     if self.prev_reward is not None:
        #         tmp = reward
        #         reward = reward - self.prev_reward
        #         self.prev_reward = tmp
        #     else:
        #         self.prev_reward = reward

        # # zero slot delay for next slot
        # self.routing_metrics['delay']['end_to_end_delay_per_flow_for_a_defined_packet_size'] = np.zeros(self.num_flows)
        return -reward

    def end_of_slot_update(self):
        '''
        this function reset the part in state that is needed to be resets (demands) betwwen each time slot
        and output data for our likings betwwens time slots
        '''

        # gather rate and delay data
        _, Avg_Rate_over_flows = self.get_rates_data()

        _, Avg_Delay_over_flows = self.get_delay_data()

        observation = self.reset()

        return observation, Avg_Rate_over_flows, Avg_Delay_over_flows

    def upate_routing_metrics(self):
        """ update routing metrics """
        # update delay metric
        if len(self.allocated) == len(self.flows):
            self.current_flow_delay_list_4EachTimeStep = [np.where(x == 0, np.nan, x) for x in self.current_flow_delay_list_4EachTimeStep]  # make all zero to nan for all finished flows
            self.routing_metrics['delay']['slot_delay'] = self.current_flow_delay_list_4EachTimeStep

        # reset for the next time slot
        self.current_link_interference_list_4EachTimeStep = []
        self.current_link_capacity_list_4EachTimeStep = []
        self.current_flow_delay_list_4EachTimeStep = []

    def get_delay_data(self):
        data = self.routing_metrics.get('delay')
        slot_delay = data['slot_delay']
        # Avg_Delay_over_flows = [np.mean([slot_delay[i][j] for i in range(len(slot_delay))]) for j in range(len(slot_delay[0]))]

        # data = self.routing_metrics.get('delay')
        # data['excess_delay_per_flow'] = [data['end_to_end_delay_per_flow'][i] - (len(f['path']) - 1) for i, f in enumerate(self.flows)]
        # data['total_excess_delay'] = np.sum(data.get('excess_delay_per_flow'))
        # data['avg_end_to_end_delay'] = np.mean(data.get('end_to_end_delay_per_flow'))
        # data['end_to_end_delay_per_flow'] = data['end_to_end_delay_per_flow']
        return data, slot_delay

    def get_rates_data(self):
        data = self.routing_metrics['rate']['rate_per_flow']
        # data[np.isinf(data)] = 0 # replace all inf to 0  (in case flow finished before slot ends)
        Avg_Rate_over_flows = []

        for second in range(int(self.slot_duration/self.Simulation_Time_Resolution)):
            avg_rate_in_sedond = 0
            divide_by = 0
            for flow in range(data.shape[0]):
                if data[flow, second] != np.inf:
                    avg_rate_in_sedond += data[flow, second]
                    divide_by += 1

            if divide_by == 0:
                avg_rate_in_sedond = None
            else:
                avg_rate_in_sedond = avg_rate_in_sedond / divide_by

            Avg_Rate_over_flows.append(avg_rate_in_sedond)

        # Avg_Rate_over_flows = np.mean(data, axis=0)
        # Avg_Rate_over_time = np.mean(data, axis=1)

        return data, Avg_Rate_over_flows

    def get_state_space(self):
        """ return possible actions indices"""
        return [[i for i, _ in enumerate(a)] for a in self.possible_actions]



    def set_tf_env(self, state=False):
        self.tf_env = state

    def edge_list_to_adj_mat(self, lst):
        mat = np.zeros((self.num_nodes, self.num_nodes))
        for eid, l in enumerate(lst):
            u, v = self.id_to_edge[eid]
            mat[u, v] = l
            mat[v, u] = l
        return mat

    def path_length(self, flow_idx):
        return len(self.flows[flow_idx]['path']) - 1

    def path_link_indexes(self, flow_idx):
        path = self.flows[flow_idx]['path']
        dual_path = []
        for i in range(len(path) - 1):
            dual_path.append(self.eids[path[i], path[i + 1]])
        return dual_path

    def __get_observation(self):
        """ returns |V|x|V|xd matrix representing the graph

        state_matrixes:   the interference, normalized_capacity and last_action matrixes.
        edges:            static? all edges of the graph.
        free_paths:       a list, with all posible routs to assign, for each flow that has not been assign with a rout:
                          the first (action_size) elements are the posible routs for the first flow, the second (action_size) elemnts are the posible routs for the second flow, and so on.
        """

        idx_for_interferance_and_capacity = next((i for i in range(len(self.current_link_interference_list_4EachTimeStep) - 1) if np.all(self.current_link_interference_list_4EachTimeStep[i + 1] == 0)), -1)

        # interference
        if self.current_link_interference_list_4EachTimeStep:
            # interference = self.edge_list_to_adj_mat(self.current_link_interference_list_4EachTimeStep[-1]) # take the last interference
            interference = self.edge_list_to_adj_mat(self.current_link_interference_list_4EachTimeStep[idx_for_interferance_and_capacity])  # take the last interference is worng!, it might be zero if the flow finished
        else:  # for reset
            interference = self.edge_list_to_adj_mat(self.current_link_interference)

        # capacity
        # normalized_capacity = self.edge_list_to_adj_mat(self.current_link_capacity)
        if self.current_link_capacity_list_4EachTimeStep:
            # normalized_capacity = self.edge_list_to_adj_mat(self.current_link_capacity_list_4EachTimeStep[-1])  # take the last capacity
            normalized_capacity = self.edge_list_to_adj_mat(self.current_link_capacity_list_4EachTimeStep[idx_for_interferance_and_capacity])  # take the last capacity
        else:  # for reset
            normalized_capacity = self.edge_list_to_adj_mat(self.current_link_capacity)  # normalized_capacity = np.zeros((self.num_nodes, self.num_nodes)) # Todo: ask ran why zeros

        if self.normalize_capacity:
            normalized_capacity = np.divide(normalized_capacity, self.bandwidth_matrix,
                                            out=np.zeros_like(normalized_capacity), where=self.bandwidth_matrix != 0)
        normalized_capacity *= self.adjacency_matrix
        state_matrixes = np.stack([interference,
                                   normalized_capacity,
                                   self.last_action], axis=-1)

        edges = np.array(self.graph.edges)

        allocated = [a[0] for a in self.allocated]
        free_actions = list(set(range(len(self.flows))) - set(
            allocated))  # unassinged flows, EMPTY free_actions means we are in the last flow in the slot
        free_paths = []
        free_paths_idx = []
        demand = []
        for a in free_actions:
            p = self.possible_actions[a]  # posible routs for unassigned flow a
            free_paths_idx += [[a, k] for k in range(len(p))]
            free_paths += p
            demand += [self.flows[a]["packets"] for k in p]

        # demand
        if free_actions:  # if we are not in the last time step of the time slot, than we can calculate the demand
            normalized_demand = np.array(demand).astype(np.float32) / self.max_demand
            # normalized_demand = np.array(demand).astype(np.float32)

        else:  # if we are in the last time step of the time slot, return initialzed demand
            normalized_demand = None

        if self.is_slotted:
            normalized_demand = (1e0) * np.ones_like(normalized_demand)  # demand = inf for every new flow

        if self.tf_env:
            return self.__get_tf_state()

        interference_deb = state_matrixes[:, :, 0]
        normalized_capacity_deb = state_matrixes[:, :, 1]
        last_action_deb = state_matrixes[:, :, 2]

        return state_matrixes, edges, free_paths, free_paths_idx, normalized_demand

    def __get_tf_state(self):
        """
        :return: state for DQN+GNN agent (tf)
        """
        norm_capacity = (np.array(self.current_link_capacity) - self.max_capacity // 2) / self.max_capacity
        betweenes = np.array(self.num_sps) / (((2.0 * self.num_nodes * (self.num_nodes - 1) * self.k) + 0.00000001))

        last_action = np.zeros((self.num_edges // 2, 3))
        for i, u in enumerate(self.last_action):
            for j, v in enumerate(u):
                if v:
                    last_action[self.eids[(i, j)]][0] = 1

        obs = np.concatenate([
            norm_capacity.reshape(-1, 1),
            betweenes.reshape(-1, 1),
            last_action,
            np.zeros((self.num_edges // 2, 15))
        ], axis=1)

        gids = np.array([0] * (self.num_edges // 2))

        return obs, gids, np.array(self.firsts), np.array(self.seconds), self.num_edges // 2

    def reset(self):
        """ reset environment """
        self.__init_links()
        self.prev_reward = None
        self.allocated = []
        # prev_residuals = self.residual_flows
        # self.residual_flows  = []
        self.routing_metrics = dict(rate=dict(rate_per_flow=np.full([self.num_flows,int(self.slot_duration/self.Simulation_Time_Resolution)],np.inf).astype(np.float64)),
                                    delay=dict(end_to_end_delay_per_flow_for_a_defined_packet_size=np.zeros(self.initial_num_of_flows), slot_delay=[]))

        # Todo: My adding, need to restart possible actions between steps
        self.possible_actions = [[] for _ in range(len(self.flows))]
        self.__calc_possible_actions()

        observation = self.__get_observation()

        # There is am option to output the residuals from previous time slot (prev_residuals)
        return observation

    def step(self, action, eval_path=False, real_run=False):
        """
        action = (flow_idx, channels_per_link)
        :return: next_state, reward
        """

        # simulate transmission of all flows from src->dst and get reward
        reward = self.__simulate_single_slot(action, eval_path=False)

        if real_run:
            return

        # next state
        next_state = self.__get_observation()

        # update routing metrics
        self.upate_routing_metrics()

        # zero all delivered in a slot to calulate delay all over again
        any(d.update({"delivered_in_a_slot": 0}) for d in self.flows)  # zero all delivered in a slot to calulate delay all over again
        return next_state, reward

    def eval(self, action):
        """
        simulate transmission of all flows from src->dst and get reward

        :param action: list of indices of the paths to allocate, one for each flow
        :return: reward for action
        """

        # simulate transmission of all flows from src->dst and get reward
        reward = self.__simulate_single_slot(action, eval_path=False)

        return reward

    def eval_all(self, actions):
        """

        :param actions: list of tuples (flow, path)
        :return: accumulated reward
        """
        self.reset()  # Todo: check if needed here, because in end_of_slot_update there is env.reset()
        rewards = [self.eval((i, a)) for i, a in enumerate(actions)]
        return sum(rewards)

    def get_routs(self, actions):
        if isinstance(actions[0], (list, tuple)):
            return [self.possible_actions[a[0]][a[1]] for a in actions]
        else:
            return [self.possible_actions[i][a] for i, a in enumerate(actions)]


if __name__ == "__main__":
    """sanity check for env and reward"""
    reward_weights = dict(rate_weight=0.5, delay_weight=0, interference_weight=0, capacity_reduction_weight=0)

    # number of nodes
    N = 4

    # Adjacency matrix
    # create 3x3 mesh graph
    A = np.array([[0, 1, 1, 1],  # means how connects to who
                  [1, 0, 1, 1],
                  [1, 1, 0, 1],
                  [1, 1, 1, 0]])

    # P = [(0, 0), (0, 1), (0, 2),                #the position of each node
    #      (1, 0), (1, 1), (1, 2),
    #      (2, 0), (2, 1), (2, 2)]

    P = [(0, 0), (0, 1),  # the position of each node
         (1, 0), (1, 1)]

    # capacity matrix
    C = 1 * np.ones((N, N))
    # C = 100 * np.array([[1, 1, 1, 1],  # means how connects to who
    #                     [1, 1, 1, 1],
    #                     [1, 1, 1, 1],
    #                     [1, 1, 1, 1]])

    # interference matrix
    I = np.ones((N, N)) - np.eye(N)

    # number of paths to choose from
    action_size = 4  # search space limitaions?

    # flow demands
    F = [
        {"source": 0, "destination": 3, "packets": 200, "time_constrain": 10},
        {"source": 0, "destination": 3, "packets": 200, "time_constrain": 10}
    ]

    env = SlottedGraphEnvPower(adjacency_matrix=A,
                               bandwidth_matrix=C,
                               flows=F,
                               node_positions=P,
                               k=action_size,
                               reward_weights=reward_weights,
                               telescopic_reward=True,
                               direction='minimize',
                               render_mode=True)

    # adj_matrix, edges, free_paths, free_paths_idx, _ = env.reset()
    # reward = 0

    # state_debug, r = env.step(action=[0,0])
    # reward += r

    # state_debug, r = env.step(action=[1,1])
    # reward += r

    # state_debug, r = env.step(action=[2,2])
    # reward += r

    # state_debug, r = env.step(action=[3,3])
    # reward += r

    # print(f"rate: {env.get_rates_data().get('avg_flow_rate')}, delay: {env.get_rates_data().get('end_to_end_delay_per_flow')}")
    # print(env.routing_metrics)
    # print("-" * 20)

    adj_matrix, edges, free_paths, free_paths_idx, normalized_demand = env.reset()
    possible_actions = free_paths.copy()
    best_score = -sys.maxsize
    best_routs = []
    best_action = -1
    for i in range(action_size):
        for j in range(action_size):
            state_debug = env.reset()  # interfernce map = state_debug[0][:,:,0]
            reward = 0
            a0 = [0, i]
            state_debug, r = env.step(action=a0)
            reward += r
            a1 = [1, j]
            state_debug, r = env.step(action=a1)
            reward += r
            routs = [possible_actions[i], possible_actions[j + action_size]]
            print(
                f"action {[i, j]} with routs {routs} -> reward: {reward} (rates: {env.get_rates_data().get('avg_flow_rate')}, delay: {env.get_delay_data().get('end_to_end_delay_per_flow')})")

            if reward > best_score:
                best_score = reward
                best_routs = routs.copy()
                best_action = [i, j]

    print("*****************************")
    print("Optimal Solution:")
    print(f"action {best_action} with routs {best_routs} -> reward: {best_score}")