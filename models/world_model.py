import os
import glob
import networkx as nx
import itertools
import pandas as pd
from collections import Counter

from .llm_api import LLMWrapper
from config import CITY_DATA_DIR


class SpatialWorld:
    """
    World Knowledge Generator
    """
    def __init__(self, platform, model_name, city_name, traj_seqs, explore_num=5):
        self.city_name = city_name
        self.max_lens = 1000
        self.max_history = 50
        
        self.llm = LLMWrapper(model_name, platform)

        his_addresses_len = min(len(traj_seqs['historical_addr']), self.max_history)
        traj_pos = [[his[0],his[1],his[3],his[2]] for his in traj_seqs['historical_addr'][-his_addresses_len:]]+[[his[0],his[1],his[3],his[2]] for his in traj_seqs['context_addr']]
        administrative_info = list(set([addr[0] for addr in traj_pos]))
        subdistrict_info = [addr[1] for addr in traj_pos]
        poi_info = [(addr[3], addr[2]) for addr in traj_pos]

        self.administrative_area = administrative_info
        self.subdistrict = subdistrict_info
        self.poi = poi_info
        self.explore_num = explore_num

        # the world model can be a structured dictionary
        self.world_model = self.build_inner_world_model()


    def get_world_info(self):
         # world_info_prompt = f"""The range of trajectory movement is primarily within {self.POI_name},{self.street_name},{self.AOI_name},{self.region_name},{self.city_name}. The geographical features of the moving area are as follows:{self.city_name}:{world_info.get("city","")}\n{self.region_name}:{world_info.get("region","")}\n{self.AOI_name}:{world_info.get("AOI","")}\n{self.street_name}:{world_info.get("street","")}\n{self.POI_name}:{world_info.get("POI","")}"""

        world_info_prompt = f"""
### Names of subdistricts that are relatively likely to be visited:\n{self.world_model.get("subdistrict","")}
### Names of POIs that are relatively likely to be visited:\n{self.world_model.get("poi","")}
        """
        if len(world_info_prompt) <= self.max_lens:
            return world_info_prompt
        else:
            return world_info_prompt[-self.max_lens:]


   # build the initial world model by LLM itself
    def build_inner_world_model(self):
        world_info = {}

        subdistrict_pre = f"This trajectory moves within following administrative areas:\n{self.administrative_area}\nThis trajectory sequentially visited following subdistricts, with the last subdistrict being the most recently visited:\n"+";".join([str(item) for item in self.subdistrict])
        poi_pre = "This trajectory sequentially visited following POIs(Each POI is represented by 'POI name, the feeder road or access road it is on'), with the last POI being the most recently visited:\n"+";".join([str(item) for item in self.poi])
        subdistrict_post = "Consider about following two aspects:\n1.The frequency each subdistrict is visited.\n2.Transition probability between two administrative areas.\nPlease predict the next subdistrict in the trajectory. Give {} subdistricts that are relatively likely to be visited. Do not output other content.".format(self.explore_num)
        poi_post = "Consider about following two aspects:\n1.The frequency each subdistrict is visited\n2.The frequency each poi is visited\n3.Transition probability between two subdistricts.\n4.Transition probability between two pois.Please predict the next poi in the trajectory.Give {} POIs that are relatively likely to be visited. Do not output other content.".format(self.explore_num)


        world_info["subdistrict"] =  self.llm.get_response(prompt_text=subdistrict_pre+subdistrict_post)
        world_info["poi"] =  self.llm.get_response(prompt_text=poi_pre+poi_post)     
        return world_info  #world_info

    # build the world model by training data and other trajectories
    def build_inner_world_model_v2(self):
        return {}
    
    # update the world model with external resources to debias
    def update_world_with_outter(self):
        return {}


class SocialWorld:
    """
    Collective Knowledge Extractor
    """
    def __init__(self, traj_dataset, save_dir, city_name, khop=1, max_neighbors=10) -> None:
        self.save_dir = save_dir
        self.city_name = city_name
        self.save_name = "{}_graph.gml".format(city_name)
        self.graph_file_path = os.path.join(self.save_dir, self.save_name)

        self.khop = khop
        self.max_neighbors = max_neighbors

        test_dataset, _ = traj_dataset.get_generated_datasets()
        self.get_processed_graph(test_dataset)


    def build_graph(self, traj_dataset):
        edges_list = []
        nodes_list = []
        for uid in traj_dataset.keys():
            traj_ids = list(traj_dataset[uid].keys())
            if len(traj_ids) == 0:
                continue
            traj_id = traj_ids[0]
            train_instance = traj_dataset[uid][traj_id]["historical_stays_long"]
            venue_ids = [x[3] for x in train_instance] # venue_id
            nodes_list.append([[x[3], x[2], x[4], x[5], x[6], x[7]] for x in train_instance]) # ['hour', 'weekday', 'venue_category_name', venue_id_type, "admin", "subdistrict", "poi", "street"]
            traj_edges = list(zip(venue_ids[:-1], venue_ids[1:]))
            edges_list.append(traj_edges)

        edges = list(itertools.chain.from_iterable(edges_list))
        nodes = list(itertools.chain.from_iterable(nodes_list))
        nodes_df = pd.DataFrame(data=nodes, columns=["venue_id", "venue_category_name",  "admin", "subdistrict", "poi", "street"])

        edges_weights = list(Counter(edges).items())
        edges_final = []
        for edge in edges_weights:
            edges_final.append([edge[0][0], edge[0][1], edge[1]])
        edges_df = pd.DataFrame(edges_final, columns=["src", "dst", "weight"])
        self.graph = nx.from_pandas_edgelist(df=edges_df, source="src", target="dst", edge_attr=["weight"])
        
        for _, row in nodes_df.iterrows():
            node = row['venue_id']
            self.graph.nodes[node]['category'] = row['venue_category_name']
            self.graph.nodes[node]['admin'] = row['admin']
            self.graph.nodes[node]['subdistrict'] = row['subdistrict']
            self.graph.nodes[node]["street"] = row["street"]
            self.graph.nodes[node]['poi'] = row['poi']
        
        nx.write_gml(self.graph, self.graph_file_path)


    def get_processed_graph(self, traj_dataset):
        for file in glob.glob(os.path.join(self.save_dir, "*")):
            if self.save_name in file:
                print("Loading existing graph from:{}".format(file))
                self.graph = nx.read_gml(self.graph_file_path)
                break
        else:
            print("Building new graph in:{}".format(self.graph_file_path))
            self.build_graph(traj_dataset)


    def retrival_neighbors(self, venue_id, context_trajs):
        try:
            if venue_id not in self.graph.nodes():
                return []
            else:
                if self.khop==1:
                    neighbors = list(self.graph.neighbors(venue_id))
                    sorted_neighbors_freq = [(n, 1) for n in neighbors if n not in context_trajs]
                else:
                    lengths = nx.single_source_shortest_path_length(self.graph, venue_id, cutoff=self.khop)
                    neighbors = [(neighbor, length) for neighbor, length in lengths.items() if (1 <= length <= self.khop) and (neighbor not in context_trajs)]
                    sorted_neighbors_freq = sorted(neighbors, key=lambda x: x[1])

            return sorted_neighbors_freq
        except:
            return []


    def get_world_info(self, venue_id, context_traj, type="all"):
        neighbors_sorted = self.retrival_neighbors(venue_id, context_traj)
        neighbors_info = {}
        count = 0
        for n, f in neighbors_sorted:
            category = self.graph.nodes[n]["category"]
            street = self.graph.nodes[n]["street"]
            poi = self.graph.nodes[n]["poi"]
            
            if type=="all":
                info = ",".join([n, category, street, poi])
            elif type=="category":
                info = category
            elif type=="address":
                info = ",".join([street, poi])
            elif type=="id":
                info = n
            else:
                info = ",".join([n, category, street, poi])
            
            if f in neighbors_info:
                neighbors_info[f].append(info)
            else:
                neighbors_info[f] = [info]
            if count >=self.max_neighbors:
                break
            count += 1
        prompts = []
        for f in neighbors_info:
            prompt_text = """{}-hop neighbor places in the social world:\n {}""".format(f, "\n".join(neighbors_info[f]))
            prompts.append(prompt_text)
        return "\n".join(prompts)
