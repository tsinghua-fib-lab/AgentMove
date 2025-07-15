import os
import json
import time
import tqdm
import random
import argparse
import multiprocessing
from datetime import datetime

from processing.data import Dataset
from models.llm_api import LLMWrapper
from models.world_model import SpatialWorld, SocialWorld
from models.personal_memory import Memory
from models.prompts import prompt_generator
from utils import create_dir, extract_json
from config import PROXY, PROCESSED_DIR
random.seed(100)


class Agent:
    def __init__(self, city_name, platform, model_name, spatial_world: SpatialWorld, social_world: SocialWorld, memory_unit: Memory, prompt_type, save_dir, use_int_venue, social_info_type):
        self.city_name = city_name
        self.platform = platform
        self.model_name = model_name

        self.llm_model = LLMWrapper(model_name, platform)
        self.spatial_world = spatial_world
        self.social_world = social_world
        self.memory_unit = memory_unit

        self.prompt_type = prompt_type
        self.save_dir = save_dir
        self.use_int_venue = use_int_venue
        self.social_info_type = social_info_type

    def predict(self, user_id, traj_id, traj_seqs, target_stay, true_value):

        # spatial world model info
        spatial_world_info = self.spatial_world.get_world_info()

        # personal memory
        memory_info = self.memory_unit.read_memory(user_id, target_stay)

        # social world mdoel
        last_venue_id = traj_seqs["context_stays"][-1][3]
        self_history_points = [x[3] for x in traj_seqs["context_stays"]]
        social_world_info = self.social_world.get_world_info(last_venue_id, self_history_points, self.social_info_type)

        # final prompt
        prompt_text = prompt_generator(traj_seqs, self.prompt_type, spatial_world_info, memory_info, social_world_info)
        pre_text = self.llm_model.get_response(prompt_text=prompt_text)
        output_json, prediction, reason = extract_json(pre_text)
        
        # true_addr = true_value["ground_addr"]
        true_venue = true_value["ground_stay"]

        predictions = {
            'input': prompt_text,
            'output': output_json,
            'prediction': prediction,
            'reason': reason,
            'true': true_venue  
        }

        # Construct the filename with model type and save to file
        filename = f"{self.llm_model.model_name}_{self.prompt_type}_{user_id}_{traj_id}_{self.use_int_venue}.json"
        file_path = os.path.join(self.save_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(predictions, f, indent=4)


class Agents:
    def __init__(self, platform, model_name, prompt_type, city_name, prompt_num, use_int_venue, dataset: Dataset, workers=1, exp_name="",
                 traj_min_len=3, traj_max_len=100, sample_one_traj_of_user=False, social_world: SocialWorld=None, social_info_type='address', memory_lens=15,
                 skip_existing_is_on=False, max_explore_places=5, max_sample_trajectories=1):
        self.city_name = city_name
        self.platform = platform
        self.model_name = model_name
        self.prompt_type = prompt_type
        self.prompt_num = prompt_num
        self.exp_name = exp_name
        self.traj_min_len = traj_min_len
        self.traj_max_len = traj_max_len
        self.sample_one_traj_of_user = sample_one_traj_of_user
        self.social_world = social_world
        self.social_info_type = social_info_type
        self.memory_lens = memory_lens
        self.skip_existing_is_on = skip_existing_is_on
        self.max_explore_places = max_explore_places
        self.max_sample_trajectories = max_sample_trajectories

        # test_dictionary, true_locations
        test_dataset, self.ground_data = dataset.get_generated_datasets()
        self.trajectories = []
        self.trajectory_groups = []
        self.known_stays = {}
        self.use_int_venue = use_int_venue
        self.workers = workers
        self.save_dir = os.path.join("results/", self.exp_name, self.city_name, "agentmove/", self.model_name, self.prompt_type)
        create_dir(self.save_dir)

        self.trajs_sampling(test_dataset)


    def trajs_sampling(self, test_dataset):
        counter = 0
        user_list = [str(y) for y in sorted([int(x) for x in list(test_dataset.keys())])]
        for user_id in user_list:
            v = test_dataset[user_id]
            traj_ids = [str(y) for y in sorted([int(x) for x in list(v.keys())])]

            if self.city_name in ["Shanghai"]:
                if len(traj_ids)==0:
                    continue
            else:
                if len(traj_ids) < self.traj_min_len:
                    continue
                if len(traj_ids) > self.traj_max_len:
                    continue

            traj_list = []
            traj_count=0
            for traj_id in traj_ids:
                self.trajectories.append((user_id, traj_id, v[traj_id]))
                traj_list.append((user_id, traj_id, v[traj_id]))

                counter += 1
                traj_count += 1
                if self.sample_one_traj_of_user:
                    break
                else:
                    if traj_count>self.max_sample_trajectories:
                        break

            self.trajectory_groups.append(tuple(traj_list))
            self.known_stays[user_id] = v[traj_ids[0]]["historical_stays_long"]

            if counter >=self.prompt_num:
                print("Data is prepared, Except:{} Real:{} Users:{}".format(self.prompt_num, counter, len(self.trajectory_groups)))
                break
        if counter < self.prompt_num:
            print("Data is not enough, Except:{} Real:{} Users:{}".format(self.prompt_num, counter, len(self.trajectory_groups)))


    def skip_existing_file(self, user_id, traj_id, ):
        filename = f"{self.model_name}_{self.prompt_type}_{user_id}_{traj_id}_{self.use_int_venue}.json"
        file_path = os.path.join(self.save_dir, filename)
        return os.path.exists(file_path)


    def get_predictions(self):
        if self.workers==1:
            for traj in tqdm.tqdm(self.trajectories):
                user_id, cur_context_stays = self.single_prediction(traj)
                self.known_stays[user_id].extend(cur_context_stays)
        elif args.sample_one_traj_of_user:
            with multiprocessing.Pool(self.workers) as pool:
                res = pool.starmap(self.single_prediction, [(traj, None) for traj in self.trajectories])
        else:
            manager = multiprocessing.Manager()
            know_stays_parallel = manager.dict()
            for u in self.known_stays:
                know_stays_parallel[u] = self.known_stays[u]
            shared_dict = manager.dict()
            shared_dict['counter'] = manager.list([0])
            shared_dict['data'] = know_stays_parallel
            shared_dict['lock'] = manager.Lock()

            with multiprocessing.Pool(min(self.workers, len(self.trajectory_groups))) as pool:
                res = pool.starmap(self.single_prediction_group, [(traj_groups, shared_dict) for traj_groups in self.trajectory_groups])

    def single_prediction_group(self, trajs, shared_dict):
        for traj in trajs:
            user_id, cur_context_stays = self.single_prediction(traj, shared_dict)
            with shared_dict["lock"]:
                shared_dict['counter'][0] += 1
                shared_dict['data'][user_id].extend(cur_context_stays)
            # self.known_stays[user_id].extend(cur_context_stays)

    def single_prediction(self, traj, shared_dict=None):
        user_id, traj_id, traj_seqs = traj

        if self.skip_existing_is_on and self.skip_existing_file(user_id=user_id, traj_id=traj_id):
            return (user_id, traj_seqs.get('context_stays', []))

        # spatial world model
        spaital_world = SpatialWorld(
            model_name=self.model_name,
            platform=self.platform,
            city_name=self.city_name,
            traj_seqs=traj_seqs,
            explore_num=self.max_explore_places
            )
        # personal memory
        cur_context_stays = traj_seqs.get('context_stays', [])
        target_stay = traj_seqs.get('target_stay', [])
        
        if self.workers==1 or self.sample_one_traj_of_user:
            cur_know_stays = self.known_stays[user_id]
        else:
            with shared_dict["lock"]:
                cur_know_stays = shared_dict['data'][user_id]
        
        memory_unit = Memory(
            know_stays=cur_know_stays,
            context_stays=cur_context_stays,
            memory_lens=self.memory_lens
        )
        
        # agent
        agent = Agent(
            city_name = self.city_name,
            platform=self.platform,
            model_name=self.model_name,
            spatial_world=spaital_world,
            social_world=self.social_world,
            memory_unit=memory_unit,
            prompt_type=self.prompt_type,
            save_dir=self.save_dir,
            use_int_venue=self.use_int_venue,
            social_info_type=self.social_info_type
        )

        # predict
        true_value = self.ground_data[user_id][traj_id]
        agent.predict(user_id, traj_id, traj_seqs, target_stay, true_value)

        return (user_id, cur_context_stays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, default="Shanghai")
    parser.add_argument('--model_name', type=str, default="qwen2.5-7b")
    parser.add_argument('--platform', type=str, default="SiliconFlow", choices=["SiliconFlow", "OpenAI", "DeepInfra", "vllm","OpenRouter"])
    parser.add_argument('--trajectory_mode', type=str, default="trajectory_split", choices=["trajectory_split"])
    parser.add_argument("--historical_stays", type=int, default=15)
    parser.add_argument('--context_stays', type=int, default=6)
    parser.add_argument('--traj_min_len', type=int, default=3)
    parser.add_argument('--traj_max_len', type=int, default=10)
    parser.add_argument('--prompt_num', type=int, default=5)
    parser.add_argument('--sample_one_traj_of_user', action='store_true',)
    parser.add_argument('--max_sample_trajectories', type=int, default=100)
    parser.add_argument('--use_int_venue', action='store_true', help='Use int Venue ID')
    parser.add_argument('--prompt_type', type=str, default="agent_move_v6", choices=["agent_move_v6", "origin", "llmmob", "llmzs"])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--social_info_type', type=str, default="address")
    parser.add_argument('--memory_lens', type=int, default=15)
    parser.add_argument('--skip_existing_prediction', action='store_true')
    parser.add_argument('--max_neighbors', type=int, default=10)
    parser.add_argument('--max_explore_places', type=int, default=5)

    args = parser.parse_args()
    if args.model_name in ['gpt35turbo', 'gpt4omini', 'gpt4o', 'gpt4turbo']:
        args.platform = "OpenAI"
    print("INFO START TIME:{}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("args:{}".format(args.__dict__))

    print("runnning experiment in city@{} model@{} samples:{} type:{}".format(args.city_name, args.model_name, args.prompt_num, args.prompt_type))
    start_time = time.time()
    dataset = Dataset(
        dataset_name=args.city_name,
        traj_min_len=2 if args.city_name in ["Shanghai"] else 3,
        trajectory_mode=args.trajectory_mode, 
        historical_stays=args.historical_stays,
        context_stays=args.context_stays,
        save_dir=PROCESSED_DIR,
        use_int_venue=args.use_int_venue,
        )
    social_world = SocialWorld(
        traj_dataset=dataset,
        save_dir=PROCESSED_DIR,
        city_name=args.city_name,
        khop=1,
        max_neighbors=args.max_neighbors
    )

    agents = Agents(
        city_name=args.city_name,
        platform=args.platform,
        model_name=args.model_name,
        prompt_type=args.prompt_type,
        prompt_num=args.prompt_num,
        use_int_venue=args.use_int_venue,
        dataset=dataset,
        workers=args.workers,
        exp_name=args.exp_name,
        traj_max_len=args.traj_max_len,
        traj_min_len=args.traj_min_len,
        sample_one_traj_of_user=args.sample_one_traj_of_user,
        social_world=social_world,
        social_info_type=args.social_info_type,
        memory_lens=args.memory_lens,
        skip_existing_is_on=args.skip_existing_prediction,
        max_explore_places=args.max_explore_places,
        max_sample_trajectories=args.max_sample_trajectories
        )
    agents.get_predictions()
    print("runnning experiment within {} seconds".format(int(time.time()-start_time)))
