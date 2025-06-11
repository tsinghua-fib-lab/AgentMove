# 🤖 AgentMove
AgentMove: A Large Language Model based Agentic Framework for Zero-shot Next Location Prediction

# 📰 News
- 2025.02 We have uploaded a new version of the AgentMove paper! [Check it out here](https://arxiv.org/abs/2408.13986)!
- 2025.01 Our paper has been accepted to [NAACL 2025](https://aclanthology.org/2025.naacl-long.61/) as a main conference paper!.

# 🌍 Introduction
Next location prediction plays a crucial role in various real-world applications. Recently, due to the limitation of existing deep learning methods, attempts have been made to apply large language models (LLMs) to zero-shot next location prediction task. However, they directly generate the final output using LLMs without systematic design, which limits the potential of LLMs to uncover complex mobility patterns and underestimates their extensive reserve of global geospatial knowledge. In this paper, we introduce \textbf{AgentMove}, a systematic agentic prediction framework to achieve generalized next location prediction. 
In AgentMove, we first decompose the mobility prediction task and design specific modules to complete them, including spatial-temporal memory for individual mobility pattern mining, world knowledge generator for modeling the effects of urban structure and collective knowledge extractor for capturing the shared patterns among population. Finally, we combine the results of three modules and conduct a reasoning step to generate the final predictions. 
Extensive experiments utilizing mobility data from two distinct sources reveal that AgentMove surpasses the leading baseline by 3.33\% to 8.57\% across 8 out of 12 metrics and it shows robust predictions with various LLMs as base and also less geographical bias across cities.

![](./assets/framework.png)

# ⌨️ Repo Structures
```
- agent.py                      # Main entry point
    - run_fsq.sh                # Example script
    - run_isp.sh                # Example script
- config.py                     # Parameter configuration
- processing                    # Raw data processing code, for details refer to[README](./scripts/README.md)
    - process_fsq_city_data.py  # Parses city trajectory data from raw global Foursquare check-in data, containing only location coordinates, ID, and category
    - process_isp_shanghai.py   # Processes raw ISP data and matches it with the Foursquare data format for unified handling later
    - osm_address_deploy.py     # Given location coordinates, retrieves nearby addresses using a self-deployed address resolution service for large-scale parallel processing, https://github.com/mediagis/nominatim-docker/tree/master/4.4
    - osm_address_web.py        # Given location coordinates, retrieves nearby addresses using the official address resolution service, suitable for small-scale testing
    - trajectory_address_match.py  # Uses various address services and GPT to match a unified four-level address structure, expanding trajectory points with new four-level address information
    - data.py                   # Final preprocessing functions for the data, no need to call manually, will be invoked by the agent automatically
    - download.py               # download raw dataset
- models
    - personal_memory.py        # Implementation related to the memory module
    - world_model.py            # Implementation related to the world model
    - prompts.py                # Prompt templates for llm-based baselines and agentmove
    - llm_api.py                # unified entry point for all LLM APIs from various LLM providers
- evaluate
    - evaluations.py            # Results statistics for a single model
    - analysis.py               # Calls evaluations.py to analyze and compare multiple models simultaneously and saves the results in results/summary
- serving/*                     # Local deployment of LLM, we use powerful vllm for local deployment
- baselines/*                   # Implementation of baseline algorithms, we use official implementation of each baselines
- utils.py
- assets/*                      # Assets
```

# 💡 Running Experiments


## LLM API Key
Configure the relevant API Key in `.bashrc`, then execute `source .bashrc`
```bash
# provide free tokons for many 7B models
export SiliconFlow_API_KEY="xx"

# following APIs may need proxy
export DeepInfra_API_KEY="xx"
export OpenAI_API_KEY="xx"

# you can also deploy model locally via vllm
export vllm_KEY="xx"

# set nominatim server address if you deploy it locally
export nominatim_deploy_server_address="IP:Port"
```
We define supported models list in `models/llm_api.py`, you can add new models and new platforms by modifying it.

## Installation
```bash
git clone https://github.com/tsinghua-fib-lab/AgentMove.git

cd AgentMove

conda create -n agentmove python==3.10
pip install -r requirements.txt
```

## Preprocessing
```bash
# Step1: you can define city collection in config.py with EXP_CITIES
EXP_CITIES = ["Shanghai"] # for WWW 2019 ISP
# EXP_CITIES = ['Tokyo', 'Nairobi', 'NewYork', 'Sydney', 'CapeTown', 'Paris', 'Beijing', 'Mumbai', 'SanFrancisco', 'London', 'SaoPaulo', 'Moscow'] # for TIST 2015

# download data tist2015(used in paper), www2019(used in paper), tsmc2014
# we have uploaded www2019 dataset in data folder
python -m processing.download --data_name=www2019

# Step2:
# # processing Foursquare data, tist2015, gowalla
# python -m processing.process_fsq_city_data

# processing IPS GPS trajectory data www2019
python -m processing.process_isp_shanghai

# Step3: get open street map address
# A local Nominatim service must be deployed prior to executing these commands. Alternatively, you may utilize the official Nominatim API
python -m processing.osm_address_deploy
# python -m processing.osm_address_web

# Step4: matching trajectory with address
python -m processing.trajectory_address_match
```

## Runing and Evaluation
Agent
```bash
python -m agent --sample_one_traj_of_user \
    --social_info_type=address \
    --traj_min_len=3 \
    --traj_max_len=50 \
    --city_name=Beijing \
    --prompt_num=50 \
    --workers=20 \
    --exp_name=test \
    --prompt_type=agent_move_v6 \
    --model_name=llama4-17b \
    --platform=DeepInfra
```
Evaluation
```bash
python -m evaluate.analysis --eval_path=results/20240505/ --level=city
python -m evaluate.analysis --eval_path=results/20240505/Beijing/agentmove/ --level=agent
python -m evaluate.analysis --eval_path=results/20240505/Beijing/agentmove/llama3-8b/ --level=llm
python -m evaluate.analysis --eval_path=results/20240505/Beijing/agentmove/llama3-8b/agent_move_v6/ --level=prompt
```
More running examples
```bash
./run_fsq.sh
./run_isp.sh
```

# DEBUGING Tips
1. If you encounter any exceptions, you can try relaxing the try-except control in the code to help with debugging.
2. You can refer to the launch.json in .vscode to debug with remote server.

# TODO List
- [ ] update LLM support, e.g., adding "qwen2.5" and "deepseek", openrouter platform
- [ ] add new baselines, e.g., adding "Taming the Long Tail in Human Mobility Prediction"
- [ ] add new datasets, e.g., adding "YJMob100K"


# 🌟 Citation

If you find this work helpful, please cite our paper.

```latex
@inproceedings{feng2025agentmove,
  title={Agentmove: A large language model based agentic framework for zero-shot next location prediction},
  author={Feng, Jie and Du, Yuwei and Zhao, Jie and Li, Yong},
  booktitle={Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={1322--1338},
  year={2025}
}
```

# 👏 Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- https://github.com/ssai-trento/LLM-zero-shot-NL for initial code structure
- https://github.com/vonfeng/DPLink for ISP data
- https://github.com/LibCity/Bigscity-LibCity for baselines
- https://github.com/songyangme/GETNext
- https://github.com/ant-research/Spatio-Temporal-Hypergraph-Model

# 📩 Contact

If you have any questions or want to use the code, feel free to contact:
Jie Feng (fengjie@tsinghua.edu.cn)
