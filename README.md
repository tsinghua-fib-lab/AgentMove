# 🤖 AgentMove
AgentMove: A Large Language Model based Agentic Framework for Zero-shot Next Location Prediction

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
    - osm_address_deploy.py     # Given location coordinates, retrieves nearby addresses using a self-deployed address resolution service for large-scale parallel processing
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
Configure the relevant API Key in .bashrc, then execute source .bashrc
```bash
export SiliconFlow_API_KEY="xx"
export DeepInfra_API_KEY="xx"
export OpenAI_API_KEY="xx"
export vllm_KEY="xx"
```

## Preprocessing
```bash
# download data tsmc2014, tist2015, www2019
python -m processing.download --data_name=www2019
# processing Foursquare data, tist2015, gowalla
python -m processing.process_fsq_city_data
# processing IPS GPS trajectory data www2019
python -m processing.process_isp_shanghai
# get OSM address
python -m processing.osm_address_deploy
# matching trajectory with address
python -m processing.trajectory_address_match
```
## Runing and Evaluation
Agent
```bash
python -m agent --cityname=Beijing --prompt_num=10 --workers=10 --prompt_type=agent_move_v6 --model_name=llama3-8b
```
Evaluation
```bash
python -m evaluate.analysis --eval_path=results/20240505/ --level=city
python -m evaluate.analysis --eval_path=results/20240505/Beijing/agentmove/ --level=agent
python -m evaluate.analysis --eval_path=results/20240505/Beijing/agentmove/llama3-8b/ --level=llm
python -m evaluate.analysis --eval_path=results/20240505/Beijing/agentmove/llama3-8b/agent_move_v6/ --level=prompt
```
Example
```bash
./run_fsq.sh
./run_isp.sh
```

# 🌟 Citation

If you find this work helpful, please cite our paper.

```latex
@article{feng2024agentmove,
  title={AgentMove: A Large Language Model based Agentic Framework for Zero-shot Next Location Prediction},
  author={Feng, Jie and Du, Yuwei and Zhao, Jie and Li, Yong},
  journal={arXiv preprint arXiv:2408.13986},
  year={2024}
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
