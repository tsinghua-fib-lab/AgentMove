PROMPT_NUM=200
WORKERS=50
PLATFORM=DeepInfra
TRAJ_MIN=3
TRAJ_MAX=50
SOCIAL_TYPE=address
EXP_NAME="20250630-$TRAJ_MIN-$TRAJ_MAX-int"
MODEL_NAME=llama4-17b
MEMORY_LEN=15

CITY="Shanghai"
PROMPT_TYPE="agent_move_v6" # llmzs, llmmob, agent_move_v6, llmmove

# running exp
python -m agent\
 --city_name=$CITY \
 --prompt_type=$PROMPT_TYPE \
 --model_name=$MODEL_NAME \
 --platform=$PLATFORM \
 --prompt_num=$PROMPT_NUM \
 --workers=$WORKERS \
 --exp_name=$EXP_NAME \
 --traj_min_len=$TRAJ_MIN \
 --traj_max_len=$TRAJ_MAX \
 --social_info_type=$SOCIAL_TYPE \
 --memory_lens=$MEMORY_LEN \
 --skip_existing_prediction \
 --sample_one_traj_of_user \
 --use_int_venue

# analysis results
python -m evaluate.analysis --eval_path="results/$EXP_NAME/" --level=city --use_int_venue
