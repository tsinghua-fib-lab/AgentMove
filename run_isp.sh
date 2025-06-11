PROMPT_NUM=200
WORKERS=50
PLATFORM=OpenAI
TRAJ_MIN=3
TRAJ_MAX=50
SOCIAL_TYPE=address
EXP_NAME="20240823-$TRAJ_MIN-$TRAJ_MAX-int"
MODEL_NAME=gpt4omini
MEMORY_LEN=15

CITY="Shanghai"
PROMPT_TYPE="llmzs" # llmzs, llmmob, agent_move_v6

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
 --sample_one_traj_of_user \
 --skip_existing_prediction \
 --use_int_venue

# analysis results
python -m evaluate.analysis --eval_path="results/$EXP_NAME/" --level=city --use_int_venue
