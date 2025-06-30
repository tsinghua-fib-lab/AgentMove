from utils import haversine_distance

COMMON_PROMPT = """
## Task
Your task is to predict <next_place_id> in <target_stay>, a location with an unknown ID, while temporal data is available.

## Predict <next_place_id> by considering:
1. The user's activity trends gleaned from <historical_stays> and the current activities from  <context_stays>.
2. Temporal details (start_time and day_of_week) of the target stay, crucial for understanding activity variations."""

OUTPUT_PROMPT = """
## Output 
Present your answer in a JSON object with:
"prediction" (list of IDs of the five most probable places, ranked by probability) and "reason" (a concise justification for your prediction).
"""

def prompt_generator(v, prompt_type, spatial_world_info, memory_info, social_world_info, rec):
    prompt = ''
    if 'origin' in prompt_type or "llmzs" in prompt_type:
        prompt = prompt_generator_llmzs(v)
    elif "agent" in prompt_type:
        prompt = prompt_generator_agent(v, prompt_type, spatial_world_info, memory_info, social_world_info)
    elif "llmmob" in prompt_type:
        prompt = prompt_generator_llmmob(v)
    elif "llmmove" in prompt_type:
        prompt = prompt_generator_llmmove(v, rec)
    return prompt


def prompt_generator_llmmob(v):
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. Note that this will be None in the <target_stay> introduced later.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
    unknown duration denoted as None, while temporal information is provided.      

    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you learned from <history>, e.g., repeated visits to certain places during certain times;
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and day_of_week) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the five most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

    The data are as follows:
    <historical>: {[[item[0], item[1], item[3]] for item in v['historical_stays']]}
    <context>: {[[item[0], item[1], item[3]] for item in v['context_stays']]}
    <target_stay>: {[v['target_stay'][0], v['target_stay'][1]]}
    """
    return prompt


def prompt_generator_llmzs(v):
    prompt = f"""
    		Your task is to predict <next_place_id> in <target_stay>, a location with an unknown ID, while temporal data is available.

                Predict <next_place_id> by considering:
                1. The user's activity trends gleaned from <historical_stays> and the current activities from  <context_stays>.
                2. Temporal details (start_time and day_of_week) of the target stay, crucial for understanding activity variations.

                Present your answer in a JSON object with:
                "prediction" (IDs of the five most probable places, ranked by probability) and "reason" (a concise justification for your prediction).
                
		The data:
                    <historical_stays>: {[[item[0],item[1],item[3]] for item in v['historical_stays']]}
                    <context_stays>: {[[item[0],item[1],item[3]] for item in v['context_stays']]}
                    <target_stay>: {[v['target_stay'][0], v['target_stay'][1]]}
                   """
    return prompt


def prompt_generator_agent(v, prompt_type, spatial_world_info, memory_info, social_world_info):
    prompt = ''
    if prompt_type == "agent_move_v6":
        prompt = f"""
{COMMON_PROMPT}
3. The potential places that users may visit based on an overall analysis of multi-level urban spaces.
4. The personal profile and memory info extracted from the long trajectory history of each user.


## The potential places from the global spatial view:
{spatial_world_info}

## The nearby places visited by other users with similar mobility pattern:
{social_world_info}

## The personal profile and long memory:
<historical_info>: {memory_info['historical_info']}
<user_profile>: {memory_info['user_profile']}

## The history data:
<historical_stays>: {[[item[0], item[1], item[2], item[3], ",".join((item[5],item[7],item[6]))] for item in v['historical_stays']]}
<context_stays>: {[[item[0], item[1], item[2], item[3], ",".join((item[5],item[7],item[6]))] for item in v['context_stays']]}
<target_stay>: {[v['target_stay'][0], v['target_stay'][1], v['target_stay'][2]]}

{OUTPUT_PROMPT}
"""

    return prompt


def prompt_generator_llmmove(v, rec):
    prompt =f"""\
<long-term check-ins> [Format: (POIID, Category)]: {[(item[3],item[2]) for item in v['historical_stays']]}
<recent check-ins> [Format: (POIID, Category)]: {[(item[3],item[2]) for item in v['context_stays']]}
<candidate set> [Format: (POIID, Distance, Category)]: {[(item['poi'],  haversine_distance(item['pos'][1],item['pos'][0],v['context_pos'][-1][1],v['context_pos'][-1][0]), item['cat']) for _, item in rec.items()]}
Your task is to recommend a user's next point-of-interest (POI) from <candidate set> based on his/her trajectory information.
The trajectory information is made of a sequence of the user's <long-term check-ins> and a sequence of the user's <recent check-ins> in chronological order.
Now I explain the elements in the format. "POIID" refers to the unique id of the POI, "Distance" indicates the distance (kilometers) between the user and the POI, and "Category" shows the semantic information of the POI.
Requirements:
1. Consider the long-term check-ins to extract users' long-term preferences since people tend to revisit their frequent visits.
2. Consider the recent check-ins to extract users' current perferences.
3. Consider the "Distance" since people tend to visit nearby pois.
4. Consider which "Category" the user would go next for long-term check-ins indicates sequential transitions the user prefer.
Please organize your answer in a JSON object containing following keys:
"recommendation" (10 distinct POIIDs of the ten most probable places in descending order of probability), and "reason" (a concise explanation that supports your recommendation according to the requirements). Do not include line breaks in your output.
"""
    return prompt