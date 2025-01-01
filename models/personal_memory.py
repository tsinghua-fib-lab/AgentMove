import pandas as pd
from .llm_api import LLMWrapper

class Memory:
    def __init__(self, know_stays, context_stays, memory_lens=15):
        self.long_term_memory = {}
        self.short_term_memory = []
        self.user_profile = ""
        self.memory_str_len=1000
        self.memory_lens=memory_lens
        
        input_lens = min(len(know_stays), self.memory_lens)
        self.write_memory(known_stays=know_stays[-input_lens:], context_stays=context_stays)


    def write_memory(self, known_stays, context_stays):

        """ 1) known_stays --> self.memory['long_term_memory'] """
        known_stays_slim = []
        for traj in known_stays:
            known_stays_slim.append(traj[:4])
        context_stays_slim = []
        for traj in context_stays:
            context_stays_slim.append(traj[:4])
        
        venue_mapping = {entry[-1]: entry[-2] for entry in known_stays_slim}

        df = pd.DataFrame(known_stays_slim, columns=['Hour', 'Weekday', 'Venue_Category_Name', 'Venue_ID'])
        # df['Time_Period'] = df['Hour'].apply(self.get_time_period)

        k = 5
        hour_counts = df['Hour'].value_counts().sort_values(ascending=False)
        top_hours = hour_counts.head(k).reset_index()
        top_hours.columns = ['Hour', 'Count']


        venue_counts = df['Venue_Category_Name'].value_counts().sort_values(ascending=False)
        top_venues = venue_counts.head(k).reset_index()
        top_venues.columns = ['Venue_Category_Name', 'Count']


        hourly_venue_counts = df.groupby(['Hour', 'Venue_Category_Name']).size().reset_index(name='Count')
        hourly_venue_summary = hourly_venue_counts.groupby('Hour').apply(
            lambda x: x.nlargest(1, 'Count')).reset_index(drop=True)


        df['Next_Venue'] = df['Venue_Category_Name'].shift(-1)
        df['Transition'] = df['Venue_Category_Name'] + ' -> ' + df['Next_Venue']
        transition_counts = df['Transition'].value_counts().reset_index()
        transition_counts.columns = ['Transition', 'Count']

        self.long_term_memory = {"venue_id_to_name": venue_mapping,
                                f"top_k_frequent_hours": top_hours.to_dict('records'),
                                f"top_k_frequent_venues": top_venues.to_dict('records'),
                                "hourly_venue_count": hourly_venue_summary.to_dict('records'),
                                f"activity_transition": transition_counts.to_dict('records')}

        """ 2) context_stays --> self.memory['short_term_memory'] """
        self.short_term_memory = {
            'last_visit': {},
            'frequent_locations': {},
            'visit_times': {}
        }

        for entry in context_stays_slim:
            time, day, location, id_ = entry


            self.short_term_memory['last_visit'] = {
                'time': time,
                'day': day,
                'location': location,
                'venue_id': id_
            }

            if location not in self.short_term_memory['frequent_locations']:
                self.short_term_memory['frequent_locations'][location] = 0
            self.short_term_memory['frequent_locations'][location] += 1


            if day not in self.short_term_memory['visit_times']:
                self.short_term_memory['visit_times'][day] = []
            self.short_term_memory['visit_times'][day].append({
                'time': time,
                'location': location,
                'id': id_
            })

    def memory_compress(self, memory_prompt):
        if len(memory_prompt)>=self.memory_str_len*2:
            return memory_prompt[:self.memory_str_len]+"\n......\n"+memory_prompt[-self.memory_str_len:]

    def read_memory(self, user_id, target_stay):
        long_mem_prompt = self.long_term_memory_readout(self.long_term_memory)
        if self.memory_lens==0:
            long_mem_prompt = self.memory_compress(long_mem_prompt)
        short_mem_prompt = self.short_term_memory_readout(self.short_term_memory)
        user_profile_prompt = self.user_profile_generation(self.long_term_memory)
        data = {'historical_info': long_mem_prompt, 'context_info': short_mem_prompt,
                'user_profile': user_profile_prompt, 'target_stay': target_stay}
        return data

    @staticmethod
    def long_term_memory_readout(long_mem):
        venue_id_to_name = long_mem['venue_id_to_name']
        top_k_frequent_hours = long_mem['top_k_frequent_hours']
        top_k_frequent_venues = long_mem['top_k_frequent_venues']
        hourly_venue_count = long_mem['hourly_venue_count']
        activity_transition = long_mem['activity_transition']

        frequent_hours = ", ".join([f"{item['Hour']} ({item['Count']} times)" for item in top_k_frequent_hours])

        frequent_venues = ", ".join(
            [f"{item['Venue_Category_Name']} ({item['Count']} times)" for item in top_k_frequent_venues])

        hourly_activity = {}
        for item in hourly_venue_count:
            hour = item['Hour']
            venue = item['Venue_Category_Name']
            count = item['Count']
            if hour not in hourly_activity:
                hourly_activity[hour] = []
            hourly_activity[hour].append(f"{venue} ({count} times)")

        hourly_activity_desc = ", ".join([f"{hour}: {', '.join(venues)}" for hour, venues in hourly_activity.items()])

        transitions = ", ".join([f"{item['Transition']} ({item['Count']} times)" for item in activity_transition])

        long_mem_prompt = (
            f"place id to name mapping: {venue_id_to_name}. "
            f"In historical stays, The user frequently engages in activities at {frequent_hours}. "
            f"The most frequently visited venues are {frequent_venues}. "
            f"Hourly venue activities include {hourly_activity_desc}. "
            f"The user's activity transitions often include sequences such as {transitions}."
        )
        return long_mem_prompt

    @staticmethod
    def short_term_memory_readout(memory):

        last_visit = memory['last_visit']
        frequent_locations = memory['frequent_locations']
        visit_times = memory['visit_times']


        short_mem_prompt = f"In recent context Stays, User's last visit was on {last_visit['day']} at {last_visit['time']} to {last_visit['location']} (ID: {last_visit['venue_id']}). "
        short_mem_prompt += "Frequently visited locations include: " + ", ".join(
            [f"{loc} ({count} times)" for loc, count in frequent_locations.items()]) + ". "
        short_mem_prompt += "Visit times: " + "; ".join([f"{day}: " + ", ".join(
            [f"{entry['time']} at {entry['location']} (ID: {entry['id']})" for entry in entries]) for day, entries in
                                                         visit_times.items()]) + "."

        return short_mem_prompt

    @staticmethod
    def user_profile_generation(long_mem):
        top_k_frequent_hours = long_mem['top_k_frequent_hours']
        top_k_frequent_venues = long_mem['top_k_frequent_venues']

        most_frequent_hour = max(top_k_frequent_hours, key=lambda x: x['Count'])
        frequent_hours = ", ".join([f"{item['Hour']} ({item['Count']} times)" for item in top_k_frequent_hours])

        most_frequent_venue = max(top_k_frequent_venues, key=lambda x: x['Count'])
        frequent_venues = ", ".join(
            [f"{item['Venue_Category_Name']} ({item['Count']} times)" for item in top_k_frequent_venues])

        insights = []

        evening_hours = {'5 PM', '6 PM', '8 PM'}
        if any(item['Hour'] in evening_hours for item in top_k_frequent_hours):
            insights.append("enjoys evening activities")

        regular_hours = {'8 AM', '9 AM', '5 PM', '6 PM'}
        if any(item['Hour'] in regular_hours for item in top_k_frequent_hours):
            insights.append("maintains a regular lifestyle")

        nightlife_hours = {'10 PM', '11 PM', '12 AM', '1 AM', '2 AM'}
        if any(item['Hour'] in nightlife_hours for item in top_k_frequent_hours):
            insights.append("enjoys nightlife")

        commuter_venues = {'Bus Station', 'Train Station'}
        if any(item['Venue_Category_Name'] in commuter_venues for item in top_k_frequent_venues):
            insights.append("has a fixed commute")

        leisure_venues = {'Beach', 'Park', 'Cafe', 'Food & Drink Shop', 'Restaurant'}
        if any(item['Venue_Category_Name'] in leisure_venues for item in top_k_frequent_venues):
            insights.append("enjoys leisure activities")

        shopping_venues = {'Department Store', 'Clothing Store', 'Cosmetics Shop'}
        if any(item['Venue_Category_Name'] in shopping_venues for item in top_k_frequent_venues):
            insights.append("frequently shops for clothes and cosmetics")

        health_venues = {'Gym / Fitness Center'}
        if any(item['Venue_Category_Name'] in health_venues for item in top_k_frequent_venues):
            insights.append("is health conscious and regularly visits the gym")

        food_venues = {'Burger Joint', 'Thai Restaurant', 'Coffee Shop', 'Food & Drink Shop'}
        if any(item['Venue_Category_Name'] in food_venues for item in top_k_frequent_venues):
            insights.append("enjoys trying different types of food and drinks")

        user_profile = (
            f"The user is most active at {most_frequent_hour['Hour']} with {most_frequent_hour['Count']} visits. "
            f"They frequently visit {most_frequent_venue['Venue_Category_Name']} with {most_frequent_venue['Count']} visits."
            # f"Other frequent activity times include: {frequent_hours}. "
            # f"Other frequently visited venues include: {frequent_venues}. "
            f"Based on the data, the user {', '.join(insights)}."
        )

        return user_profile
