import sys
import os
import numpy as np
import json
import re
from datetime import datetime
from scipy.stats import skewtest, mannwhitneyu
from datetime import timedelta
import copy
import spacy
import shutil

with open("results/results_per_user.json", "r") as file:
    stats = json.load(file)
with open("user_study_data/setup_per_user.json", "r") as file:
    setup_per_user = json.load(file)

if os.path.exists("qualitative_analysis/selected_cases/"):
    shutil.rmtree("qualitative_analysis/selected_cases/")

x = 0.25

bounds = {}
for topic in ["sleep", "quarto", "blackholes"]:
    bounds[topic] = {}
    for setting in ["base", "enhanced"]:
        bounds[topic][setting] = {}
        for value_name in ["pre_subj_comprehension", "post_subj_comprehension", "post_obj_comprehension", "post_enabledness", "post_constructiveness"]:
            all_values = []
            for user_id in stats:
                if topic == setup_per_user[user_id]["topic"] and setting == setup_per_user[user_id]["setting"]:
                    all_values.append(stats[user_id][value_name])
            # Evaluate best/worse x% users 
            sorted_values = sorted(all_values)
            n = len(sorted_values)
            # Get lower bound value of x% best users
            top_x_percent_values = sorted_values[-int(n * x):]
            lower_bound = min(top_x_percent_values)
            # Get upper bound of x% worse users
            top_x_percent_values = sorted_values[:int(n * x)]
            upper_bound = max(top_x_percent_values)
            bounds[topic][setting][value_name] = {"upper_bound": upper_bound, "lower_bound": lower_bound, "mean": np.mean(all_values)}
            
        # Subjective comprehension gain
        pre_comprehension = []
        post_comprehension = []
        for user_id in stats:
            if topic == setup_per_user[user_id]["topic"] and setting == setup_per_user[user_id]["setting"]:
                pre_comprehension.append(stats[user_id]["pre_subj_comprehension"])
                post_comprehension.append(stats[user_id]["post_subj_comprehension"])
        comprehension_gain = [a / b for a, b in zip(post_comprehension, pre_comprehension)]
        # Evaluate best/worse x% users 
        sorted_values = sorted(comprehension_gain)
        n = len(sorted_values)
        # Get lower bound value of x% best users
        top_x_percent_values = sorted_values[-int(n * x):]
        lower_bound = min(top_x_percent_values)
        # Get upper bound of x% worse users
        top_x_percent_values = sorted_values[:int(n * x)]
        upper_bound = max(top_x_percent_values)
        bounds[topic][setting]["subj_comprehension_gain"] = {"upper_bound": upper_bound, "lower_bound": lower_bound, "mean": np.mean(comprehension_gain)}

# Filter interesting edge cases
for user_id in stats:
    topic = setup_per_user[user_id]["topic"]
    setting = setup_per_user[user_id]["setting"]

    path = f"chats/{topic}/{setting}/"
    file_path = f"{path}/{user_id}.txt"

    # High subjective comprehension gain, high objective understanding and high co-construct behavior
    if stats[user_id]["post_subj_comprehension"]/stats[user_id]["pre_subj_comprehension"] >= bounds[topic][setting]["subj_comprehension_gain"]["lower_bound"]:
        if stats[user_id]["post_obj_comprehension"] >= bounds[topic][setting]["post_obj_comprehension"]["lower_bound"] and stats[user_id]["post_enabledness"] >= bounds[topic][setting]["post_enabledness"]["lower_bound"]:
            if stats[user_id]["post_constructiveness"] >= bounds[topic][setting]["post_constructiveness"]["lower_bound"]:
                quali_path = f"qualitative_analysis/selected_cases/{topic}/{setting}/high_subj_gain__high_obj__high_coconstruct"
                os.makedirs(quali_path, exist_ok=True)
                shutil.copy(file_path, quali_path)
    # High subjective comprehension gain, low objective understanding and high co-construct behavior
    if stats[user_id]["post_subj_comprehension"]/stats[user_id]["pre_subj_comprehension"] >= bounds[topic][setting]["subj_comprehension_gain"]["lower_bound"]:
        if stats[user_id]["post_obj_comprehension"] <= bounds[topic][setting]["post_obj_comprehension"]["upper_bound"] and stats[user_id]["post_enabledness"] <= bounds[topic][setting]["post_enabledness"]["upper_bound"]:
            if stats[user_id]["post_constructiveness"] >= bounds[topic][setting]["post_constructiveness"]["lower_bound"]:
                quali_path = f"qualitative_analysis/selected_cases/{topic}/{setting}/high_subj_gain__low_obj__high_coconstruct"
                os.makedirs(quali_path, exist_ok=True)
                shutil.copy(file_path, quali_path)
    # High subjective comprehension gain, high objective understanding and low co-construct behavior
    if stats[user_id]["post_subj_comprehension"]/stats[user_id]["pre_subj_comprehension"] >= bounds[topic][setting]["subj_comprehension_gain"]["lower_bound"]:
        if stats[user_id]["post_obj_comprehension"] >= bounds[topic][setting]["post_obj_comprehension"]["lower_bound"] and stats[user_id]["post_enabledness"] >= bounds[topic][setting]["post_enabledness"]["lower_bound"]:
            if stats[user_id]["post_constructiveness"] <= bounds[topic][setting]["post_constructiveness"]["upper_bound"]:
                quali_path = f"qualitative_analysis/selected_cases/{topic}/{setting}/high_subj_gain__high_obj__low_coconstruct"
                os.makedirs(quali_path, exist_ok=True)
                shutil.copy(file_path, quali_path)
    # High subjective comprehension gain, low objective understanding and low co-construct behavior
    if stats[user_id]["post_subj_comprehension"]/stats[user_id]["pre_subj_comprehension"] >= bounds[topic][setting]["subj_comprehension_gain"]["lower_bound"]:
        if stats[user_id]["post_obj_comprehension"] <= bounds[topic][setting]["post_obj_comprehension"]["upper_bound"] and stats[user_id]["post_enabledness"] <= bounds[topic][setting]["post_enabledness"]["upper_bound"]:
            if stats[user_id]["post_constructiveness"] <= bounds[topic][setting]["post_constructiveness"]["upper_bound"]:
                quali_path = f"qualitative_analysis/selected_cases/{topic}/{setting}/high_subj_gain__low_obj__low_coconstruct"
                os.makedirs(quali_path, exist_ok=True)
                shutil.copy(file_path, quali_path)
    # Low subjective comprehension gain, low objective understanding and high co-construct behavior
    if stats[user_id]["pre_subj_comprehension"] <= bounds[topic][setting]["pre_subj_comprehension"]["upper_bound"]:
        if stats[user_id]["post_subj_comprehension"]/stats[user_id]["pre_subj_comprehension"] <= bounds[topic][setting]["subj_comprehension_gain"]["upper_bound"]:
            if stats[user_id]["post_obj_comprehension"] <= bounds[topic][setting]["post_obj_comprehension"]["upper_bound"] and stats[user_id]["post_enabledness"] <= bounds[topic][setting]["post_enabledness"]["upper_bound"]:
                if stats[user_id]["post_constructiveness"] >= bounds[topic][setting]["post_constructiveness"]["lower_bound"]:
                    quali_path = f"qualitative_analysis/selected_cases/{topic}/{setting}/low_subj__low_subj_gain__low_obj__high_coconstruct"
                    os.makedirs(quali_path, exist_ok=True)
                    shutil.copy(file_path, quali_path)
    # Low subjective comprehension gain, high objective understanding and high co-construct behavior
    if stats[user_id]["pre_subj_comprehension"] <= bounds[topic][setting]["pre_subj_comprehension"]["upper_bound"]:
        if stats[user_id]["post_subj_comprehension"]/stats[user_id]["pre_subj_comprehension"] <= bounds[topic][setting]["subj_comprehension_gain"]["upper_bound"]:
            if stats[user_id]["post_obj_comprehension"] >= bounds[topic][setting]["post_obj_comprehension"]["lower_bound"] and stats[user_id]["post_enabledness"] >= bounds[topic][setting]["post_enabledness"]["lower_bound"]:
                if stats[user_id]["post_constructiveness"] >= bounds[topic][setting]["post_constructiveness"]["lower_bound"]:
                        quali_path = f"qualitative_analysis/selected_cases/{topic}/{setting}/low_subj__low_subj_gain__high_obj__high_coconstruct"
                        os.makedirs(quali_path, exist_ok=True)
                        shutil.copy(file_path, quali_path)
    # Low subjective comprehension gain, low objective understanding and low co-construct behavior
    if stats[user_id]["pre_subj_comprehension"] <= bounds[topic][setting]["pre_subj_comprehension"]["upper_bound"]:
        if stats[user_id]["post_subj_comprehension"]/stats[user_id]["pre_subj_comprehension"] <= bounds[topic][setting]["subj_comprehension_gain"]["upper_bound"]:
            if stats[user_id]["post_obj_comprehension"] <= bounds[topic][setting]["post_obj_comprehension"]["upper_bound"] and stats[user_id]["post_enabledness"] <= bounds[topic][setting]["post_enabledness"]["upper_bound"]:
                if stats[user_id]["post_constructiveness"] <= bounds[topic][setting]["post_constructiveness"]["upper_bound"]:
                    quali_path = f"qualitative_analysis/selected_cases/{topic}/{setting}/low_subj__low_subj_gain__low_obj__low_coconstruct"
                    os.makedirs(quali_path, exist_ok=True)
                    shutil.copy(file_path, quali_path)
    # Low subjective comprehension gain, high objective understanding and low co-construct behavior
    if stats[user_id]["pre_subj_comprehension"] <= bounds[topic][setting]["pre_subj_comprehension"]["upper_bound"]:
        if stats[user_id]["post_subj_comprehension"]/stats[user_id]["pre_subj_comprehension"] <= bounds[topic][setting]["subj_comprehension_gain"]["upper_bound"]:
            if stats[user_id]["post_obj_comprehension"] >= bounds[topic][setting]["post_obj_comprehension"]["lower_bound"] and stats[user_id]["post_enabledness"] >= bounds[topic][setting]["post_enabledness"]["lower_bound"]:
                if stats[user_id]["post_constructiveness"] <= bounds[topic][setting]["post_constructiveness"]["upper_bound"]:
                    quali_path = f"qualitative_analysis/selected_cases/{topic}/{setting}/low_subj__low_subj_gain__high_obj__low_coconstruct"
                    os.makedirs(quali_path, exist_ok=True)
                    shutil.copy(file_path, quali_path)

# Save used upper and lower bounds
    with open("qualitative_analysis/upper_lower_bounds.json", "w") as file:
        json.dump(bounds, file, indent=4)