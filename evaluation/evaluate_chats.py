import sys
import os
import numpy as np
import json
import re
from datetime import datetime
from scipy.stats import skewtest, mannwhitneyu, fisher_exact
from datetime import timedelta
import copy
import spacy
import shutil
import textstat
from textcomplexity import surface
from textcomplexity.utils.text import Text
from collections import namedtuple
import nltk
Token = namedtuple("Token", ["word", "pos"])
nlp = spacy.load("en_core_web_sm")


sys.path.append('../')
from study.pages import SLEEPCYCLE_OBJECTIVE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE, BLACKHOLES_OBJECTIVE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE, QUARTO_OBJECTIVE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE
from study.pages import SLEEPCYCLE_OBJECTIVE_CHOICE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE, BLACKHOLES_OBJECTIVE_CHOICE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE, QUARTO_OBJECTIVE_CHOICE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE
from study.pages import COCONSTRUCT_POST_QUESTIONNAIRE_PAGE

import sqlite3
import csv
from operator import itemgetter

with open("user_study_data/chat_per_user.json", "r") as file:
    chat_per_user = json.load(file)

with open("user_study_data/questionnaires_results_per_user.json", "r") as file:
    questionnaires_results_per_user = json.load(file)

with open("user_study_data/setup_per_user.json", "r") as file:
    setup_per_user = json.load(file)

with open("user_study_data/understanding_questionnaires.json", "r") as file:
    understanding_questionnaires = json.load(file)

likert_scale = {"strongly disagree": 1,
                "disagree": 2,
                "neutral": 3,
                "agree": 4,
                "strongly agree": 5,}

def get_time_from_string(date_string):
    # Parse the string to a datetime object
    dt_obj = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S.%f")
    # Format the datetime object to exclude microseconds and milliseconds
    dt_obj = dt_obj.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S.%f")
    
    return datetime.strptime(dt_obj, "%Y-%m-%d %H:%M:%S.%f")

db_name = '../db_finalstudy.sqlite3'

results = {}

for user_id in questionnaires_results_per_user:
    results[user_id] = {}
    for questionnaire_name in questionnaires_results_per_user[user_id]:
        current_result = questionnaires_results_per_user[user_id][questionnaire_name]
        # Pre/post subjective comprehension of the user (5=expert, 1=non-expert) and the pre/post motivation of the user (5=high, 1=low)
        if "subj_comprehension" in questionnaire_name or "motivation" in questionnaire_name:
            all_scores = []
            for question_id, score_text in current_result.items():
                if question_id == "q11":
                    # Reverse score since the question is asked the other way around
                    all_scores.append(score)
                elif "attention" not in question_id:
                    score = likert_scale[score_text]
                    all_scores.append(score)
            results[user_id][questionnaire_name] = np.mean(all_scores)
        # Post objective understanding of the user
        elif questionnaire_name == "post_obj_comprehension" or questionnaire_name == "post_enabledness":
            topic = setup_per_user[user_id]["topic"]
            correctly_answered = 0
            for question_id in understanding_questionnaires[topic][questionnaire_name]:
                correct_answer = understanding_questionnaires[topic][questionnaire_name][question_id]["correct_answer"]
                if current_result[question_id] == correct_answer:
                    correctly_answered += 1
            results[user_id][questionnaire_name] = correctly_answered
            results[user_id][questionnaire_name+"_open-question-q1"] = current_result["open-question-q1"]
            results[user_id][questionnaire_name+"_open-question-q2"] = current_result["open-question-q2"]

        # Rated co-constructive behavior of the LLM
        elif questionnaire_name == "post_constructiveness":
            mean = np.mean([likert_scale[score_text] for question_id, score_text in current_result.items() if question_id not in ("q8", "q13")])
            results[user_id][questionnaire_name] = mean
            results[user_id][questionnaire_name+"_q13"] = current_result["q13"]

for user_id in chat_per_user:
    topic = setup_per_user[user_id]["topic"]
    setting = setup_per_user[user_id]["setting"]

    num_EE_queries = 0
    EE_processing_times = []
    num_EX_sentences = []
    num_EX_words = []
    EX_write_times = []

    old_timestamp = None
    last_message = -1
    for turn in chat_per_user[user_id]:
        timestamp = get_time_from_string(turn["timestamp"])
        if turn["turn_text"]["author"] == "Explainee":
            if old_timestamp:
                EE_processing_times.append(timestamp - old_timestamp)
            old_timestamp = timestamp
            if last_message != turn["turn_text"]["author"]:
                num_EE_queries += 1
                last_message = turn["turn_text"]["author"]
        else:
            if old_timestamp:
                EX_write_times.append(timestamp - old_timestamp)
            old_timestamp = timestamp            

            message = turn["turn_text"]["text"]
            message_without_numbers = re.sub(r'\b\d+[.]?\d*\b', '', message)
            message_without_numbers = re.sub(r'\*.*\*', '', message_without_numbers)

            EX_sentences = [sent.text for sent in nlp(message_without_numbers).sents]
            num_EX_sentences.append(len(EX_sentences))
            for sentence in EX_sentences:
                EX_words = [token.text for token in nlp(sentence) if token.is_alpha]
                num_EX_words.append(len(EX_words))

            last_message = turn["turn_text"]["author"]

    time_span = get_time_from_string(chat_per_user[user_id][-1]["timestamp"]) - get_time_from_string(chat_per_user[user_id][0]["timestamp"])   
    mean_EE_processing_times = np.mean(EE_processing_times)
    mean_num_EX_sentences = np.mean(num_EX_sentences)
    mean_num_EX_words = np.mean(num_EX_words)
    mean_EX_write_times = np.mean(EX_write_times)

    results[user_id]["chat_duration_time"] = time_span
    results[user_id]["num_EE_queries"] = num_EE_queries
    results[user_id]["EE_processing_times"] = mean_EE_processing_times
    results[user_id]["num_EX_sentences"] = mean_num_EX_sentences
    results[user_id]["num_EX_words"] = mean_num_EX_words
    results[user_id]["EX_write_times"] = mean_EX_write_times

    # Create txt file for each user chat
    path = f"chats/{topic}/{setting}/"
    os.makedirs(path, exist_ok=True)
    file_path = f"{path}/{user_id}.txt"
    os.remove(file_path) if os.path.exists(file_path) else None
    with open(file_path, 'a') as f:
        f.write("User results pre chat:\n")
        f.write(f"Motivation (high=5): {results[user_id]["pre_motivation"]:.1f}\n")
        f.write(f"Subjective comprehension (expert=5): {results[user_id]["pre_subj_comprehension"]:.1f}\n\n")
        f.write("User results post chat:\n")
        f.write(f"Motivation (high=5): {results[user_id]["post_motivation"]:.1f}\n")
        f.write(f"Subjective comprehension (expert=5): {results[user_id]["post_subj_comprehension"]:.1f}\n")
        f.write(f"Correctly answered yes/no questions (objective comprehension): {(results[user_id]["post_obj_comprehension"]/len(understanding_questionnaires[topic]["post_obj_comprehension"]))*100:.1f}%\n")
        f.write(f"Sufficient chat to answer yes/no questions: {results[user_id]["post_obj_comprehension_open-question-q1"]}\n")
        f.write(f"Correctly answered choice questions (enabledness): {(results[user_id]["post_enabledness"]/len(understanding_questionnaires[topic]["post_enabledness"]))*100:.1f}%\n")
        f.write(f"Sufficient chat to answer choice questions: {results[user_id]["post_enabledness_open-question-q1"]}\n\n\n")
        f.write("Chat stats:\n")
        f.write(f"Duration: {time_span}\n")
        f.write(f"Number of EE queries (without duplicates): {num_EE_queries}\n")
        f.write(f"Avg processing time of EE (h:m:s): {str(mean_EE_processing_times).split(".")[0]}\n")
        f.write(f"Avg number of sentences per EX answer: {mean_num_EX_sentences:.1f}\n")
        f.write(f"Avg number of words per EX answer: {mean_num_EX_words:.1f}\n")
        f.write(f"Avg writing time of EX (h:m:s): {str(mean_EX_write_times).split(".")[0]}\n")
        f.write(f"Co-constructive behavior of EX (perfect=5): {results[user_id]["pre_motivation"]:.1f}\n\n\n")
        f.write("--------------------------------[START CHAT]--------------------------------\n\n")
        
        start_timestamp = get_time_from_string(chat_per_user[user_id][0]["timestamp"])
        for turn in chat_per_user[user_id]:
            timestamp = get_time_from_string(turn["timestamp"])
            message = turn["turn_text"]["text"]
            current_time = timestamp - start_timestamp
            if turn["turn_text"]["author"] == "Explainee":
                f.write(f"{current_time} EE:\n{message}\n\n\n")
            else:
                f.write(f"{current_time} EX:\n{message}\n\n")

        f.write("--------------------------------[END CHAT]--------------------------------\n\n")
        f.write(f"\nEE answers to open questions:\n\n")
        f.write(f"Question 1: Were the explanations in the chat sufficient to answer the questions in the objective comprehension questionnaire?\n{results[user_id]["post_obj_comprehension_open-question-q2"]}\n\n")
        f.write(f"Question 2: Were the explanations in the chat sufficient to answer the questions in the enabledness questionnaire?\n{results[user_id]["post_enabledness_open-question-q2"]}\n\n")
        f.write(f"Question 3: Please specify additional behavior of your dialogue partner during the chat that helped you better understand the explanations.\n{results[user_id]["post_constructiveness_q13"]}")


# Create statistics
stats = {}
stats_setting = {}
for user_id in results:
    topic = setup_per_user[user_id]["topic"]
    setting = setup_per_user[user_id]["setting"]

    if topic not in stats:
        stats[topic] = {}
    if setting not in stats[topic]:
        stats[topic][setting] = {}
    if setting not in stats_setting:
        stats_setting[setting] = {}

    for value_name in results[user_id]:
        if "q2" in value_name or "q13" in value_name:
            continue
        if "q1" in value_name:
            results[user_id][value_name] = True if results[user_id][value_name] == "yes" else False
        if value_name not in stats[topic][setting]:
            stats[topic][setting][value_name] = []
        if value_name not in stats_setting[setting]:
            stats_setting[setting][value_name] = []

        stats[topic][setting][value_name].append(results[user_id][value_name])
        stats_setting[setting][value_name].append(results[user_id][value_name])

# Perform significance test
significance_test = {}
for topic in stats:
    significance_test[topic] = {}
    for value_name in stats[topic]["base"]:
        if "q1" in value_name:
            data_base = stats[topic]["base"][value_name]
            data_enhanced = stats[topic]["enhanced"][value_name]
            table = [[sum(data_base), len(data_base)-sum(data_base)],
                    [sum(data_enhanced), len(data_enhanced)-sum(data_base)]]
            odds_ratio, p_value = fisher_exact(table)
        else:
            data_base = stats[topic]["base"][value_name]
            if isinstance(data_base[0], timedelta):
                data_base = [td.seconds for td in data_base]
            data_enhanced = stats[topic]["enhanced"][value_name]
            if isinstance(data_enhanced[0], timedelta):
                data_enhanced = [td.seconds for td in data_enhanced]
            statistic, p_value = mannwhitneyu(data_base, data_enhanced)
        significance_test[topic][value_name] = p_value

with open("results/p_values_topics_settings.json", "w") as file:
    json.dump(significance_test, file, indent=4)

# Perform significance test
significance_test = {}
for topic in stats:
    significance_test[topic] = {}
    for setting in stats[topic]:
        significance_test[topic][setting] = {}
        for value_name in stats[topic][setting]:
            if "pre_" in value_name:
                data_pre = stats[topic][setting][value_name]
                data_post = stats[topic][setting][value_name.replace("pre_", "post_")]
                statistic, p_value = mannwhitneyu(data_pre, data_post)
                significance_test[topic][setting][value_name.replace("pre_", "")] = p_value

with open("results/p_values_topics_settings_pre_post.json", "w") as file:
    json.dump(significance_test, file, indent=4)

# Perform significance test
significance_test = {}
for value_name in stats_setting["base"]:
    if "sufficient_chat" in value_name:
        data_base = stats_setting["base"][value_name]
        data_enhanced = stats_setting["enhanced"][value_name]
        table = [[sum(data_base), len(data_base)-sum(data_base)],
                 [sum(data_enhanced), len(data_enhanced)-sum(data_base)]]
        #print(table)
        odds_ratio, p_value = fisher_exact(table)
    else:
        data_base = stats_setting["base"][value_name]
        if isinstance(data_base[0], timedelta):
            data_base = [td.seconds for td in data_base]
        data_enhanced = stats_setting["enhanced"][value_name]
        if isinstance(data_enhanced[0], timedelta):
            data_enhanced = [td.seconds for td in data_enhanced]
        statistic, p_value = mannwhitneyu(data_base, data_enhanced)
    significance_test[value_name] = p_value

with open("results/p_values_settings.json", "w") as file:
    json.dump(significance_test, file, indent=4)

# Perform significance test
significance_test = {}
for setting in stats_setting:
    significance_test[setting] = {}
    for value_name in stats_setting[setting]:
        if "pre_" in value_name:
            data_pre = stats_setting[setting][value_name]
            data_post = stats_setting[setting][value_name.replace("pre_", "post_")]
            statistic, p_value = mannwhitneyu(data_pre, data_post)
            significance_test[setting][value_name.replace("pre_", "")] = p_value

with open("results/p_values_settings_pre_post.json", "w") as file:
    json.dump(significance_test, file, indent=4)

# Calculate mean+std over all users
for topic in stats:
    for setting in stats[topic]:
        for value_name in stats[topic][setting]:
            if value_name == "post_obj_comprehension":
                data = stats[topic][setting]["post_obj_comprehension"]
                num_objective_understanding_questions = len(understanding_questionnaires[topic]["post_obj_comprehension"])
                percentages = [(answer / num_objective_understanding_questions) * 100 for answer in data]
                values_mean = np.mean(percentages)
                values_std = np.std(percentages)
                stats[topic][setting]["post_obj_comprehension"] = f"{values_mean:.1f}% +-{values_std:.1f}"
            elif value_name == "post_enabledness":
                data = stats[topic][setting]["post_enabledness"]
                num_objective_understanding_choice_questions = len(understanding_questionnaires[topic]["post_enabledness"])
                percentages = [(answer / num_objective_understanding_choice_questions) * 100 for answer in data]
                values_mean = np.mean(percentages)
                values_std = np.std(percentages)
                stats[topic][setting]["post_enabledness"] = f"{values_mean:.1f}% +-{values_std:.1f}"
            elif "q1" in value_name:
                data = stats[topic][setting][value_name]
                percentage = 100 * sum(data) / len(data)
                stats[topic][setting][value_name] = f"{percentage:.1f}% found the chat sufficient to answer the questions/statements"
            elif "time" in value_name:
                data = stats[topic][setting][value_name]
                values_mean = str(np.mean(data)).split(".")[0]
                seconds_list = [td.seconds for td in data]
                values_std = str(np.std(seconds_list)).split(".")[0]
                stats[topic][setting][value_name] = f"{values_mean} +-{values_std}s"
            else:
                data = stats[topic][setting][value_name]
                values_mean = np.mean(data)
                values_std = np.std(data)
                stats[topic][setting][value_name] = f"{values_mean:.1f} +-{values_std:.1f}"

for setting in stats_setting:
    for value_name in stats_setting[setting]:
        if value_name == "post_obj_comprehension":
            data = stats_setting[setting]["post_obj_comprehension"]
            num_objective_understanding_choice_questions = len(understanding_questionnaires[topic]["post_obj_comprehension"])
            percentages = [(answer / num_objective_understanding_choice_questions) * 100 for answer in data]
            values_mean = np.mean(percentages)
            values_std = np.std(percentages)
            stats_setting[setting]["post_obj_comprehension"] = f"{values_mean:.1f}% +-{values_std:.1f}"
        elif value_name == "post_enabledness":
            data = stats_setting[setting]["post_enabledness"]
            num_objective_understanding_choice_questions = len(understanding_questionnaires[topic]["post_enabledness"])
            percentages = [(answer / num_objective_understanding_choice_questions) * 100 for answer in data]
            values_mean = np.mean(percentages)
            values_std = np.std(percentages)
            stats_setting[setting]["post_enabledness"] = f"{values_mean:.1f}% +-{values_std:.1f}"
        elif "q1" in value_name:
            data = stats_setting[setting][value_name]
            percentage = 100 * sum(data) / len(data)
            stats_setting[setting][value_name] = f"{percentage:.1f}% found the chat sufficient to answer the questions/statements"
        elif "time" in value_name:
            data = stats_setting[setting][value_name]
            values_mean = str(np.mean(data)).split(".")[0]
            seconds_list = [td.seconds for td in data]
            values_std = str(np.std(seconds_list)).split(".")[0]
            stats_setting[setting][value_name] = f"{values_mean} +-{values_std}s"
        else:
            data = stats_setting[setting][value_name]
            values_mean = np.mean(data)
            values_std = np.std(data)
            stats_setting[setting][value_name] = f"{values_mean:.1f} +-{values_std:.1f}"
                 

# Save results per user
os.makedirs("results", exist_ok=True)
for user_id in results:
    for value_name in results[user_id]:
        if "time" in value_name:
            seconds = results[user_id][value_name].seconds
            results[user_id][value_name] = seconds

with open("results/results_per_user.json", "w") as file:
    json.dump(results, file, indent=4)

# Save the statistics
with open("results/stats_topics_settings.json", "w") as file:
    json.dump(stats, file, indent=4)

with open("results/stats_settings.json", "w") as file:
    json.dump(stats_setting, file, indent=4)