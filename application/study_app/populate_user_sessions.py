import os
from datetime import datetime
from uuid import uuid4

import django

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "study_app.settings")
django.setup()

from study.models import UserSession
from study.pages import PAGE_SEQUENCES

from study_app.settings import LLM_SYSTEM_PROMPT

# Pre-defined list of study topics
STUDY_TOPICS = list(PAGE_SEQUENCES.keys())
# Number of sessions to create per topic and prompt setting
NUM_SESSIONS_PER_TOPIC_PER_PROMPT_SETTING = 50


def create_random_user_sessions(num_sessions, topic):
    for _ in range(num_sessions):
        for prompt_setting in LLM_SYSTEM_PROMPT.keys():
            UserSession.objects.create(
                user_id=str(uuid4()),  # Generate random UUID
                study_topic=topic,
                system_prompt=prompt_setting,
                current_page_index=0,
                prolific_pid=0,
                last_progress_timestamp=datetime.now(),
                failed_attention_check=False,
            )


def main():
    num_system_prompt_settings = len(LLM_SYSTEM_PROMPT.keys())
    for topic in STUDY_TOPICS:
        print(
            f"==Creating {NUM_SESSIONS_PER_TOPIC_PER_PROMPT_SETTING * num_system_prompt_settings} sessions for topic '{topic}'"
        )
        create_random_user_sessions(NUM_SESSIONS_PER_TOPIC_PER_PROMPT_SETTING, topic)


if __name__ == "__main__":
    main()
    print("Done.")
