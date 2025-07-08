import csv
import os
from datetime import datetime

import django

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "study_app.settings")
django.setup()

from study.models import UserSession

BASE_URL = "https://vm188179-ai.hosting.uni-hannover.de"


def main():
    # Fetch all user sessions from the database
    sessions = UserSession.objects.all()

    current_date = datetime.now().strftime("%Y-%m-%d-%H%M")
    file_name = f"user-session-export_{current_date}.csv"

    # Open the file for writing
    with open(file_name, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(
            [
                "User ID",
                "Topic",
                "System Prompt",
                "Current Page Index",
                "Custom URL",
            ]
        )

        # Write each session's data
        for session in sessions:
            writer.writerow(
                [
                    session.user_id,
                    session.study_topic,
                    session.system_prompt,
                    session.current_page_index,
                    f"{BASE_URL}?uid={session.user_id}",
                ]
            )
    print(f"Exported {len(sessions)} user sessions to {file_name}")


if __name__ == "__main__":
    main()
    print("Done.")
