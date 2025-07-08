from django.db import models


class UserSession(models.Model):
    user_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    study_topic = models.CharField(max_length=32)
    system_prompt = models.CharField(max_length=32)
    prolific_pid = models.CharField(max_length=32)
    failed_attention_check = models.BooleanField(default=False)

    # Fields for tracking progress
    current_page_index = models.IntegerField(default=0)
    last_progress_timestamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.user_id


class UserResponse(models.Model):
    user_session = models.ForeignKey(UserSession, on_delete=models.CASCADE)
    form_id = models.CharField(max_length=100)  # Identifier for the form
    question_id = models.CharField(max_length=100)  # Identifier for each question
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Response {self.question_id} from {self.user_session.user_id} for form {self.form_id}"


class ChatPage(models.Model):
    user_session = models.ForeignKey(UserSession, on_delete=models.CASCADE)
    feedback = models.TextField()  # Example field specific to Page 1.5

    def __str__(self):
        return f"Page 1.5 response by {self.user_session.user_id}"


class ChatMessage(models.Model):
    user_session = models.ForeignKey(UserSession, on_delete=models.CASCADE)
    message = models.TextField()
    is_user_message = (
        models.BooleanField()
    )  # True if the message is from the user, False if from LLM
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        sender = "User" if self.is_user_message else "LLM"
        return f"{sender} message at {self.timestamp}"
