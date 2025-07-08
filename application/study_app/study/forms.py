from django import forms

from .models import ChatPage, UserResponse


class UserResponseForm(forms.Form):
    def __init__(self, *args, questions=None, **kwargs):
        super().__init__(*args, **kwargs)
        for question in questions:
            question_id = question["question_id"]
            prompt = question["prompt"]
            question_type = question["type"]

            if question_type == "text":
                self.fields[question_id] = forms.CharField(
                    label=prompt,
                    widget=forms.Textarea(
                        attrs={
                            "class": "border p-3 rounded w-full resize-y h-40",
                            "style": "overflow-y:auto;",  # Ensure scrolling for large text
                        }
                    ),
                    required=question.get("required", True),
                )
            elif question_type == "radio":
                options = question.get("options", [])
                self.fields[question_id] = forms.ChoiceField(
                    label=prompt,
                    choices=[(option, option) for option in options],
                    widget=forms.RadioSelect(attrs={"class": "p-2"}),
                    required=question.get("required", True),
                )


class ChatPageForm(forms.ModelForm):
    class Meta:
        model = ChatPage
        fields = ["feedback"]
        widgets = {
            "feedback": forms.Textarea(
                attrs={
                    "class": "w-full p-3 border border-gray-300 bg-gray-50 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500",
                    "rows": 4,
                    "placeholder": "Type your detailed answer here...",
                }
            )
        }
