from datetime import timedelta

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI

from study_app.settings import (
    CHAT_TIME_SPAN_MINUTES,
    LLM_BACKEND_ADDRESS,
    LLM_BACKEND_API_KEY,
    LLM_MODEL_NAME,
    LLM_SEED,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    PROLIFIC_ATTENTION_CHECK_FAILED_URL,
    PROLIFIC_COMPLETION_URL)

from .forms import UserResponseForm
from .models import ChatMessage, UserResponse, UserSession
from .pages import PAGE_SEQUENCES
from .utils import attention_checks_ok


def user_identifier(request):
    if request.method == "POST":
        user_id = request.POST.get("user_id")
        prolific_pid = request.POST.get("prolific_pid")

        # Check if prolific id already exists
        prolific_session = UserSession.objects.filter(prolific_pid=prolific_pid).first()
        if prolific_session:
            user_session = UserSession.objects.filter(
                user_id=user_id, prolific_pid=prolific_pid
            ).first()
        else:
            user_session = UserSession.objects.filter(user_id=user_id).first()

        # Check if the user session exists
        if user_session and not prolific_session:
            # Redirect to the form page if session exists
            user_session.prolific_pid = prolific_pid
            user_session.save()
            return redirect(
                "study:form_page", user_id=user_session.user_id, page_index=0
            )
        elif user_session and prolific_session:
            return redirect(
                "study:form_page", user_id=user_session.user_id, page_index=0
            )
        else:
            # Reload the page if session doesn't exist
            messages.error(
                request,
                "This user session id does not exist",
            )
            return render(
                request,
                "study/user_identifier.html",
            )

    # If the view reaches this part, it is not a POST request, but most likely a GET request
    prolific_pid_query_parameter = request.GET.get("PROLIFIC_PID", "")
    return render(
        request,
        "study/user_identifier.html",
        {
            "prolific_pid": prolific_pid_query_parameter,
        },
    )


def form_page_view(request, user_id, page_index=None):
    user_session = get_object_or_404(UserSession, user_id=user_id)
    PAGE_SEQUENCE = PAGE_SEQUENCES[user_session.study_topic]

    # Determine the page index based on progress if page_index is None
    if page_index is None:
        page_index = 0

    # Redirect to the completion page if page_index is beyond the page sequence length
    if page_index >= len(PAGE_SEQUENCE):
        return redirect("study:completion", user_id=user_id)

    # Redirect to attention check failed page, if attention check was failed before
    if user_session.failed_attention_check:
        return redirect("study:attention_check_failed", user_id=user_id)

    # Retrieve the current page data from the sequence
    page_data = PAGE_SEQUENCE[page_index]

    # Check if revisiting this page is restricted
    allow_revisit_current = page_data.get("allow_revisit")
    if page_index > 0:
        allow_revisit_previous = PAGE_SEQUENCE[page_index - 1].get("allow_revisit")
    else:
        allow_revisit_previous = False

    # Redirect if user tries to access a completed page where revisiting is not allowed
    if not allow_revisit_current and page_index < user_session.current_page_index:
        error_message = "You have already completed this page. Redirecting to your last uncompleted page."
        # Only add the error message if it's not already present in the message queue
        if not any(
            msg.message == error_message for msg in messages.get_messages(request)
        ):
            messages.error(
                request,
                error_message,
            )
        return redirect(
            "study:form_page",
            user_id=user_id,
            page_index=user_session.current_page_index,
        )

    # Redirect to completion if the page index exceeds the sequence length
    if page_index >= len(PAGE_SEQUENCE):
        return redirect("study:completion", user_id=user_id)

    # Handle transition pages
    if page_data["type"] == "transition":
        user_session.current_page_index = page_index + 1
        user_session.last_progress_timestamp = timezone.now()
        user_session.save()

        return render(
            request,
            "study/transition_page.html",
            {
                "user_id": user_id,
                "page_index": page_index,
                "next_page": (
                    page_index + 1 if page_index + 1 < len(PAGE_SEQUENCE) else None
                ),
                "previous_page": page_index - 1 if page_index > 0 else None,
                "allow_revisit_previous": allow_revisit_previous,
                "text": page_data["text"],
                "subtext": page_data.get("subtext", ""),
            },
        )

    # Handle chat pages
    if page_data["type"] == "chat" and page_data["form_id"] == "chat":
        user_session.current_page_index = page_index
        user_session.last_progress_timestamp = timezone.now()
        user_session.save()

        return render(
            request,
            "study/chat_page.html",
            {
                "user_id": user_id,
                "page_index": page_index,
                "next_page": (
                    page_index + 1 if page_index + 1 < len(PAGE_SEQUENCE) else None
                ),
                "previous_page": page_index - 1 if page_index > 0 else None,
                "allow_revisit_previous": allow_revisit_previous,
            },
        )

    # Handle form pages
    form_id = page_data["form_id"]
    questions = page_data["questions"]

    if request.method == "POST":
        form = UserResponseForm(request.POST, questions=questions)
        if form.is_valid():
            for question in questions:
                question_id = question["question_id"]
                answer_text = form.cleaned_data[question_id]
                UserResponse.objects.update_or_create(
                    user_session=user_session,
                    form_id=form_id,
                    question_id=question_id,
                    defaults={"answer": answer_text},
                )

            user_session.current_page_index = page_index + 1
            user_session.last_progress_timestamp = timezone.now()
            user_session.save()

            if attention_checks_ok(filled_form=form, form_questions=questions):
                return redirect(
                    "study:form_page", user_id=user_id, page_index=page_index + 1
                )
            else:
                user_session.failed_attention_check = True
                user_session.save()

                return redirect(PROLIFIC_ATTENTION_CHECK_FAILED_URL)
    else:
        initial_data = {}
        for question in questions:
            question_id = question["question_id"]
            try:
                response = UserResponse.objects.get(
                    user_session=user_session, form_id=form_id, question_id=question_id
                )
                initial_data[question_id] = response.answer
            except UserResponse.DoesNotExist:
                initial_data[question_id] = ""
        form = UserResponseForm(initial=initial_data, questions=questions)

        # Prepare a context variable that includes image URLs alongside form fields
        questions_with_images = []
        for question in questions:
            question_data = {
                "field": form[question["question_id"]],
                "label": question["prompt"],
                "image_url": question.get(
                    "image_url", None
                ),  # Add image URL if it exists
            }
            questions_with_images.append(question_data)

        return render(
            request,
            "study/form_page.html",
            {
                "form": form,
                "questions_with_images": questions_with_images,  # Pass this to the template
                "user_id": user_id,
                "page_index": page_index,
                "total_pages": len(PAGE_SEQUENCE),
                "next_page": (
                    page_index + 1 if page_index + 1 < len(PAGE_SEQUENCE) else None
                ),
                "previous_page": page_index - 1 if page_index > 0 else None,
                "allow_revisit_previous": allow_revisit_previous,
                "form_title": page_data["form_title"],
                "subtext": page_data.get("subtext", ""),
            },
        )


@csrf_exempt
def chat_view(request, user_id):
    user_session = get_object_or_404(UserSession, user_id=user_id)

    # Check if a message has been sent already to start the timer if not
    first_message = (
        ChatMessage.objects.filter(user_session=user_session, is_user_message=True)
        .order_by("timestamp")
        .first()
    )

    # We add a minute to account for LLM response times
    if first_message:
        chat_timelimit_reached = timezone.now() > (
            first_message.timestamp + timedelta(minutes=CHAT_TIME_SPAN_MINUTES + 1)
        )
    else:
        chat_timelimit_reached = False

    # Load full past conversation
    past_conversation = [
        {"message": msg.message, "is_user_message": msg.is_user_message}
        for msg in ChatMessage.objects.filter(user_session=user_session).order_by(
            "timestamp"
        )
    ]

    if chat_timelimit_reached:
        llm_message = (
            "You have reached your chat time limit. Please continue to the next page."
        )
        conversation = [
            *past_conversation,
            {"message": llm_message, "is_user_message": False},
        ]
        return JsonResponse(
            {
                "conversation": conversation,
                "first_message_timestamp": (
                    first_message.timestamp if first_message else None
                ),
                "chat_deadline_timestamp": (
                    (
                        first_message.timestamp
                        + timedelta(minutes=CHAT_TIME_SPAN_MINUTES)
                    )
                    if first_message
                    else None
                ),
            }
        )
    elif not chat_timelimit_reached and request.method == "POST":
        user_message = request.POST.get("message")

        if user_message:
            new_conversation_with_prompt = [
                {
                    "role": "system",
                    "content": LLM_SYSTEM_PROMPT[user_session.system_prompt],
                },
                *[
                    {
                        "role": "user" if msg["is_user_message"] else "assistant",
                        "content": msg["message"],
                    }
                    for msg in past_conversation
                ],
                {"role": "user", "content": user_message},
            ]

            # Save the user's message
            ChatMessage.objects.create(
                user_session=user_session,
                message=user_message,
                is_user_message=True,
            )

            # Generate LLM response using OpenAI API
            client = OpenAI(base_url=LLM_BACKEND_ADDRESS, api_key=LLM_BACKEND_API_KEY)

            response = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=new_conversation_with_prompt,
                temperature=LLM_TEMPERATURE,
                seed=LLM_SEED,
            )
            llm_message = response.choices[0].message.content

            # Save the LLM's response
            ChatMessage.objects.create(
                user_session=user_session,
                message=llm_message,
                is_user_message=False,
            )

            # Retrieve all messages to update the chat
            final_updated_conversation_from_db = [
                {"message": msg.message, "is_user_message": msg.is_user_message}
                for msg in ChatMessage.objects.filter(
                    user_session=user_session
                ).order_by("timestamp")
            ]

            return JsonResponse(
                {
                    "conversation": final_updated_conversation_from_db,
                    "first_message_timestamp": (
                        first_message.timestamp if first_message else None
                    ),
                    "chat_deadline_timestamp": (
                        (
                            first_message.timestamp
                            + timedelta(minutes=CHAT_TIME_SPAN_MINUTES)
                        )
                        if first_message
                        else None
                    ),
                }
            )
    elif not chat_timelimit_reached and request.method == "GET":
        return JsonResponse(
            {
                "conversation": past_conversation,
                "first_message_timestamp": (
                    first_message.timestamp if first_message else None
                ),
                "chat_deadline_timestamp": (
                    (
                        first_message.timestamp
                        + timedelta(minutes=CHAT_TIME_SPAN_MINUTES)
                    )
                    if first_message
                    else None
                ),
            }
        )


def attention_check_failed_view(request, user_id):
    return render(
        request,
        "study/attention_check_failed.html",
        {
            "user_id": user_id,
            "prolific_attention_check_failed_url": PROLIFIC_ATTENTION_CHECK_FAILED_URL,
        },
    )


def completion_view(request, user_id):
    return render(
        request,
        "study/completion.html",
        {"user_id": user_id, "prolific_completion_url": PROLIFIC_COMPLETION_URL},
    )
