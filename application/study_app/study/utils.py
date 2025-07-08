from study_app.settings import ATTENTION_CHECK_FAILS_ALLOWED


def attention_checks_ok(filled_form, form_questions) -> bool:
    failed_checks = 0

    for question in form_questions:
        question_id = question["question_id"]
        answer_text = filled_form.cleaned_data[question_id]

        if "attention_check_question" in question.keys():
            if (
                question["attention_check_question"]
                and answer_text not in question["correct_answers"]
            ):
                failed_checks += 1

    if failed_checks > ATTENTION_CHECK_FAILS_ALLOWED:
        return False

    return True
