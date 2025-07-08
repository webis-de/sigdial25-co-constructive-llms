from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from . import views

app_name = "study"
urlpatterns = [
    path("", views.user_identifier, name="user_identifier"),
    path("<str:user_id>/", views.form_page_view, name="form_page"),
    path(
        "form/<str:user_id>/<int:page_index>/", views.form_page_view, name="form_page"
    ),
    path("chat/chat/<str:user_id>/", views.chat_view, name="chat"),
    path(
        "attention_check_failed/<str:user_id>/",
        views.attention_check_failed_view,
        name="attention_check_failed",
    ),
    path("completion/<str:user_id>/", views.completion_view, name="completion"),
]

# Serve static and media files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
