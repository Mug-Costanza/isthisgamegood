from django.urls import path

from . import views

urlpatterns = [
    path("", views.Index.as_view(), name="index"),
    path("about", views.About.as_view(), name="about"),
    path("privacy", views.Privacy.as_view(), name="privacy"),
    path("terms", views.Terms.as_view(), name="terms"),
]
