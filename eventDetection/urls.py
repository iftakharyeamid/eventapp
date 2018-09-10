from django.urls import path
from . import views
from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name = 'index'),
    path('news/', views.event, name = 'news'),
    path('about/', views.about, name = 'about'),
]