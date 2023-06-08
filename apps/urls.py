from django.contrib import admin
from django.urls import path
from apps import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path('', views.index, name='index' ),
    path('home', views.home, name='home' ),
    path('reload', views.reload, name='reload' )
]