from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_file, name='upload_file'),
    path('file/<int:file_id>/', views.view_file, name='view_file'),
    path('edit/<int:file_id>/', views.edit_file, name='edit_file'),
    path('delete/<int:file_id>/', views.delete_file, name='delete_file'),
    path('analyze/<int:file_id>/', views.analyze_file, name='analyze_file'),  # Новый путь
]