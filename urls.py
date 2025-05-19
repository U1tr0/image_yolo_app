from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView  # Добавьте этот импорт

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', RedirectView.as_view(url='upload/')),  # Перенаправление с / на /upload/
    path('', include('editor.urls')),  # Оставьте это для других URL приложения
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)