from django.db import models
from django.core.files.storage import default_storage
from django.utils import timezone
import os


class MediaFile(models.Model):
    PROCESSING_CHOICES = [
        ('pending', 'Ожидает'),
        ('processing', 'В обработке'),
        ('completed', 'Завершено'),
        ('failed', 'Ошибка')
    ]

    FILE_TYPE_CHOICES = [
        ('image', 'Изображение'),
        ('video', 'Видео')
    ]

    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_type = models.CharField(
        max_length=10,
        choices=FILE_TYPE_CHOICES,
        verbose_name='Тип файла'
    )
    processing_status = models.CharField(
        max_length=20,
        choices=PROCESSING_CHOICES,
        default='completed',
        verbose_name='Статус обработки'
    )
    duration = models.FloatField(
        null=True,
        blank=True,
        verbose_name='Длительность (сек)',
        help_text='Для видеофайлов'
    )
    resolution = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        verbose_name='Разрешение'
    )

    class Meta:
        verbose_name = 'Медиафайл'
        verbose_name_plural = 'Медиафайлы'
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.get_file_type_display()}: {self.filename}"

    @property
    def file_exists(self):
        """Проверяет существование файла с учетом storage"""
        try:
            return default_storage.exists(self.file.name)
        except:
            return False

    @property
    def filename(self):
        """Возвращает только имя файла без пути"""
        return os.path.basename(self.file.name)

    @property
    def file_size(self):
        """Возвращает реальный размер файла"""
        try:
            if default_storage.exists(self.file.name):
                return default_storage.size(self.file.name)
            return None
        except:
            return None

    @property
    def formatted_size(self):
        """Возвращает размер в удобном формате"""
        size = self.file_size
        if size is None:
            return "N/A"
        if size < 1024:
            return f"{size} bytes"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.1f} MB"

    def save(self, *args, **kwargs):
        """Определяем тип файла при сохранении"""
        if not self.pk:  # Только для новых файлов
            ext = os.path.splitext(self.file.name)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                self.file_type = 'image'
            elif ext in ['.mp4', '.mov', '.avi', '.mkv']:
                self.file_type = 'video'

        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """Удаляет файл и запись"""
        if self.file_exists:
            try:
                default_storage.delete(self.file.name)
            except:
                pass
        super().delete(*args, **kwargs)