import subprocess
import os
from django.conf import settings


def process_video(input_path, output_path, effects):
    """Применяет эффекты к видео через FFmpeg"""
    try:
        filters = []

        # Яркость/контраст
        if effects.get('brightness', 0) != 0 or effects.get('contrast', 1.0) != 1.0:
            brightness = effects.get('brightness', 0) / 100
            contrast = effects.get('contrast', 1.0)
            filters.append(f"eq=brightness={brightness}:contrast={contrast}")

        # Другие эффекты
        if effects.get('blur', 0) > 0:
            filters.append(f"boxblur={effects['blur']}")

        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Перезапись без подтверждения
            '-i', input_path,
            '-vf', ','.join(filters) if filters else 'null',
            '-c:a', 'copy',  # Сохраняем оригинальный аудиопоток
            output_path
        ]

        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")

        return True
    except Exception as e:
        raise Exception(f"Video processing failed: {str(e)}")