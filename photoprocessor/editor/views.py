from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import default_storage
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.utils import timezone
from django.conf import settings
from .models import MediaFile
from .forms import UploadFileForm
import os
import cv2
import numpy as np
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

# Глобальные настройки обработки
FFMPEG_PATH = 'ffmpeg'  # или полный путь к бинарнику
MAX_VIDEO_DURATION = 600  # 10 минут в секундах
YOLO_MODEL = YOLO('yolov8n.pt')

def upload_file(request):
    # Автоматическая очистка битых записей
    MediaFile.objects.filter(
        file__in=[f.file for f in MediaFile.objects.all() if not default_storage.exists(f.file.name)]
    ).delete()

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save(commit=False)
            file_name = instance.file.name.lower()

            # Определяем тип файла
            if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                instance.file_type = 'image'
            elif file_name.endswith(('.mp4', '.mov', '.avi', '.mkv')):
                instance.file_type = 'video'
            else:
                messages.error(request, 'Неподдерживаемый формат файла')
                return redirect('upload_file')

            instance.save()
            messages.success(request, 'Файл успешно загружен!')
            return redirect('upload_file')
    else:
        form = UploadFileForm()

    files = MediaFile.objects.all().order_by('-uploaded_at')
    return render(request, 'editor/upload.html', {
        'form': form,
        'files': files
    })


def view_file(request, file_id):
    file_obj = get_object_or_404(MediaFile, id=file_id)
    return render(request, 'editor/view_file.html', {
        'file': file_obj,
        'is_video': file_obj.file_type == 'video'
    })


def edit_file(request, file_id):
    file_obj = get_object_or_404(MediaFile, id=file_id)

    # Параметры по умолчанию
    default_params = {
        'brightness': 0,
        'contrast': 1.0,
        'saturation': 1.0,
        'blur': 0,
        'preset_filter': None
    }

    if request.method == 'POST':
        try:
            params = {
                'brightness': int(request.POST.get('brightness', 0)),
                'contrast': max(0.1, min(3.0, float(request.POST.get('contrast', 1.0)))),
                'saturation': max(0.1, min(3.0, float(request.POST.get('saturation', 1.0)))),
                'blur': max(0, min(20, int(request.POST.get('blur', 0)))),
                'preset_filter': request.POST.get('preset_filter'),
                'action': request.POST.get('action', 'save')
            }

            # Проверка существования файла
            if not file_obj.file_exists:
                raise ValueError("Исходный файл не найден")

            # Обработка видео
            if file_obj.file_type == 'video':
                return process_video_file(request, file_obj, params)

            # Обработка изображения
            return process_image_file(request, file_obj, params)

        except Exception as e:
            messages.error(request, f'Ошибка обработки: {str(e)}')
            return render(request, 'editor/edit.html', {
                'file': file_obj,
                'params': params if 'params' in locals() else default_params,
                'error': str(e)
            })

    return render(request, 'editor/edit.html', {
        'file': file_obj,
        'params': default_params,
        'is_video': file_obj.file_type == 'video'
    })


def process_image_file(request, file_obj, params):
    """Обработка и сохранение изображения"""
    try:
        img = cv2.imread(file_obj.file.path)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")

        # Применение эффектов
        img = apply_effects(img, params)

        # Сохранение результата
        if params['action'] == 'save_copy':
            new_file = save_as_copy(img, file_obj)
            messages.success(request, 'Копия изображения создана!')
            return redirect('upload_file')
        else:
            save_original(img, file_obj)
            messages.success(request, 'Изменения сохранены!')
            return redirect('view_file', file_id=file_obj.id)

    except Exception as e:
        raise Exception(f"Ошибка обработки изображения: {str(e)}")


def process_video_file(request, file_obj, params):
    """Обработка и сохранение видео"""
    try:
        file_obj.processing_status = 'processing'
        file_obj.save()

        input_path = file_obj.file.path
        output_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
        os.makedirs(output_dir, exist_ok=True)

        if params['action'] == 'save_copy':
            output_path = os.path.join(output_dir, f"copy_{int(time.time())}_{os.path.basename(input_path)}")
            process_video_with_ffmpeg(input_path, output_path, params)

            new_file = MediaFile.objects.create(
                file=os.path.relpath(output_path, settings.MEDIA_ROOT),
                file_type='video',
                processing_status='completed'
            )
            messages.success(request, 'Копия видео создана!')
            return redirect('upload_file')
        else:
            output_path = os.path.join(output_dir, f"processed_{int(time.time())}_{os.path.basename(input_path)}")
            process_video_with_ffmpeg(input_path, output_path, params)

            file_obj.file = os.path.relpath(output_path, settings.MEDIA_ROOT)
            file_obj.processing_status = 'completed'
            file_obj.save()
            messages.success(request, 'Видео успешно обработано!')
            return redirect('view_file', file_id=file_obj.id)

    except Exception as e:
        file_obj.processing_status = 'failed'
        file_obj.save()
        raise Exception(f"Ошибка обработки видео: {str(e)}")


def process_video_with_ffmpeg(input_path, output_path, params):
    """Применяет эффекты к видео через FFmpeg"""
    filters = []

    # Яркость/контраст
    if params['brightness'] != 0 or params['contrast'] != 1.0:
        brightness = params['brightness'] / 100
        filters.append(f"eq=brightness={brightness}:contrast={params['contrast']}")

    # Размытие
    if params['blur'] > 0:
        filters.append(f"boxblur={params['blur']}")

    # Сборка команды FFmpeg
    cmd = [
        FFMPEG_PATH,
        '-y',  # Перезапись без подтверждения
        '-i', input_path,
        '-vf', ','.join(filters) if filters else 'null',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'copy',
        '-t', str(MAX_VIDEO_DURATION),
        output_path
    ]

    # Запуск в отдельном потоке
    with ThreadPoolExecutor() as executor:
        future = executor.submit(
            subprocess.run,
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        result = future.result()

        if result.returncode != 0:
            raise Exception(result.stderr.decode())


def apply_effects(img, params):
    """Применяет эффекты к изображению"""
    # Яркость и контраст
    img = cv2.convertScaleAbs(img, alpha=params['contrast'], beta=params['brightness'])

    # Насыщенность
    if params['saturation'] != 1.0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[..., 1] = np.clip(hsv[..., 1] * params['saturation'], 0, 255)
        img = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

    # Размытие
    if params['blur'] > 0:
        kernel_size = params['blur'] * 2 + 1
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Применение фильтра
    if params['preset_filter']:
        img = apply_preset_filter(img, params['preset_filter'])

    return img


# В views.py добавляем обработку фильтров
def apply_preset_filter(img, filter_name):
    if filter_name == 'bw':
        # Черно-белый (с оттенками серого)
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif filter_name == 'noisy':
        # Зашумленный (зернистость)
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        return cv2.add(img, noise)
    elif filter_name == 'vintage':
        # Винтаж (сепия + виньетирование)
        sepia = np.array([[0.272, 0.534, 0.131],
                         [0.349, 0.686, 0.168],
                         [0.393, 0.769, 0.189]])
        img = cv2.transform(img, sepia)
        rows, cols = img.shape[:2]
        vignette = np.zeros((rows, cols), dtype=np.uint8)
        cv2.circle(vignette, (cols//2, rows//2), min(rows, cols)//2, (255), -1)
        vignette = cv2.GaussianBlur(vignette, (501, 501), 0)
        return cv2.addWeighted(img, 0.7, cv2.cvtColor(vignette, cv2.COLOR_GRAY2BGR), 0.3, 0)
    elif filter_name == 'cold':
        # Холодные тона (усиление синего)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = np.clip(hsv[..., 0] * 0.9, 0, 179)  # Сдвиг в синюю область
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.2, 0, 255)  # Увеличение насыщенности
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif filter_name == 'warm':
        # Теплые тона (янтарный)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = np.clip(hsv[..., 0] * 1.1, 0, 179)  # Сдвиг в оранжевую область
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255)  # Увеличение насыщенности
        hsv[..., 2] = np.clip(hsv[..., 2] * 1.1, 0, 255)  # Увеличение яркости
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif filter_name == 'dramatic':
        # Драматичный (высокий контраст + затемнение)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif filter_name == 'pastel':
        # Пастельный (мягкие тона)
        blurred = cv2.GaussianBlur(img, (0,0), 3)
        return cv2.addWeighted(img, 0.6, blurred, 0.4, 0)
    elif filter_name == 'neon':
        # Неоновый (усиление цветов)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * 2.5, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * 1.2, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def save_as_copy(img, original_file):
    """Создает копию файла"""
    processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    filename = f"copy_{int(time.time())}_{original_file.id}_{os.path.basename(original_file.file.name)}"
    output_path = os.path.join(processed_dir, filename)

    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    new_file = MediaFile(
        file=os.path.join('processed', filename),
        file_type=original_file.file_type,
        uploaded_at=timezone.now()
    )
    new_file.save()
    return new_file


def save_original(img, file_obj):
    """Перезаписывает оригинальный файл"""
    processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    filename = f"processed_{int(time.time())}_{file_obj.id}_{os.path.basename(file_obj.file.name)}"
    output_path = os.path.join(processed_dir, filename)

    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    file_obj.file = os.path.join('processed', filename)
    file_obj.save()

@require_POST
def delete_file(request, file_id):
    file_obj = get_object_or_404(MediaFile, id=file_id)

    try:
        # Удаляем физический файл
        file_path = file_obj.file.path
        if os.path.exists(file_path):
            os.remove(file_path)

        # Удаляем запись из БД
        file_obj.delete()

        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def analyze_file(request, file_id):
    """Обработка файла через YOLO с кэшированием результатов"""
    file_obj = get_object_or_404(MediaFile, id=file_id)

    if not file_obj.file_exists:
        messages.error(request, "Файл не найден")
        return redirect('upload_file')

    # Если есть кэшированный результат и файл существует - используем его
    if file_obj.yolo_processed_file and default_storage.exists(file_obj.yolo_processed_file.name):
        return render_cached_result(request, file_obj)

    # Иначе проводим анализ
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, file_obj.file.name)

        if file_obj.file_type == 'image':
            results = YOLO_MODEL(file_path)
            annotated_img_path = save_yolo_result_image(results, file_obj)
            detection_data = prepare_detection_data(results)

            # Сохраняем результаты в модель
            file_obj.yolo_processed_file = os.path.relpath(annotated_img_path, settings.MEDIA_ROOT)
            file_obj.yolo_objects = detection_data
            file_obj.yolo_processed_at = timezone.now()
            file_obj.save()

            return render(request, 'editor/analyze.html', {
                "file": file_obj,
                "annotated_img_url": annotated_img_path,
                "objects": detection_data,
                "is_video": False
            })

        elif file_obj.file_type == 'video':
            output_filename = f"processed_{file_obj.id}_{os.path.basename(file_obj.file.name)}"
            output_path = os.path.join(settings.MEDIA_ROOT, 'yolo_results', output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            process_video_with_yolo(
                os.path.join(settings.MEDIA_ROOT, file_obj.file.name),
                output_path
            )

            preview_filename = f"preview_{file_obj.id}.jpg"
            preview_path = os.path.join(settings.MEDIA_ROOT, 'yolo_results', preview_filename)
            extract_video_preview(output_path, preview_path)

            # Сохраняем результаты в модель
            file_obj.yolo_processed_file = os.path.relpath(output_path, settings.MEDIA_ROOT)
            file_obj.yolo_processed_at = timezone.now()
            file_obj.save()

            return render(request, 'editor/analyze_video.html', {
                "file": file_obj,
                "processed_video_url": os.path.join(settings.MEDIA_URL, 'yolo_results', output_filename).replace('\\',
                                                                                                                 '/'),
                "preview_url": os.path.join(settings.MEDIA_URL, 'yolo_results', preview_filename).replace('\\', '/'),
                "is_video": True
            })

    except Exception as e:
        messages.error(request, f"Ошибка анализа: {str(e)}")
        return redirect('view_file', file_id=file_id)


def render_cached_result(request, file_obj):
    """Рендерит кэшированный результат анализа"""
    if file_obj.file_type == 'image':
        return render(request, 'editor/analyze.html', {
            "file": file_obj,
            "annotated_img_url": file_obj.yolo_processed_file.url,
            "objects": file_obj.yolo_objects or [],
            "is_video": False
        })
    else:
        preview_filename = f"preview_{file_obj.id}.jpg"
        preview_path = os.path.join(settings.MEDIA_ROOT, 'yolo_results', preview_filename)

        # Если превью нет - создаем
        if not os.path.exists(preview_path):
            extract_video_preview(file_obj.yolo_processed_file.path, preview_path)

        return render(request, 'editor/analyze_video.html', {
            "file": file_obj,
            "processed_video_url": file_obj.yolo_processed_file.url,
            "preview_url": os.path.join(settings.MEDIA_URL, 'yolo_results', preview_filename).replace('\\', '/'),
            "is_video": True
        })


def process_video_with_yolo(input_path, output_path):
    """Обрабатывает видео через YOLO и сохраняет результат"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видеофайл")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Используем более совместимый кодек
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 кодек
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise ValueError("Не удалось создать выходной видеофайл")

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Обрабатываем каждый N-й кадр для ускорения (например, каждый 2-й)
            if frame_count % 2 == 0:
                results = YOLO_MODEL(frame)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

            frame_count += 1

            # Логирование прогресса
            if frame_count % 10 == 0:
                print(f"Обработано кадров: {frame_count}")

    finally:
        cap.release()
        out.release()
        print(f"Видео обработка завершена. Всего кадров: {frame_count}")


def extract_video_preview(video_path, output_path):
    """Извлекает первый кадр видео для превью"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    cap.release()


def save_yolo_result_image(results, file_obj):
    """Сохраняет изображение с bounding boxes и возвращает абсолютный путь"""
    try:
        output_dir = os.path.join(settings.MEDIA_ROOT, 'yolo_results')
        os.makedirs(output_dir, exist_ok=True)

        original_name = os.path.basename(file_obj.file.name)
        output_filename = f"image_{file_obj.id}_{original_name}"
        output_path = os.path.join(output_dir, output_filename)

        for result in results:
            im_array = result.plot()
            cv2.imwrite(output_path, im_array)

        return output_path  # Возвращаем абсолютный путь
    except Exception as e:
        print(f"Ошибка сохранения результата: {str(e)}")
        return None

def prepare_detection_data(results):
    """Формирует список обнаруженных объектов из результатов YOLO"""
    detection_data = []
    for result in results:
        for box in result.boxes:
            detection_data.append({
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]  # Координаты bounding box
            })
    # Сортируем по уверенности (от высокой к низкой)
    return sorted(detection_data, key=lambda x: x['confidence'], reverse=True)


@require_POST
def reanalyze_file(request, file_id):
    """Принудительно перезапускает анализ YOLO"""
    file_obj = get_object_or_404(MediaFile, id=file_id)

    # Удаляем старые результаты
    if file_obj.yolo_processed_file:
        try:
            # Удаляем физический файл
            if default_storage.exists(file_obj.yolo_processed_file.name):
                default_storage.delete(file_obj.yolo_processed_file.name)
        except Exception as e:
            print(f"Ошибка удаления файла: {str(e)}")

    # Очищаем поля в БД
    file_obj.yolo_processed_file = None
    file_obj.yolo_objects = None
    file_obj.yolo_processed_at = None
    file_obj.save()

    return redirect('analyze_file', file_id=file_id)