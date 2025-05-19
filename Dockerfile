# Используем Python 3.11
FROM python:3.11-slim

# Устанавливаем зависимости для OpenCV и других библиотек
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Создаем и переходим в рабочую директорию
WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект (исключая ненужные файлы через .dockerignore)
COPY . .

# Указываем порт (можно заменить на ваш, если нужно)
EXPOSE 8000

# Команда для запуска сервера (используем Gunicorn для продакшна)
RUN python manage.py migrate
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "photoprocessor.wsgi:application"]