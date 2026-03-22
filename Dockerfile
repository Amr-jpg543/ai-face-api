FROM python:3.10-slim

# منع مشاكل الـ buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# تثبيت dependencies أساسية (مهمة لـ opencv و insightface)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# نسخ الملفات
COPY . /app

# تحديث pip
RUN pip install --upgrade pip

# تثبيت requirements
RUN pip install --no-cache-dir -r requirements.txt

# فتح البورت الخاص بـ HuggingFace
EXPOSE 7860

# تشغيل السيرفر
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "7860"]