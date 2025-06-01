import os
import json
from datetime import datetime
import googleapiclient.discovery
from langdetect import detect, LangDetectException
import logging
import argparse
from googleapiclient.http import HttpError
import time
import isodate

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Настройка YouTube API
API_KEY = os.getenv('YOUTUBE_API_KEY')
if not API_KEY:
    logger.error("Переменная окружения YOUTUBE_API_KEY не установлена")
    exit(1)
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=API_KEY, cache_discovery=False)

# Путь к файлу кэша видео
VIDEO_CACHE_FILE = 'video_cache.json'

def load_video_cache():
    """Загрузка кэша видео из файла"""
    try:
        if os.path.exists(VIDEO_CACHE_FILE):
            with open(VIDEO_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Ошибка при загрузке кэша видео: {str(e)}")
        return {}

def save_video_cache(cache):
    """Сохранение кэша видео в файл"""
    try:
        with open(VIDEO_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка при сохранении кэша видео: {str(e)}")

def is_language_content(title, description, target_language):
    """Проверка, соответствует ли контент указанному языку"""
    if not target_language:  # Если язык не указан, пропускаем проверку
        return True
    try:
        if title and detect(title) == target_language:
            return True
        if description and detect(description) == target_language:
            return True
        return False
    except LangDetectException:
        logger.warning(f"Не удалось определить язык контента (цель: {target_language})")
        return False

def get_video_durations(video_ids, video_cache):
    """Получение длительности видео с кэшированием"""
    durations = {}
    uncached_ids = [vid for vid in video_ids if vid not in video_cache or 'duration' not in video_cache[vid]]
    
    if uncached_ids:
        for i in range(0, len(uncached_ids), 50):
            batch = uncached_ids[i:i+50]
            try:
                request = youtube.videos().list(
                    part="contentDetails",
                    id=','.join(batch)
                )
                response = request.execute()
                time.sleep(0.2)  # Задержка для снижения нагрузки
                for item in response.get('items', []):
                    vid = item['id']
                    duration = item.get('contentDetails', {}).get('duration', 'PT0S')
                    video_cache[vid] = video_cache.get(vid, {})
                    video_cache[vid]['duration'] = duration
                    durations[vid] = duration
                    logger.debug(f"Получена длительность для видео {vid}: {duration}")
                save_video_cache(video_cache)
            except HttpError as e:
                if e.resp.status == 403 and 'quotaExceeded' in str(e):
                    logger.error("Квота YouTube API превышена. Проверьте квоту в Google Cloud Console или используйте новый API-ключ.")
                    raise
                logger.error(f"Ошибка при получении длительности видео: {str(e)}")
                break

    for vid in video_ids:
        if vid in video_cache and 'duration' in video_cache[vid]:
            durations[vid] = video_cache[vid]['duration']
        elif vid not in durations:
            durations[vid] = 'PT0S'
            logger.warning(f"Длительность для видео {vid} неизвестна, установлено PT0S")
    return durations

def duration_to_seconds(duration):
    """Конвертация ISO 8601 длительности в секунды"""
    try:
        parsed = isodate.parse_duration(duration)
        return int(parsed.total_seconds())
    except Exception as e:
        logger.error(f"Ошибка при конвертации длительности {duration}: {str(e)}")
        return 0

def filter_videos(input_file, min_duration_seconds=60, target_language=None):
    """Фильтрация видео по языку (если указан) и длительности"""
    filtered_videos = []
    video_cache = load_video_cache()

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            videos = json.load(f)
        logger.info(f"Найдено {len(videos)} видео для фильтрации")
    except Exception as e:
        logger.error(f"Ошибка при чтении {input_file}: {str(e)}")
        return

    video_ids = [video['video_id'] for video in videos]
    try:
        durations = get_video_durations(video_ids, video_cache)
    except HttpError:
        logger.error("Прервано из-за ошибки API")
        return

    for video in videos:
        video_id = video['video_id']
        title = video['title']
        description = video['description']

        # Проверка языка (если указан)
        if target_language and not is_language_content(title, description, target_language):
            logger.debug(f"Видео исключено (не на языке {target_language}): {title}")
            continue

        # Проверка длительности
        duration = durations.get(video_id, 'PT0S')
        duration_seconds = duration_to_seconds(duration)
        if duration_seconds < min_duration_seconds:
            logger.debug(f"Видео исключено (слишком короткое, {duration_seconds} секунд): {title}")
            continue

        filtered_videos.append(video)
        logger.info(f"Видео прошло фильтр: {title} (длительность: {duration_seconds} секунд)")

    output_file = 'filtered_videos.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_videos, f, ensure_ascii=False, indent=2)
        logger.info(f"Фильтрация завершена! Найдено {len(filtered_videos)} видео, соответствующих критериям.")
        logger.info(f"Результаты сохранены в {output_file}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении результатов в {output_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Фильтрация видео по языку и длительности")
    parser.add_argument('--input-file', type=str, default='all_videos.json',
                        help='Входной файл с видео (по умолчанию: all_videos.json)')
    parser.add_argument('--min-duration', type=int, default=60,
                        help='Минимальная длительность видео в секундах (по умолчанию: 60)')
    parser.add_argument('--language', type=str, default=None,
                        help='Код языка для фильтрации (например, ru, en), если не указан, фильтр по языку отключается')
    args = parser.parse_args()

    filter_videos(args.input_file, args.min_duration, args.language)

if __name__ == "__main__":
    main()