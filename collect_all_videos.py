import os
import json
from datetime import datetime, timedelta
import googleapiclient.discovery
import logging
import argparse
from googleapiclient.http import HttpError
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Настройка YouTube API
API_KEY = os.getenv('YOUTUBE_API_KEY')
if not API_KEY:
    logger.error("Переменная окружения YOUTUBE_API_KEY не установлена")
    exit(1)
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=API_KEY, cache_discovery=False)

# Путь к файлу кэша каналов
CHANNEL_CACHE_FILE = 'channel_cache.json'

def load_channel_cache():
    """Загрузка кэша каналов из файла"""
    try:
        if os.path.exists(CHANNEL_CACHE_FILE):
            with open(CHANNEL_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Ошибка при загрузке кэша каналов: {str(e)}")
        return {}

def save_channel_cache(cache):
    """Сохранение кэша каналов в файл"""
    try:
        with open(CHANNEL_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка при сохранении кэша каналов: {str(e)}")

def get_subscriber_count(channel_ids, channel_cache):
    """Получение количества подписчиков для списка каналов с кэшированием"""
    stats = {}
    uncached_ids = [cid for cid in channel_ids if cid not in channel_cache or 'subscriber_count' not in channel_cache[cid]]
    
    if uncached_ids:
        for i in range(0, len(uncached_ids), 50):
            batch = uncached_ids[i:i+50]
            try:
                request = youtube.channels().list(
                    part="statistics",
                    id=','.join(batch)
                )
                response = request.execute()
                time.sleep(0.2)  # Задержка для снижения нагрузки
                for item in response.get('items', []):
                    cid = item['id']
                    subs = int(item['statistics'].get('subscriberCount', 0))
                    channel_cache[cid] = channel_cache.get(cid, {})
                    channel_cache[cid]['subscriber_count'] = subs
                    stats[cid] = subs
                save_channel_cache(channel_cache)
            except HttpError as e:
                if e.resp.status == 403 and 'quotaExceeded' in str(e):
                    logger.error("Квота YouTube API превышена. Проверьте квоту в Google Cloud Console или используйте новый API-ключ.")
                    raise
                logger.error(f"Ошибка при получении статистики каналов: {str(e)}")

    # Дополняем статистику из кэша
    for cid in channel_ids:
        if cid in channel_cache and 'subscriber_count' in channel_cache[cid]:
            stats[cid] = channel_cache[cid]['subscriber_count']
        elif cid not in stats:
            stats[cid] = 0
    return stats

def is_channel_active(channel_id, channel_cache, skip_activity_check=False):
    """Проверка активности канала за последние 6 месяцев"""
    if skip_activity_check:
        return True
    if channel_id in channel_cache and 'is_active' in channel_cache[channel_id]:
        return channel_cache[channel_id]['is_active']
    try:
        six_months_ago = (datetime.now() - timedelta(days=180)).isoformat() + 'Z'
        request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            publishedAfter=six_months_ago,
            maxResults=1
        )
        response = request.execute()
        time.sleep(0.2)  # Задержка для снижения нагрузки
        is_active = len(response.get('items', [])) > 0
        channel_cache[channel_id] = channel_cache.get(channel_id, {})
        channel_cache[channel_id]['is_active'] = is_active
        save_channel_cache(channel_cache)
        return is_active
    except HttpError as e:
        if e.resp.status == 403 and 'quotaExceeded' in str(e):
            logger.error("Квота YouTube API превышена. Проверьте квоту в Google Cloud Console или используйте новый API-ключ.")
            raise
        logger.error(f"Ошибка при проверке активности канала {channel_id}: {str(e)}")
        return False

def collect_videos(start_date, end_date, min_subscribers=500000, skip_activity_check=False):
    """Сбор всех видео за указанный диапазон дат от каналов с min_subscribers+ подписчиков"""
    all_videos = []
    channel_cache = load_channel_cache()
    current_date = start_date

    while current_date <= end_date:
        next_date = current_date + timedelta(days=1)
        logger.info(f"Сбор видео за {current_date.date()}")

        start_time = current_date.isoformat() + 'Z'
        end_time = (next_date - timedelta(seconds=1)).isoformat() + 'Z'

        try:
            request = youtube.search().list(
                part="snippet",
                type="video",
                publishedAfter=start_time,
                publishedBefore=end_time,
                maxResults=50,  # Максимум для эффективности
                order="date"
            )

            channel_ids = set()
            videos_batch = []
            while request:
                response = request.execute()
                time.sleep(0.2)  # Задержка для снижения нагрузки
                videos_batch.extend(response.get('items', []))
                channel_ids.update(item['snippet']['channelId'] for item in response.get('items', []))
                request = youtube.search().list_next(request, response)

            # Получаем статистику каналов
            channel_stats = get_subscriber_count(list(channel_ids), channel_cache)

            # Фильтруем видео по подписчикам и активности
            for item in videos_batch:
                channel_id = item['snippet']['channelId']
                if channel_stats.get(channel_id, 0) < min_subscribers:
                    continue
                if not is_channel_active(channel_id, channel_cache, skip_activity_check):
                    continue

                video_info = {
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'channel_id': channel_id,
                    'channel_title': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'description': item['snippet']['description']
                }
                all_videos.append(video_info)
                logger.info(f"Найдено видео: {video_info['title']}")

        except HttpError as e:
            if e.resp.status == 403 and 'quotaExceeded' in str(e):
                logger.error("Квота YouTube API превышена. Остановка сбора.")
                logger.info("Посетите https://console.cloud.google.com/apis/credentials для проверки квоты или запроса её увеличения.")
                break
            logger.error(f"Ошибка при сборе видео за {current_date.date()}: {str(e)}")
            break

        current_date = next_date

    # Сохранение результатов
    output_file = 'all_videos.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_videos, f, ensure_ascii=False, indent=2)
        logger.info(f"Сбор завершён! Найдено {len(all_videos)} видео, соответствующих критериям.")
        logger.info(f"Результаты сохранены в {output_file}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении результатов в {output_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Сбор всех видео с YouTube от каналов с 500k+ подписчиков")
    parser.add_argument('--start-date', type=str, default='2024-05-26',
                        help='Начальная дата в формате ГГГГ-ММ-ДД (по умолчанию: 2024-05-26)')
    parser.add_argument('--end-date', type=str, default='2024-06-01',
                        help='Конечная дата в формате ГГГГ-ММ-ДД (по умолчанию: 2024-06-01)')
    parser.add_argument('--min-subscribers', type=int, default=500000,
                        help='Минимальное количество подписчиков (по умолчанию: 500000)')
    parser.add_argument('--skip-activity-check', action='store_true',
                        help='Пропустить проверку активности канала для экономии квоты')
    args = parser.parse_args()

    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        if start_date > end_date:
            logger.error("Начальная дата должна быть раньше конечной")
            exit(1)
        if end_date > datetime.now():
            logger.warning("Конечная дата в будущем; возможно, результаты будут пустыми")
    except ValueError as e:
        logger.error(f"Неверный формат даты. Используйте ГГГГ-ММ-ДД: {str(e)}")
        exit(1)

    collect_videos(start_date, end_date, args.min_subscribers, args.skip_activity_check)

if __name__ == "__main__":
    main()