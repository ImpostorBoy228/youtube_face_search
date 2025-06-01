import cv2
import numpy as np
import face_recognition
import requests
from io import BytesIO
from PIL import Image
import json
import os
import tempfile
import shutil
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import argparse
import yt_dlp
import logging
import time
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('face_search.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_image(url):
    """Скачивает изображение по URL"""
    logger.info(f"Скачивание изображения: {url}")
    try:
        response = requests.get(url, timeout=10)
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logger.error(f"Ошибка скачивания изображения: {e}")
        return None

def load_known_faces(known_faces_dir):
    """Загружает известные лица из папки"""
    logger.info(f"Загрузка известных лиц из папки: {known_faces_dir}")
    known_face_encodings = []
    known_face_names = []
    if not os.path.exists(known_faces_dir):
        logger.error(f"Папка {known_faces_dir} не найдена")
        return [], []
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(known_faces_dir, filename)
            logger.info(f"Обработка файла: {filename}")
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
                logger.info(f"Загружено лицо: {filename}")
            else:
                logger.warning(f"Лицо не найдено в {filename}")
    logger.info(f"Загружено {len(known_face_encodings)} известных лиц")
    return known_face_encodings, known_face_names

def is_frame_blurry(frame, threshold=150):
    """Проверяет, является ли кадр размытым, с помощью метрики Лапласа"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def has_low_contrast(frame, threshold=30):
    """Проверяет, имеет ли кадр низкий контраст"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    return contrast < threshold

def detect_known_faces_in_image(image, known_face_encodings, known_face_names):
    """Ищет известные лица на изображении"""
    if image is None:
        logger.warning("Изображение отсутствует")
        return False
    logger.info("Поиск известных лиц в изображении")
    try:
        image_np = np.array(image)
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            logger.info("Лица в изображении не найдены")
            return False
        for (top, right, bottom, left) in face_locations:
            if (right - left) < 40 or (bottom - top) < 40:
                logger.debug("Пропущено лицо: слишком маленькое")
                continue
            face_encodings = face_recognition.face_encodings(rgb_image, [(top, right, bottom, left)])
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
                if True in matches:
                    match_index = matches.index(True)
                    logger.info(f"Найдено совпадение с лицом: {known_face_names[match_index]}")
                    return True
        logger.info("Совпадений с известными лицами не найдено")
        return False
    except Exception as e:
        logger.error(f"Ошибка при поиске лиц в изображении: {e}")
        return False

def extract_frames(video_path, interval=2.5, look_around=0.25):
    """Извлекает кадры из видео каждые 2.5 секунд, выбирая самый чёткий кадр в окне ±0.25 сек"""
    logger.info(f"Извлечение кадров из видео: {video_path}")
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        look_around_frames = int(fps * look_around)  # Количество кадров для проверки (±0.25 сек)
        frames = []
        timestamps = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while frame_count < total_frames:
            # Переходим к целевой временной метке
            target_frame = frame_count
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, target_frame_data = cap.read()
            if not ret:
                break

            # Проверяем кадры в окне ±0.25 сек
            best_frame = None
            best_laplacian = -1
            best_frame_pos = target_frame
            start_pos = max(0, target_frame - look_around_frames)
            end_pos = min(total_frames, target_frame + look_around_frames + 1)

            for pos in range(start_pos, end_pos):
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var > best_laplacian:
                    best_laplacian = laplacian_var
                    best_frame = frame
                    best_frame_pos = pos

            # Проверяем, что лучший кадр не размытый и имеет достаточный контраст
            if best_frame is not None:
                if is_frame_blurry(best_frame, threshold=150):
                    logger.debug(f"Кадр около {target_frame/fps:.2f} сек: пропущен, размытый")
                elif has_low_contrast(best_frame, threshold=30):
                    logger.debug(f"Кадр около {target_frame/fps:.2f} сек: пропущен, низкий контраст")
                else:
                    frames.append(best_frame)
                    timestamps.append(best_frame_pos / fps)
                    logger.debug(f"Выбран кадр на {best_frame_pos/fps:.2f} сек (чёткость: {best_laplacian:.2f})")

            frame_count += frame_interval

        cap.release()
        logger.info(f"Извлечено {len(frames)} кадров")
        return frames, timestamps
    except Exception as e:
        logger.error(f"Ошибка извлечения кадров: {e}")
        return [], []

def detect_known_faces_in_frames(frames, timestamps, known_face_encodings, known_face_names):
    """Ищет известные лица в кадрах видео"""
    logger.info("Поиск известных лиц в кадрах видео")
    try:
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            logger.debug(f"Обработка кадра {i+1}/{len(frames)} на {timestamp:.2f} сек")
            if is_frame_blurry(frame, threshold=150):
                logger.debug(f"Кадр {i+1} на {timestamp:.2f} сек: пропущен, размытый")
                continue
            if has_low_contrast(frame, threshold=30):
                logger.debug(f"Кадр {i+1} на {timestamp:.2f} сек: пропущен, низкий контраст")
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            if not face_locations:
                logger.debug(f"Кадр {i+1} на {timestamp:.2f} сек: лица не найдены")
                continue
            for (top, right, bottom, left) in face_locations:
                if (right - left) < 40 or (bottom - top) < 40:
                    logger.debug(f"Кадр {i+1} на {timestamp:.2f} сек: пропущено лицо, слишком маленькое")
                    continue
                face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
                    if True in matches:
                        match_index = matches.index(True)
                        logger.info(f"Найдено совпадение с лицом {known_face_names[match_index]} на {timestamp:.2f} сек")
                        return True
        logger.info("Совпадений с известными лицами в кадрах не найдено")
        return False
    except Exception as e:
        logger.error(f"Ошибка при поиске лиц в кадрах: {e}")
        return False

def download_video(video_id, temp_dir, retries=3):
    """Скачивает видео с YouTube через yt-dlp с повторными попытками"""
    logger.info(f"Скачивание видео: {video_id}")
    for attempt in range(retries):
        try:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            output_path = os.path.join(temp_dir, f"{video_id}.mp4")
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': output_path,
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            if os.path.exists(output_path):
                logger.info(f"Видео {video_id} успешно скачано")
                return output_path
            logger.warning(f"Видео {video_id} не найдено после скачивания")
            return None
        except Exception as e:
            logger.error(f"Попытка {attempt+1}/{retries} скачивания {video_id} не удалась: {e}")
            time.sleep(5)
    logger.error(f"Не удалось скачать {video_id} после {retries} попыток")
    return None

def get_video_and_channel_info(video_id, channel_id, youtube, retries=3):
    """Получает URL превью и аватарки канала через YouTube API с повторными попытками"""
    logger.info(f"Получение данных для видео {video_id} и канала {channel_id}")
    for attempt in range(retries):
        try:
            video_response = youtube.videos().list(
                part="snippet",
                id=video_id
            ).execute()
            thumbnail_url = video_response['items'][0]['snippet']['thumbnails'].get('high', {}).get('url', '')
            channel_response = youtube.channels().list(
                part="snippet",
                id=channel_id
            ).execute()
            avatar_url = channel_response['items'][0]['snippet']['thumbnails'].get('high', {}).get('url', '')
            logger.info(f"Получены URL: превью={thumbnail_url}, аватар={avatar_url}")
            return thumbnail_url, avatar_url
        except HttpError as e:
            logger.error(f"Попытка {attempt+1}/{retries} получения данных не удалась: {e}")
            time.sleep(5)
    logger.error(f"Не удалось получить данные для {video_id} после {retries} попыток")
    return '', ''

def process_video(video, temp_dir, youtube, known_face_encodings, known_face_names):
    """Обрабатывает видео: проверяет превью, аватарку и кадры на известные лица"""
    video_id = video['video_id']
    channel_id = video['channel_id']
    logger.info(f"Начало обработки видео: {video_id}")

    try:
        thumbnail_url, avatar_url = get_video_and_channel_info(video_id, channel_id, youtube)

        has_known_face_in_thumbnail = False
        if thumbnail_url:
            thumbnail = download_image(thumbnail_url)
            has_known_face_in_thumbnail = detect_known_faces_in_image(thumbnail, known_face_encodings, known_face_names)

        has_known_face_in_avatar = False
        if avatar_url:
            avatar = download_image(avatar_url)
            has_known_face_in_avatar = detect_known_faces_in_image(avatar, known_face_encodings, known_face_names)

        has_known_face_in_video = False
        video_path = download_video(video_id, temp_dir)
        if video_path:
            frames, timestamps = extract_frames(video_path)
            has_known_face_in_video = detect_known_faces_in_frames(frames, timestamps, known_face_encodings, known_face_names)
            os.remove(video_path)
            logger.info(f"Временный файл {video_path} удалён")

        result = {
            'video_id': video_id,
            'has_known_face_in_thumbnail': has_known_face_in_thumbnail,
            'has_known_face_in_avatar': has_known_face_in_avatar,
            'has_known_face_in_video': has_known_face_in_video
        }
        logger.info(f"Обработано видео: {video_id} - "
                    f"Превью: {has_known_face_in_thumbnail}, "
                    f"Аватар: {has_known_face_in_avatar}, "
                    f"Видео: {has_known_face_in_video}")
        return result
    except Exception as e:
        logger.error(f"Ошибка обработки видео {video_id}: {e}")
        return {
            'video_id': video_id,
            'has_known_face_in_thumbnail': False,
            'has_known_face_in_avatar': False,
            'has_known_face_in_video': False
        }

def main(json_file):
    """Основная функция обработки видео из JSON"""
    logger.info("Запуск обработки видео")
    youtube = build('youtube', 'v3', developerKey='AIzaSyBTMiJ4AaAC2uNURS4w9q_pjkLrrW2d7aI')

    known_faces_dir = 'known_faces'
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)
    if not known_face_encodings:
        logger.error("Нет известных лиц для распознавания. Завершение.")
        return

    temp_dir = tempfile.mkdtemp()
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            videos = json.load(f)
        logger.info(f"Найдено {len(videos)} видео для обработки")

        results = []
        for video in videos:
            result = process_video(video, temp_dir, youtube, known_face_encodings, known_face_names)
            results.append(result)

        output_file = 'face_recognition_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Результаты сохранены в {output_file}")
    except Exception as e:
        logger.error(f"Ошибка обработки JSON: {e}")
    finally:
        shutil.rmtree(temp_dir)
        logger.info("Временная папка удалена")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Распознавание лиц в видео YouTube")
    parser.add_argument('--json-file', type=str, required=True, help='Путь к JSON-файлу с видео')
    args = parser.parse_args()
    main(args.json_file)