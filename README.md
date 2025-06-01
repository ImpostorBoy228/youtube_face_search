# YouTube Face Search

This script searches for specific people in YouTube videos within a given date range.

## Requirements
- Python 3.10
- Virtual environment with installed dependencies
- Reference face images in `known_faces` directory

## Setup
1. Create a `known_faces` directory and add clear photos of the people you want to find
2. Make sure the virtual environment is activated:
```bash
source venv/bin/activate  # On Linux/WSL
```

## Features
- Searches videos between May 26 and June 1, 2025
- Checks both Russian and international channels
- Only processes channels with 500k+ subscribers
- Skips inactive channels (no posts in 6 months)
- Extracts frames every 2.5 seconds from videos
- Performs face recognition on extracted frames
- Supports multiple reference faces

## Running the Script
```bash
python youtube_search.py
```

The script will output URLs of videos where any of the target people are found.

## Notes
- The script uses the YouTube Data API v3, which has quota limits
- Face recognition accuracy depends on the quality of reference images
- Make sure your reference images are clear and contain only the faces you're looking for
- Supported image formats: JPG, JPEG, PNG
- Each reference image should contain only one face 