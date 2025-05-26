"""
Given the data set in .txt format, process individual trajectory information and extract invidiaul frames.
"""

import numpy as np
import os
import cv2
from pytube import YouTube
import tempfile
from typing import Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor
import warnings

import numpy as np
import os
import cv2
import tempfile
from typing import Tuple, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor
import warnings
import subprocess
import re
import requests
from bs4 import BeautifulSoup

class RealEstate10KLoader:
    def __init__(self, dataset_root: str, download_dir: Optional[str] = None):
        """
        Initialize the loader with multiple download strategies.
        
        Args:
            dataset_root: Path to dataset containing train/test subdirs
            download_dir: Where to cache videos (default: temp directory)
        """
        self.dataset_root = dataset_root
        self.train_dir = os.path.join(dataset_root, 'train')
        self.test_dir = os.path.join(dataset_root, 'test')
        self.download_dir = download_dir if download_dir else tempfile.mkdtemp()
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Configure user agent to mimic browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _download_video(self, url: str) -> Optional[str]:
        """
        Try multiple methods to download YouTube video.
        
        Returns:
            Path to downloaded video or None if all methods fail
        """
        video_id = self._extract_video_id(url)
        if not video_id:
            return None
            
        # Check cache first
        cache_path = os.path.join(self.download_dir, f"{video_id}.mp4")
        if os.path.exists(cache_path):
            return cache_path
            
        # Try different download methods
        methods = [
            self._download_with_yt_dlp,
            self._download_with_pytube,
            self._download_with_direct_link
        ]
        
        for method in methods:
            try:
                result = method(url, video_id)
                if result:
                    return result
            except Exception as e:
                warnings.warn(f"Method {method.__name__} failed: {str(e)}")
        
        return None

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL."""
        patterns = [
            r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})',
            r'youtube\.com\/shorts\/([^"&?\/\s]{11})'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _download_with_yt_dlp(self, url: str, video_id: str) -> Optional[str]:
        """Download using yt-dlp (most reliable)."""
        try:
            output_path = os.path.join(self.download_dir, f"{video_id}.%(ext)s")
            cmd = [
                'yt-dlp',
                '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                '-o', output_path,
                '--merge-output-format', 'mp4',
                url
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return os.path.join(self.download_dir, f"{video_id}.mp4")
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _download_with_pytube(self, url: str, video_id: str) -> Optional[str]:
        """Fallback to pytube with headers."""
        try:
            from pytube import YouTube
            yt = YouTube(url, headers=self.headers)
            stream = yt.streams.filter(
                file_extension='mp4',
                progressive=True
            ).order_by('resolution').desc().first()
            
            if stream:
                output_path = os.path.join(self.download_dir, f"{video_id}.mp4")
                stream.download(
                    output_path=self.download_dir,
                    filename=f"{video_id}.mp4",
                    skip_existing=True
                )
                return output_path
        except Exception:
            return None

    def _download_with_direct_link(self, url: str, video_id: str) -> Optional[str]:
        """Experimental: Try to find direct download link."""
        try:
            # Get video page to find player script
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # This is fragile and may break - last resort only
            scripts = soup.find_all('script')
            for script in scripts:
                if 'ytInitialPlayerResponse' in str(script):
                    match = re.search(r'"url":"(https://[^"]+googlevideo[^"]+)"', str(script))
                    if match:
                        video_url = match.group(1).replace('\\u0026', '&')
                        response = requests.get(video_url, stream=True, headers=self.headers)
                        output_path = os.path.join(self.download_dir, f"{video_id}.mp4")
                        with open(output_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        return output_path
        except Exception:
            return None

    def _extract_frame(self, video_path: str, timestamp_ms: int) -> Optional[np.ndarray]:
        """Extract frame with precise timing."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        # Convert timestamp to frame number
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_pos = int((timestamp_ms / 1000) * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        cap.release()
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None

    def load_trajectory(self, split: str, video_id: str) -> Dict:
        """Load trajectory data from text file."""
        video_path = os.path.join(self.dataset_root, split, f"{video_id}.txt")
        
        trajectory = {
            'url': None,
            'timestamps': [],
            'intrinsics': [],
            'poses': []
        }
        
        with open(video_path, 'r') as f:
            trajectory['url'] = f.readline().strip()
            for line in f:
                parts = line.strip().split()
                if len(parts) != 19:
                    continue
                
                trajectory['timestamps'].append(int(parts[0]))
                trajectory['intrinsics'].append(tuple(map(float, parts[1:5])))
                trajectory['poses'].append(np.array(list(map(float, parts[5:17]))).reshape(3, 4))
        
        return trajectory

    def get_frame(self, trajectory: Dict, frame_idx: int, 
                 image_size: Optional[Tuple[int, int]] = None) -> Dict:
        """Get complete frame data with fallback handling."""
        if frame_idx >= len(trajectory['poses']):
            raise IndexError("Frame index out of range")
            
        # Try downloading with multiple methods
        print(trajectory["url"])
        video_path = self._download_video(trajectory['url'])
        print(video_path)
        if not video_path:
            warnings.warn(f"All download methods failed for {trajectory['url']}")
            return None
        
        # Extract frame
        timestamp_ms = trajectory['timestamps'][frame_idx] // 1000
        print(video_path)
        image = self._extract_frame(video_path, timestamp_ms)
        if image is None:
            warnings.warn(f"Frame extraction failed at {timestamp_ms}ms")
            return None
        
        # Process image
        if image_size:
            image = cv2.resize(image, image_size)
            width, height = image_size
        else:
            height, width = image.shape[:2]
        
        # Prepare intrinsics
        fx, fy, cx, cy = trajectory['intrinsics'][frame_idx]
        intrinsics = np.array([
            [width * fx, 0, width * cx],
            [0, height * fy, height * cy],
            [0, 0, 1]
        ])
        
        return {
            'image': image,
            'pose': trajectory['poses'][frame_idx],
            'intrinsics': intrinsics,
            'timestamp': trajectory['timestamps'][frame_idx],
            'video_path': video_path
        }

    def get_all_frames(self, split: str, video_id: str, 
                      image_size: Optional[Tuple[int, int]] = None,
                      max_workers: int = 4) -> List[Dict]:
        """Batch process frames with parallel downloading."""
        trajectory = self.load_trajectory(split, video_id)
        video_path = self._download_video(trajectory['url'])
        
        if not video_path:
            return []
            
        frames = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for frame_idx in range(len(trajectory['poses'])):
                futures.append(executor.submit(
                    self._process_single_frame,
                    video_path=video_path,
                    trajectory=trajectory,
                    frame_idx=frame_idx,
                    image_size=image_size
                ))
            
            for future in futures:
                try:
                    frame = future.result()
                    if frame:
                        frames.append(frame)
                except Exception as e:
                    warnings.warn(f"Frame processing failed: {str(e)}")
        
        return frames

    def _process_single_frame(self, video_path: str, trajectory: Dict,
                            frame_idx: int, image_size: Optional[Tuple[int, int]]) -> Optional[Dict]:
        """Helper method for parallel frame processing."""
        timestamp_ms = trajectory['timestamps'][frame_idx] // 1000
        image = self._extract_frame(video_path, timestamp_ms)
        if image is None:
            return None
            
        if image_size:
            image = cv2.resize(image, image_size)
            width, height = image_size
        else:
            height, width = image.shape[:2]
        
        fx, fy, cx, cy = trajectory['intrinsics'][frame_idx]
        intrinsics = np.array([
            [width * fx, 0, width * cx],
            [0, height * fy, height * cy],
            [0, 0, 1]
        ])
        
        return {
            'image': image,
            'pose': trajectory['poses'][frame_idx],
            'intrinsics': intrinsics,
            'timestamp': trajectory['timestamps'][frame_idx]
        }


if __name__ == "__main__":
    loader = RealEstate10KLoader(
        dataset_root='final_project/217-final-proj/data/RealEstate10K',
        download_dir='final_project/217-final-proj/data/RealEstate10K/cache'
    )

    frame_data = loader.get_frame(
        loader.load_trajectory('test', '0a3b5fb184936a83'),  # 0a3f289636ba10a7
        frame_idx=10,
        image_size=(1280, 720)
    )
    if frame_data is None:
        print("No frame data")
    else:
        # loader = RealEstate10KLoader("final_project/217-final-proj/data/RealEstate10K")
        # frame_data = loader.get_frame_data("test", "0a3b5fb184936a83", 0)

        print(frame_data["timestamp"])
        print(frame_data["pose"])
        print(frame_data["intrinsics"])
        cv2.imshow(frame_data["image"])
        cv2.waitKey(-1)
        cv2.destroyAllWindows()

