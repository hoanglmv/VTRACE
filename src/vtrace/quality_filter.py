import os
import shutil
import logging
import cv2

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def score_image(path, blur_threshold, exposure_min, exposure_max):
    img = cv2.imread(str(path))
    if img is None:
        return "unreadable"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    exposure = float(gray.mean())

    flags = []
    if blur < blur_threshold:
        flags.append("blur")
    if exposure < exposure_min:
        flags.append("dark")
    if exposure > exposure_max:
        flags.append("overexposed")

    if flags:
        logger.info(f"Image {path.name if hasattr(path, 'name') else os.path.basename(path)} flagged: {', '.join(flags)} (blur={blur:.2f}, exp={exposure:.2f})")
        return ",".join(flags)
    return "ok"

class QualityFilter:
    def __init__(self, image_dir, blur_threshold=100.0, exposure_min=20.0, exposure_max=235.0):
        self.image_dir = image_dir
        self.blur_threshold = blur_threshold
        self.exposure_min = exposure_min
        self.exposure_max = exposure_max
        self.flagged_dir = os.path.join(os.path.dirname(image_dir), "flagged_images")
        self.moved_files = []

    def apply(self):
        """Find bad quality images and move them out of the image directory."""
        if not os.path.exists(self.image_dir):
            return

        images = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) 
                  if os.path.splitext(f)[1] in IMAGE_EXTS]
        
        if not images:
            return

        logger.info(f"Scanning {len(images)} images for quality in {self.image_dir}...")
        
        flagged_count = 0
        for img_path in images:
            status = score_image(img_path, self.blur_threshold, self.exposure_min, self.exposure_max)
            if status != "ok":
                if not os.path.exists(self.flagged_dir):
                    os.makedirs(self.flagged_dir)
                
                filename = os.path.basename(img_path)
                dest = os.path.join(self.flagged_dir, filename)
                shutil.move(img_path, dest)
                self.moved_files.append((dest, img_path))
                flagged_count += 1

        if flagged_count > 0:
            logger.info(f"Moved {flagged_count} flagged images to {self.flagged_dir}")

    def restore(self):
        """Restore moved images back to their original location."""
        if not self.moved_files:
            return

        logger.info(f"Restoring {len(self.moved_files)} flagged images back to {self.image_dir}...")
        for src, dest in self.moved_files:
            if os.path.exists(src):
                shutil.move(src, dest)
        
        if os.path.exists(self.flagged_dir) and not os.listdir(self.flagged_dir):
            os.rmdir(self.flagged_dir)
            
        self.moved_files = []
        logger.info("Restore complete.")
