import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import tiffslide


def bbox_intersects(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    return not (
        x1_max <= x2_min or
        x1_min >= x2_max or
        y1_max <= y2_min or
        y1_min >= y2_max
    )


def polygon_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def extract_polygons(annotation_json):
    """
    You may need to modify this depending on your JSON format.

    Expected output:
        [
            [(x1, y1), (x2, y2), ...],
            [(x1, y1), (x2, y2), ...]
        ]
    """

    polygons = []

    if annotation_json is None:
        return polygons

    # Common Girder / Histomics style
    if isinstance(annotation_json, dict):
        if "annotation" in annotation_json:
            annotation_json = annotation_json["annotation"]

        if "elements" in annotation_json:
            for element in annotation_json["elements"]:
                if "points" in element:
                    poly = [(p[0], p[1]) for p in element["points"]]
                    polygons.append(poly)

                elif "vertices" in element:
                    poly = [(p["x"], p["y"]) for p in element["vertices"]]
                    polygons.append(poly)

                elif "coordinates" in element:
                    poly = [(p[0], p[1]) for p in element["coordinates"]]
                    polygons.append(poly)

    # If JSON is directly a list of annotations
    elif isinstance(annotation_json, list):
        for item in annotation_json:
            if "points" in item:
                poly = [(p[0], p[1]) for p in item["points"]]
                polygons.append(poly)

            elif "coordinates" in item:
                poly = [(p[0], p[1]) for p in item["coordinates"]]
                polygons.append(poly)

    return polygons


class DynamicWSICortMedDataset(Dataset):
    def __init__(
        self,
        root_dir,
        tile_size=512,
        num_steps=100000,
        batch_size=12,
        transform=None,
        min_class_pixels=100,
        max_sampling_attempts=50,
    ):
        self.root_dir = Path(root_dir)
        self.tile_size = tile_size
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.transform = transform
        self.min_class_pixels = min_class_pixels
        self.max_sampling_attempts = max_sampling_attempts

        self.samples = []
        self.cortical_sample_ids = []
        self.medullary_sample_ids = []
        self.background_sample_ids = []

        self._slide_cache = {}

        self._scan_samples()

    def __len__(self):
        return self.num_steps * self.batch_size

    def _scan_samples(self):
        slide_exts = [".svs", ".tif", ".tiff", ".SVS", ".TIF", ".TIFF"]

        for cohort_dir in sorted(self.root_dir.iterdir()):
            if not cohort_dir.is_dir():
                continue

            cohort_name = cohort_dir.name

            for sample_dir in sorted(cohort_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue

                slide_path = None
                for ext in slide_exts:
                    matches = list(sample_dir.glob(f"*{ext}"))
                    if matches:
                        slide_path = matches[0]
                        break

                if slide_path is None:
                    continue

                cortical_json_path = self._find_json(sample_dir, "cortical")
                medullary_json_path = self._find_json(sample_dir, "medullary")

                if cortical_json_path is None and medullary_json_path is None:
                    continue

                cortical_polygons = self._load_polygons(cortical_json_path)
                medullary_polygons = self._load_polygons(medullary_json_path)

                if len(cortical_polygons) == 0 and len(medullary_polygons) == 0:
                    continue

                slide = tiffslide.TiffSlide(str(slide_path))
                width, height = slide.dimensions
                slide.close()

                cortical_bboxes = [polygon_bbox(p) for p in cortical_polygons]
                medullary_bboxes = [polygon_bbox(p) for p in medullary_polygons]

                sample = {
                    "cohort": cohort_name,
                    "sample_name": sample_dir.name,
                    "slide_path": str(slide_path),
                    "width": width,
                    "height": height,
                    "cortical_polygons": cortical_polygons,
                    "medullary_polygons": medullary_polygons,
                    "cortical_bboxes": cortical_bboxes,
                    "medullary_bboxes": medullary_bboxes,
                }

                sample_id = len(self.samples)
                self.samples.append(sample)

                if len(cortical_polygons) > 0:
                    self.cortical_sample_ids.append(sample_id)

                if len(medullary_polygons) > 0:
                    self.medullary_sample_ids.append(sample_id)

                # Any annotated slide can be used to sample background tiles
                self.background_sample_ids.append(sample_id)

        if len(self.cortical_sample_ids) == 0:
            raise RuntimeError("No cortical annotations found.")

        if len(self.medullary_sample_ids) == 0:
            raise RuntimeError("No medullary annotations found.")

        if len(self.background_sample_ids) == 0:
            raise RuntimeError("No valid samples found.")

    def _find_json(self, sample_dir, keyword):
        for json_path in sample_dir.glob("*.json"):
            if keyword.lower() in json_path.name.lower():
                return json_path
        return None

    def _load_polygons(self, json_path):
        if json_path is None:
            return []

        with open(json_path, "r") as f:
            data = json.load(f)

        return extract_polygons(data)

    def _get_slide(self, sample):
        slide_path = sample["slide_path"]

        if slide_path not in self._slide_cache:
            self._slide_cache[slide_path] = tiffslide.TiffSlide(slide_path)

        return self._slide_cache[slide_path]

    def __getitem__(self, idx):
        target_class = self._get_target_class(idx)

        for _ in range(self.max_sampling_attempts):
            sample_id = self._choose_sample_id(target_class)
            sample = self.samples[sample_id]

            x, y = self._sample_random_coordinate(sample, target_class)

            image = self._read_tile(sample, x, y)
            mask = self._make_tile_mask(sample, x, y)

            if self._is_valid_tile(mask, target_class):
                break

        image = np.array(image)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
            "target_class": target_class,
            "cohort": sample["cohort"],
            "sample_name": sample["sample_name"],
            "slide_path": sample["slide_path"],
            "x": x,
            "y": y,
        }

    def _get_target_class(self, idx):
        """
        Forces balanced sampling.

        0 = background
        1 = cortical
        2 = medullary
        """
        return idx % 3

    def _choose_sample_id(self, target_class):
        if target_class == 1:
            return random.choice(self.cortical_sample_ids)

        elif target_class == 2:
            return random.choice(self.medullary_sample_ids)

        else:
            return random.choice(self.background_sample_ids)

    def _sample_random_coordinate(self, sample, target_class):
        if target_class == 1:
            bbox = random.choice(sample["cortical_bboxes"])
            return self._sample_coordinate_near_bbox(sample, bbox)

        elif target_class == 2:
            bbox = random.choice(sample["medullary_bboxes"])
            return self._sample_coordinate_near_bbox(sample, bbox)

        else:
            return self._sample_background_coordinate(sample)

    def _sample_coordinate_near_bbox(self, sample, bbox):
        xmin, ymin, xmax, ymax = bbox

        cx = random.randint(int(xmin), int(xmax))
        cy = random.randint(int(ymin), int(ymax))

        # Randomly place the selected point somewhere inside the tile
        offset_x = random.randint(0, self.tile_size - 1)
        offset_y = random.randint(0, self.tile_size - 1)

        x = cx - offset_x
        y = cy - offset_y

        x = self._clamp(x, 0, max(0, sample["width"] - self.tile_size))
        y = self._clamp(y, 0, max(0, sample["height"] - self.tile_size))

        return x, y

    def _sample_background_coordinate(self, sample):
        """
        Background tile is sampled near annotations, then rejected
        if it contains cortical or medullary pixels.

        This avoids mostly blank white WSI background.
        """

        all_bboxes = sample["cortical_bboxes"] + sample["medullary_bboxes"]

        if len(all_bboxes) > 0:
            bbox = random.choice(all_bboxes)

            xmin, ymin, xmax, ymax = bbox

            pad = self.tile_size * 2

            xmin = max(0, xmin - pad)
            ymin = max(0, ymin - pad)
            xmax = min(sample["width"], xmax + pad)
            ymax = min(sample["height"], ymax + pad)

            cx = random.randint(int(xmin), int(xmax))
            cy = random.randint(int(ymin), int(ymax))

            x = cx - random.randint(0, self.tile_size - 1)
            y = cy - random.randint(0, self.tile_size - 1)

        else:
            x = random.randint(0, max(0, sample["width"] - self.tile_size))
            y = random.randint(0, max(0, sample["height"] - self.tile_size))

        x = self._clamp(x, 0, max(0, sample["width"] - self.tile_size))
        y = self._clamp(y, 0, max(0, sample["height"] - self.tile_size))

        return x, y

    def _read_tile(self, sample, x, y):
        slide = self._get_slide(sample)

        image = slide.read_region(
            location=(int(x), int(y)),
            level=0,
            size=(self.tile_size, self.tile_size)
        ).convert("RGB")

        return image

    def _make_tile_mask(self, sample, x, y):
        mask = Image.new("L", (self.tile_size, self.tile_size), 0)
        draw = ImageDraw.Draw(mask)

        tile_bbox = (
            x,
            y,
            x + self.tile_size,
            y + self.tile_size
        )

        # Draw cortical first
        for poly, bbox in zip(sample["cortical_polygons"], sample["cortical_bboxes"]):
            if bbox_intersects(tile_bbox, bbox):
                local_poly = [
                    (px - x, py - y)
                    for px, py in poly
                ]
                draw.polygon(local_poly, fill=1)

        # Draw medullary second so it overwrites cortical if overlapping
        for poly, bbox in zip(sample["medullary_polygons"], sample["medullary_bboxes"]):
            if bbox_intersects(tile_bbox, bbox):
                local_poly = [
                    (px - x, py - y)
                    for px, py in poly
                ]
                draw.polygon(local_poly, fill=2)

        return np.array(mask, dtype=np.uint8)

    def _is_valid_tile(self, mask, target_class):
        if target_class == 0:
            return np.sum(mask > 0) == 0

        elif target_class == 1:
            return np.sum(mask == 1) >= self.min_class_pixels

        elif target_class == 2:
            return np.sum(mask == 2) >= self.min_class_pixels

        return False

    def _clamp(self, value, min_value, max_value):
        return int(max(min_value, min(value, max_value)))
    
import albumentations as A
from albumentations.pytorch import ToTensorV2


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(),
    ToTensorV2()
])


val_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])