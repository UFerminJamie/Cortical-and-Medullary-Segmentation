"""Create synthetic TIFF pyramids, embeddings, predictions, and XLSX manifests.

This is an illustration of the data contract, not a biological simulation or
model evaluation dataset. No research data or model checkpoint is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import tifffile
import torch


def tissue_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    width, height = 3072, 2048
    image = Image.new("RGB", (width, height), (248, 246, 244))
    draw = ImageDraw.Draw(image)
    draw.ellipse((220, 180, 2840, 1880), fill=(228, 184, 207))
    for _ in range(2200):
        x, y = rng.integers(220, 2840), rng.integers(180, 1880)
        if ((x - 1530) / 1280) ** 2 + ((y - 1030) / 810) ** 2 > .93:
            continue
        rx, ry = rng.integers(12, 35), rng.integers(16, 48)
        if x > 1750:
            rx, ry = rng.integers(8, 16), rng.integers(35, 85)
        draw.ellipse((x-rx, y-ry, x+rx, y+ry), fill=(196, 130, 167), outline=(161, 100, 148), width=2)
        draw.ellipse((x-rx+5, y-ry+7, x+rx-5, y+ry-7), fill=(246, 229, 233))
        for angle in rng.uniform(0, 2*np.pi, 5):
            nx, ny = x+int(rx*np.cos(angle)), y+int(ry*np.sin(angle))
            draw.ellipse((nx-3, ny-4, nx+3, ny+4), fill=(100, 66, 134))
    for x, y in [(650,600),(1000,900),(750,1300),(1300,550),(1400,1450)]:
        draw.ellipse((x-65,y-65,x+65,y+65), fill=(245,224,232), outline=(176,114,156), width=4)
        for _ in range(100):
            dx, dy = rng.integers(-43,44,2)
            if dx*dx+dy*dy < 1800:
                draw.ellipse((x+dx-4,y+dy-4,x+dx+4,y+dy+4), fill=(137,80,135))
    return image


def generate(output: Path, runtime_root: Path):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Demo output must be empty; existing data will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    datasets, original = [], []
    for group, cohort in enumerate(("development", "validation")):
        root = output / f"KPMP_synthetic_{cohort}"
        (root / "features").mkdir(parents=True)
        (root / "outputs/predictions").mkdir(parents=True)
        rows, summaries = [], []
        for sample in range(2):
            name = f"DEMO_{group+1:02d}_{sample+1:02d}"
            image = tissue_image(42+group*2+sample)
            wsi = root / f"{name}.tiff"
            with tifffile.TiffWriter(wsi, bigtiff=True) as writer:
                writer.write(np.asarray(image), photometric="rgb", tile=(256,256), compression="deflate", subifds=3, metadata=None)
                for factor in (2,4,8):
                    level = image.resize((image.width//factor, image.height//factor), Image.Resampling.BILINEAR)
                    writer.write(np.asarray(level), photometric="rgb", tile=(256,256), compression="deflate", subfiletype=1, metadata=None)
            coords = np.array([(x,y) for y in range(128,1920,128) for x in range(128,2944,128)
                               if ((x+64-1530)/1280)**2+((y+64-1030)/810)**2 < .9], dtype=np.int64)
            rng = np.random.default_rng(100+group*2+sample)
            scores = ((coords[:,0]+64-1650)/500+rng.normal(0,.2,len(coords))).astype(np.float32)
            pt = root / "features" / f"{name}.pt"
            torch.save({"features":torch.from_numpy(rng.normal(size=(len(coords),16)).astype(np.float32)),
                        "coordinates":torch.from_numpy(coords), "tile_size":128}, pt)
            pd.DataFrame({"x":coords[:,0], "y":coords[:,1], "decision_score":scores,
                          "predicted_label_encoded":(scores>=0).astype(np.int8)}).to_csv(root/"outputs/predictions"/f"{name}.csv", index=False)
            runtime_wsi = runtime_root / wsi.relative_to(output)
            runtime_pt = runtime_root / pt.relative_to(output)
            rows.append({"sample_name":name+".tiff", "wsi_path":str(runtime_wsi),
                         "feature_path":str(runtime_pt), "case_id":f"SYNTHETIC-{group+1}-{sample+1}"})
            if group*2+sample < 3:
                original.append({**rows[-1],"split":("train","test","val")[group*2+sample]})
            summaries.append({"sample":name,"tiles":len(coords),"cortical":int((scores<0).sum()),"medullary":int((scores>=0).sum())})
        pd.DataFrame(rows).to_excel(root/"manifest.xlsx", index=False)
        (root/"outputs/inference_summary.json").write_text(json.dumps({"threshold":0,"samples":summaries}, indent=2)+"\n")
        datasets.append({"id":cohort,"name":f"Synthetic {cohort.title()}",
                         "manifest":f"{root.name}/manifest.xlsx","predictions":f"{root.name}/outputs"})
    pd.DataFrame(original).to_excel(output/"case_split.xlsx", index=False)
    (output/"datasets.json").write_text(json.dumps(datasets, indent=2)+"\n")
    (output/"SYNTHETIC.txt").write_text("Synthetic illustrations and random embeddings; no model inference or patient data.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, help="Paths written into XLSX, e.g. /data for containers")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    runtime_root = args.runtime_root or output
    if not runtime_root.is_absolute():
        parser.error("--runtime-root must be absolute")
    try:
        generate(output, runtime_root)
    except ValueError as error:
        parser.error(str(error))
    print(f"Created four synthetic slides and two datasets in {output}")


if __name__ == "__main__":
    main()
