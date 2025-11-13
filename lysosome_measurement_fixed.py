import os
import re
import numpy as np
import pandas as pd
import cv2
import imageio
import xml.etree.ElementTree as ET

import czifile
import tifffile as tiff

from skimage.feature import blob_log
from skimage.filters import gaussian, threshold_local
from skimage.morphology import (
    remove_small_objects, binary_opening, binary_closing, ball, binary_erosion
)
from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage import rotate as ndrotate
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, find_boundaries
from scipy.ndimage import binary_fill_holes

import napari

# ===============================
# USER CONFIGURATION
# ===============================
# Path to your CZI or TIFF/OME-TIFF:
file_path = r"/mnt/data/40A_UAS-TMEM1923x-HA x 71G10 40A MARCM_L3_2_Airy_010724.tif"

# If XY pixel size is missing in metadata, use this fallback (µm/px).
# >>>>>>>>>>>> SET THIS TO YOUR TRUE CALIBRATION <<<<<<<<<<<<
DEFAULT_VX_VY_UM = 0.055  # e.g., 55 nm/px for Airyscan @ 63x (example)

# Sanity limit to prevent silently wrong scales
MAX_REASONABLE_VXY_UM = 0.5  # µm/px; typical high-NA confocal/airyscan is << 0.5

# ---------------------------
# Helper: refine radii by DT
# ---------------------------
def refine_radii_via_dt(img3d, blobs, win_px=25, bin_method="sauvola"):
    """
    Refine per-blob radius using a local 2D binary mask + distance transform on the Z slice.
    - img3d: float32 (Z, Y, X), same you used for LoG (e.g., smoothed)
    - blobs: ndarray [N, 4] with columns [z, y, x, r_px] (r is initial estimate; overwritten)
    - win_px: half-window around (y,x) to build local mask
    - bin_method: 'sauvola' | 'local' | 'otsu'
    Returns: blobs_refined with same centers [z,y,x] and r_px from DT (continuous).
    """
    from skimage.filters import threshold_sauvola, threshold_local, threshold_otsu
    from skimage.morphology import remove_small_objects, disk, binary_opening
    from scipy.ndimage import distance_transform_edt as _edt
    from skimage.measure import label as _label

    if blobs is None or len(blobs) == 0:
        return blobs

    Z, H, W = img3d.shape
    out = blobs.copy().astype(np.float32)

    for i, (zc, yc, xc, _) in enumerate(out):
        z = int(round(float(zc)))
        y = int(round(float(yc)))
        x = int(round(float(xc)))

        if not (0 <= z < Z and 0 <= y < H and 0 <= x < W):
            continue

        y1, y2 = max(0, y - win_px), min(H, y + win_px + 1)
        x1, x2 = max(0, x - win_px), min(W, x + win_px + 1)
        if (y2 <= y1) or (x2 <= x1):
            continue

        patch = img3d[z, y1:y2, x1:x2]

        # Adaptive/local binarization
        if bin_method == "sauvola":
            ws = max(21, 2*(win_px//2)+1)
            thr = threshold_sauvola(patch, window_size=ws, k=0.2)
            bw = patch > thr
        elif bin_method == "local":
            ws = max(21, 2*(win_px//2)+1)
            thr = threshold_local(patch, block_size=ws, offset=-0.2*np.std(patch))
            bw = patch > thr
        else:  # 'otsu'
            try:
                thr = threshold_otsu(patch)
                bw = patch > thr
            except ValueError:
                continue

        # light clean-up
        bw = binary_opening(bw, footprint=disk(1))
        bw = remove_small_objects(bw, min_size=3, connectivity=2)

        # Keep only the component that contains (or is nearest to) the center
        yy, xx = y - y1, x - x1
        if not (0 <= yy < bw.shape[0] and 0 <= xx < bw.shape[1]) or not bw[yy, xx]:
            found = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yyy, xxx = yy + dy, xx + dx
                    if 0 <= yyy < bw.shape[0] and 0 <= xxx < bw.shape[1] and bw[yyy, xxx]:
                        yy, xx = yyy, xxx
                        found = True
                        break
                if found:
                    break
            if not found:
                continue

        lab = _label(bw, connectivity=2)
        lbl = lab[yy, xx]
        if lbl == 0:
            continue
        bw_obj = (lab == lbl)

        # DT-based radius (continuous, in pixels on this slice)
        dt = _edt(bw_obj)
        r_px = float(dt[yy, xx])
        if r_px <= 0:
            continue

        out[i, 3] = r_px  # overwrite radius in pixels (continuous)

    return out

# ==========================================
# Generic loader for CZI and TIFF (OME/ImageJ)
# Returns: ch1, ch2, (vx_um, vy_um, vz_um), meta
# ==========================================
def _try_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _parse_ome_xml(xml_text: str):
    """Parse OME-XML PhysicalSizeX/Y/Z → µm (graceful fallback)."""
    if not xml_text:
        return None
    def grab(attr):
        m = re.search(fr'PhysicalSize{attr}="([\d\.eE+-]+)"(?:\s+PhysicalSize{attr}Unit="([^"]+)")?', xml_text)
        if not m:
            return None, None
        return m.group(1), (m.group(2) or "")

    x_val, x_unit = grab('X')
    y_val, y_unit = grab('Y')
    z_val, z_unit = grab('Z')

    def to_um(val_str, unit):
        v = _try_float(val_str)
        if v is None:
            return None
        u = (unit or "").lower()
        if u in ("µm", "um", "micrometer", "micrometre", "microns", "micron"):
            return v
        if u in ("m", "meter", "metre"):
            return v * 1e6
        # If unit missing: treat tiny as meters, else µm
        return v * 1e6 if v < 1e-3 else v

    vx = to_um(x_val, x_unit) if x_val is not None else None
    vy = to_um(y_val, y_unit) if y_val is not None else None
    vz = to_um(z_val, z_unit) if z_val is not None else None
    return vx, vy, vz

def _voxel_from_tiff_tags(tf: tiff.TiffFile):
    """
    Try ImageJ-style X/Y resolution (inch/cm → µm/px) and Z spacing from ImageDescription.
    Returns (vx_um, vy_um, vz_um) with None if not found.
    """
    vx = vy = vz = None
    try:
        page0 = tf.pages[0]
        xres = getattr(page0, "tags", {}).get("XResolution", None)
        yres = getattr(page0, "tags", {}).get("YResolution", None)
        resunit = getattr(page0, "tags", {}).get("ResolutionUnit", None)

        def res_to_um(res_tag, unit_tag):
            if res_tag is None:
                return None
            val = res_tag.value
            if isinstance(val, tuple) and len(val) == 2:
                num, den = val
            else:
                try:
                    num, den = val.numerator, val.denominator
                except Exception:
                    return None
            if den == 0:
                return None
            ppu = num / den  # pixels per unit
            if ppu <= 0:
                return None
            unit = (unit_tag.value if unit_tag else 2)  # 2=inches, 3=cm
            if unit == 2:
                um_per_unit = 25400.0
            elif unit == 3:
                um_per_unit = 10000.0
            else:
                return None
            return um_per_unit / ppu  # µm per pixel

        vx = res_to_um(xres, resunit)
        vy = res_to_um(yres, resunit)

        # Try ImageDescription for Z spacing (often in µm)
        desc = page0.tags.get("ImageDescription", None)
        if desc is not None:
            txt = str(desc.value)
            m = re.search(r'(spacing|SliceSpacing)[=:]\s*([0-9.+-eE]+)', txt)
            if m:
                vz = _try_float(m.group(2))
    except Exception:
        pass
    return vx, vy, vz

def load_any(file_path):
    """
    Load a 2-channel 3D stack from CZI or TIFF/OME-TIFF.
    Returns: ch1, ch2, (vx_um, vy_um, vz_um), meta
    NOTE: vx_um/vy_um/vz_um may be None if not found in metadata.
    """
    ext = os.path.splitext(file_path)[1].lower()
    vx_um = vy_um = vz_um = None

    if ext == ".czi":
        # -------- CZI --------
        with czifile.CziFile(file_path) as czi:
            img = czi.asarray()
            meta_xml = czi.metadata()
        img = np.squeeze(img)

        # Infer channels
        if img.ndim < 3:
            raise RuntimeError(f"CZI has unexpected ndim={img.ndim}")
        if img.shape[0] == 2:        # (C, Z, Y, X)
            ch1, ch2 = img[0], img[1]
        elif img.shape[1] == 2:      # (Z, C, Y, X)
            ch1, ch2 = img[:, 0], img[:, 1]
        else:
            raise RuntimeError(f"Can't auto-detect 2 channels in CZI shape {img.shape}")

        # Voxel size from CZI XML
        try:
            r = ET.fromstring(meta_xml)
            def _get_um(axis: str) -> float:
                v = r.find(f".//{{*}}Scaling/{{*}}Items/{{*}}Distance[@Id='{axis}']/{{*}}Value")
                u = r.find(f".//{{*}}Scaling/{{*}}Items/{{*}}Distance[@Id='{axis}']/{{*}}DefaultUnit")
                if v is None:
                    return None
                val = _try_float(v.text)
                unit = (u.text or "").lower() if u is not None else ""
                if val is None:
                    return None
                if unit in ("m", "meter", "metre") or (unit == "" and val < 1e-3):
                    val *= 1e6
                elif unit in ("µm", "um", "micrometer", "micrometre"):
                    pass
                else:
                    if val < 1e-3:
                        val *= 1e6
                return val
            vx_um, vy_um, vz_um = _get_um("X"), _get_um("Y"), _get_um("Z")
        except Exception:
            pass
        return ch1, ch2, (vx_um, vy_um, vz_um), {"type": "czi"}

    elif ext in (".tif", ".tiff"):
        # -------- TIFF / OME-TIFF --------
        with tiff.TiffFile(file_path) as tf:
            arr = tf.asarray()
            ome_xml = None
            try:
                ome_xml = tf.ome_metadata
            except Exception:
                pass

            if ome_xml:
                vx_um, vy_um, vz_um = _parse_ome_xml(ome_xml) or (None, None, None)
            if vx_um is None or vy_um is None or vz_um is None:
                tx, ty, tz = _voxel_from_tiff_tags(tf)
                vx_um = vx_um if vx_um is not None else tx
                vy_um = vy_um if vy_um is not None else ty
                vz_um = vz_um if vz_um is not None else tz

        img = np.squeeze(arr)

        # Layouts we support:
        if img.ndim == 4:
            if img.shape[0] == 2:        # (C, Z, Y, X)
                ch1, ch2 = img[0], img[1]
            elif img.shape[1] == 2:      # (Z, C, Y, X)
                ch1, ch2 = img[:, 0], img[:, 1]
            elif img.shape[-1] == 2:     # (Z, Y, X, C)
                ch1, ch2 = img[..., 0], img[..., 1]
            else:
                raise RuntimeError(f"Cannot infer 2 channels from TIFF shape {img.shape}")
        elif img.ndim == 3:
            if img.shape[-1] == 2:       # (Y, X, C) → promote Z=1
                ch1 = img[..., 0][None, ...]
                ch2 = img[..., 1][None, ...]
            else:
                raise RuntimeError("TIFF is single-channel; expected 2 channels.")
        else:
            raise RuntimeError(f"Unsupported TIFF ndim: {img.ndim}")

        return ch1, ch2, (vx_um, vy_um, vz_um), {"type": "tiff", "ome": bool(ome_xml)}

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

# ==========================================
# 1) Load image (CZI or TIFF) + voxel size (µm)
# ==========================================
img_ch1, img_ch2, (vx_um, vy_um, vz_um), meta = load_any(file_path)

# Choose which channel is lysosomes vs neuron channel
image   = img_ch1      # Ch1: lysosome channel
image_2 = img_ch2      # Ch2: neuron (CELL vs OUTSIDE)

# ===============================
# XY scale enforcement & sanity
# ===============================
if vx_um is None or vy_um is None:
    print("[WARN] Missing XY pixel size in metadata. Using DEFAULT_VX_VY_UM =", DEFAULT_VX_VY_UM, "µm/px")
    vx_um = DEFAULT_VX_VY_UM
    vy_um = DEFAULT_VX_VY_UM

if (vx_um is None) or (vy_um is None):
    raise RuntimeError("XY pixel size is undefined. Set DEFAULT_VX_VY_UM to your microscope calibration.")

if (vx_um > MAX_REASONABLE_VXY_UM) or (vy_um > MAX_REASONABLE_VXY_UM):
    raise ValueError(f"XY pixel size looks too large (vx={vx_um}, vy={vy_um} µm/px). Check calibration.")

# If Z spacing missing, fall back to XY scale (least-bad assumption) or 1.0 with warning
if vz_um is None:
    vz_um = vx_um
    print("[WARN] Missing Z spacing; assuming isotropic voxels with vz_um =", vz_um, "µm")

# Per-voxel metrics
voxel_um3    = vz_um * vy_um * vx_um                 # µm^3 per voxel
lin_equiv_um = voxel_um3 ** (1.0 / 3.0)              # linear scale preserving volume
px_um_xy     = float(np.sqrt(vx_um * vy_um))         # effective XY µm/px for radii

print(f"[{meta.get('type','?').upper()}] Voxel (µm): X={vx_um:.4g}, Y={vy_um:.4g}, Z={vz_um:.4g} | XY µm/px={px_um_xy:.4g}")
print("Ch1 shape:", image.shape, "| Ch2 shape:", image_2.shape)

# ==========================================
# 2) CH1: lysosome centers (LoG) + DT-refined radii (continuous)
# ==========================================
image_smooth = gaussian(image, sigma=1.5)

blobs = blob_log(
    image_smooth,
    min_sigma=0.8,
    max_sigma=3.0,
    num_sigma=4,
    threshold=0.004,
    overlap=1
)
if len(blobs) > 0:
    blobs[:, 3] = blobs[:, 3] * np.sqrt(3)
print(f"Detected {len(blobs)} lysosomes (LoG centers).")

image_smooth = image_smooth.astype(np.float32, copy=False)
blobs = refine_radii_via_dt(image_smooth, blobs, win_px=25, bin_method="sauvola")

# Convert per-lysosome metrics to physical units (µm)
if len(blobs) > 0:
    z_um = blobs[:, 0] * vz_um
    y_um = blobs[:, 1] * vy_um
    x_um = blobs[:, 2] * vx_um

    # Continuous radius from DT (in XY pixels) → microns
    radius_px   = blobs[:, 3]
    radius_um   = radius_px * px_um_xy
    diameter_um = 2.0 * radius_um
    volume_um3  = (4.0/3.0) * np.pi * (radius_um ** 3)

    # Sanity checks
    if (np.median(radius_um) > 1.0) or (np.max(radius_um) > 2.0):
        print("[WARN] Radii in µm look large (median or max > 1–2 µm). Re-check XY calibration.")

    # Pixel-quantization note (repeated radii)
    vals, counts = np.unique(np.round(radius_px, 2), return_counts=True)
    repeats = vals[counts > 3]
    if repeats.size > 0:
        print(f"[NOTE] Identical radii in px at {repeats[:10]}... likely pixel/threshold quantization.")

else:
    z_um = y_um = x_um = np.array([])
    radius_um = diameter_um = volume_um3 = np.array([])

# Per-blob DF (µm only)
blob_ids = np.arange(1, len(blobs) + 1) if blobs is not None else np.array([], dtype=int)
df = pd.DataFrame({
    "id": blob_ids,
    "z_um": z_um,
    "y_um": y_um,
    "x_um": x_um,
    "diameter_um": diameter_um,
    "radius_um": radius_um,
    "volume_um3": volume_um3,
})
df.to_csv("lysosome_blobs_regions.csv", index=False)
print("Saved: lysosome_blobs_regions.csv (µm-only, radii refined by DT)")

# ==========================================
# 3) Viewer base
# ==========================================
viewer = napari.Viewer()
viewer.add_image(image_2, name='Ch2 raw', blending='additive')
viewer.add_image(image,  name='Ch1 raw', blending='additive')

# ==========================================
# 4) CH2: segmentation (CELL vs OUTSIDE)
# ==========================================
vol = image_2.astype(np.float32)
vmin, vmax = float(vol.min()), float(vol.max())
if vmax > vmin:
    vol = (vol - vmin) / (vmax - vmin)
else:
    vol[:] = 0.0
ch2 = gaussian(vol, sigma=1.5, preserve_range=True)

# Local threshold per z
neuron_mask = np.zeros_like(ch2, dtype=bool)
for z in range(ch2.shape[0]):
    R = ch2[z]
    t = threshold_local(R, block_size=161, offset=-0.4*np.std(R))
    neuron_mask[z] = R > t
neuron_mask = remove_small_objects(neuron_mask, min_size=20000, connectivity=3)
neuron_mask = binary_closing(neuron_mask, ball(10))
neuron_mask = binary_erosion(neuron_mask, ball(2))

# Soma via distance + cleanup
dist = edt(neuron_mask)
cell_min_radius_vox = 1
cell_mask = (dist >= cell_min_radius_vox)
cell_mask &= neuron_mask
cell_mask = binary_opening(cell_mask, ball(5))
cell_mask = binary_closing(cell_mask, ball(10))
cell_mask = binary_fill_holes(cell_mask)

# Label and territories
body_lab = label(cell_mask, connectivity=3)
n_cells = int(body_lab.max())
print(f"Detected {n_cells} cells (soma).")

if n_cells > 0:
    dist_inside = edt(neuron_mask)
    cell_seg = watershed(-dist_inside, markers=body_lab, mask=neuron_mask)
else:
    cell_seg = np.zeros_like(neuron_mask, dtype=np.int32)

print("neuron voxels:", int(neuron_mask.sum()))
print("cell voxels:", int(cell_mask.sum()))

# ==========================================
# ADD-ON: optional filter out oversized cell bodies
# ==========================================
#MAX_BODY_VOXELS = 20000  # set >0 to enable filtering
try:
    if isinstance(cell_seg, np.ndarray) and cell_seg.max() > 0 and MAX_BODY_VOXELS > 0:
        counts = np.bincount(cell_seg.ravel().astype(np.int64))
        drop_labels = np.where(counts < 60000)[0]
        #drop_labels = np.where((counts < 70000) | (counts > 140000))[0]
        drop_labels = drop_labels[drop_labels != 0]

        if drop_labels.size > 0:
            print(f"Filtered {len(drop_labels)} oversized cells. IDs: {drop_labels.tolist()}")
            to_remove = np.isin(cell_seg, drop_labels)
            cell_seg[to_remove] = 0
            if 'cell_mask' in globals() and isinstance(cell_mask, np.ndarray):
                cell_mask[to_remove] = False

            pd.DataFrame({
                "filtered_cell_id_ch2": drop_labels,
                "voxel_count": counts[drop_labels]
            }).to_csv("filtered_cells_gt_threshold.csv", index=False)
        else:
            print(f"No cells exceeded {MAX_BODY_VOXELS} voxels; nothing filtered.")
except Exception as e:
    print("Cell size filtering failed:", e)

# ==========================================
# 5) Map lysosomes to (cell/outside) with per-cell IDs
# ==========================================
location_ch2 = []
cell_id_list = []

if len(blobs) > 0:
    Z, Y, X = neuron_mask.shape
    for zc, yc, xc in blobs[:, :3]:
        zz, yy, xx = int(round(zc)), int(round(yc)), int(round(xc))
        if not (0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X):
            location_ch2.append("outside"); cell_id_list.append(0); continue
        if cell_mask[zz, yy, xx]:
            location_ch2.append("cell")
            cid = int(cell_seg[zz, yy, xx]) if n_cells > 0 else 0
            cell_id_list.append(cid)
        else:
            location_ch2.append("outside")
            cell_id_list.append(0)

if len(df) > 0:
    df["location_ch2"] = location_ch2
    df["cell_id_ch2"]  = cell_id_list

    df.groupby("location_ch2").size().reset_index(name="count") \
      .to_csv("lysosome_counts_cell_vs_outside.csv", index=False)

    (df[df["location_ch2"] == "cell"]
        .groupby("cell_id_ch2").size()
        .reset_index(name="count")
        .to_csv("lysosome_counts_by_cell.csv", index=False))

    df.to_csv("lysosomes_with_cell_vs_outside.csv", index=False)

print("Saved: lysosome_counts_cell_vs_outside.csv, lysosome_counts_by_cell.csv, lysosomes_with_cell_vs_outside.csv (µm-only metrics)")

# Include zero-count cells
try:
    if isinstance(cell_seg, np.ndarray) and cell_seg.max() > 0:
        all_cells = pd.DataFrame({
            "cell_id_ch2": np.arange(1, int(cell_seg.max()) + 1, dtype=int)
        })

        if len(df) > 0 and "location_ch2" in df and "cell_id_ch2" in df:
            lys_counts_nonzero = (
                df[df["location_ch2"] == "cell"]
                .groupby("cell_id_ch2")
                .size()
                .reset_index(name="count")
            )
        else:
            lys_counts_nonzero = pd.DataFrame(columns=["cell_id_ch2", "count"])

        lys_counts_all = (
            all_cells.merge(lys_counts_nonzero, on="cell_id_ch2", how="left")
                     .fillna({"count": 0})
        )
        lys_counts_all["count"] = lys_counts_all["count"].astype(int)

        lys_counts_all.to_csv("lysosome_counts_by_cell.csv", index=False)
        print("Updated: lysosome_counts_by_cell.csv now includes cells with 0 lysosomes.")
except Exception as e:
    print("Could not expand lysosome_counts_by_cell with zero-count cells:", e)

# ==========================================
# 5b) Per-cell (Ch2) volumes (µm^3)
# ==========================================
cell_volume_df = pd.DataFrame(columns=["cell_id_ch2", "voxel_count", "volume_um3"])
if isinstance(cell_seg, np.ndarray) and cell_seg.max() > 0:
    counts = np.bincount(cell_seg.ravel().astype(np.int64))
    cell_ids = np.arange(1, counts.size, dtype=int)
    voxels = counts[1:].astype(np.int64)
    vol_um3 = voxels.astype(float) * voxel_um3

    cell_volume_df = pd.DataFrame({
        "cell_id_ch2": cell_ids,
        "voxel_count": voxels,
        "volume_um3": vol_um3
    })
    cell_volume_df.to_csv("cell_volumes_ch2.csv", index=False)
    print("Saved: cell_volumes_ch2.csv")

    # Merge volume with lysosome counts per cell
    try:
        if len(df) > 0 and "location_ch2" in df and "cell_id_ch2" in df:
            lys_counts = (df[df["location_ch2"] == "cell"]
                          .groupby("cell_id_ch2")
                          .size()
                          .reset_index(name="lysosome_count"))
            merged = (cell_volume_df
                      .merge(lys_counts, on="cell_id_ch2", how="left")
                      .fillna({"lysosome_count": 0}))
            merged.to_csv("cell_metrics_ch2.csv", index=False)
            print("Saved: cell_metrics_ch2.csv")
    except Exception as e:
        print("Merge with lysosome counts failed:", e)

# ==========================================
# 6) Visualization (CELL vs OUTSIDE only)
# ==========================================
cell_layer = viewer.add_labels(cell_mask.astype(np.uint8), name='Cell (Ch2)', opacity=0.35)
try:
    cell_layer.color = {1: (0.0, 1.0, 0.0, 1.0)}  # green
except Exception:
    pass
cell_layer.blending = 'translucent_no_depth'

try:
    cellid_layer = viewer.add_labels(
        cell_seg.astype(np.uint16),
        name='Cell ID (Ch2)',
        opacity=0.25
    )
    cellid_layer.blending = 'translucent_no_depth'

    boundaries = find_boundaries(cell_seg, connectivity=1, mode='outer')
    viewer.add_image(
        boundaries.astype(np.uint8),
        name='Cell ID boundaries',
        blending='additive',
        contrast_limits=(0, 1),
        colormap='magenta',
        opacity=0.6
    )
except Exception:
    pass

# Lysosome points overlay (show ONLY lysosomes inside cells)
if len(blobs) > 0 and "location_ch2" in df and "cell_id_ch2" in df:
    in_cell_mask = (df["location_ch2"].to_numpy() == "cell")
    if in_cell_mask.any():
        blobs_cell = blobs[in_cell_mask, :3]            # z,y,x
        radii_cell = blobs[in_cell_mask, 3] if blobs.shape[1] > 3 else np.ones(np.count_nonzero(in_cell_mask))
        pts = viewer.add_points(
            blobs_cell,
            size=np.clip(radii_cell * 2, 2, None),
            name='Lysosomes (cell only)'
        )
        try:
            pts.face_color = [0.0, 1.0, 1.0, 1.0]       # cyan
            pts.edge_color = 'black'
            pts.edge_width = 0.3
            pts.properties = {
                'lys_id': df.loc[in_cell_mask, 'id'].to_numpy(),
                'cell':   df.loc[in_cell_mask, 'cell_id_ch2'].to_numpy(),
                'diameter_um': df.loc[in_cell_mask, 'diameter_um'].to_numpy(),
                'volume_um3':  df.loc[in_cell_mask, 'volume_um3'].to_numpy()
            }
            pts.text = {'text': 'ID:{lys_id}  C:{cell}', 'size': 10, 'color': 'yellow', 'anchor': 'upper left'}
        except Exception:
            pass

# ==========================================
# 7) Quick fused 2D video (optional)
# ==========================================

img_norm_2 = (ch2 * 255).astype(np.uint8)
frames_fused = []
Z = img_norm_2.shape[0]
for z in range(Z):
    base = cv2.cvtColor(img_norm_2[z], cv2.COLOR_GRAY2BGR)
    cell = (cell_mask[z].astype(np.uint8) * 255)

    overlay = base.copy()
    overlay[..., 1] = np.maximum(overlay[..., 1], cell)  # green for cell
    overlay = cv2.addWeighted(base, 1.0, overlay, 0.35, 0.0)
    
    if len(blobs) > 0:
        z_blobs = blobs[np.abs(blobs[:, 0] - z) < 0.5]
        # iterate only over blobs that are inside the cell on this slice
        for b in z_blobs:
            y, x = int(round(b[1])), int(round(b[2]))
            r = max(2, int(round(b[3]))
)
            if 0 <= y < cell_mask.shape[1] and 0 <= x < cell_mask.shape[2] and cell_mask[z, y, x]:
                # draw only in-cell lysosomes (cyan/yellow are both OK; keeping yellow for visibility)
                cv2.circle(overlay, (x, y), r, (255, 255, 0), 2)
        # NOTE: no drawing at all for outside points
    
    frames_fused.append(overlay)

try:
    imageio.mimsave('ch2_fused_cell.mp4', frames_fused, fps=8, format='FFMPEG')
    print("Saved: ch2_fused_cell.mp4")
except TypeError:
    imageio.mimsave('ch2_fused_cell.gif', frames_fused, fps=8)
    print("Saved: ch2_fused_cell.gif")

# ==========================================
# 8) Per-cell rotation videos (X, Y, Z axes) – only for cells with lysosomes
# ==========================================

from scipy.ndimage import rotate as ndrotate

def _to_uint8(vol):
    v = vol.astype(np.float32)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin:
        return np.zeros_like(v, dtype=np.uint8)
    return (255.0 * (v - vmin) / (vmax - vmin)).astype(np.uint8)

def _rotate_points(pts_zyx, center_zyx, angle_deg, axis="Y"):
    #Rotate 3D points around chosen axis (X, Y, or Z).
    if pts_zyx.size == 0:
        return pts_zyx
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    p = pts_zyx - center_zyx
    if axis.upper() == "Y":     # rotate in ZX plane
        z = p[:, 0]*c + p[:, 2]*s
        y = p[:, 1]
        x = -p[:, 0]*s + p[:, 2]*c
    elif axis.upper() == "X":   # rotate in YZ plane
        z = p[:, 0]
        y = p[:, 1]*c - p[:, 2]*s
        x = p[:, 1]*s + p[:, 2]*c
    else:                       # Z axis (rotate in XY plane)
        z = p[:, 0]
        y = p[:, 1]*c + p[:, 2]*s
        x = -p[:, 1]*s + p[:, 2]*c
    return np.stack([z, y, x], axis=1) + center_zyx

def save_cell_spin_videos_allaxes(cell_labels, blobs, df,
                                  out_dir="cell_spin_videos",
                                  angles=tuple(range(0, 360, 10)),
                                  pad=8, fps=12,
                                  draw_point_radius_px=3):
    
    #Create per-cell rotation videos around X, Y, and Z axes.
    #Only exports cells that contain lysosomes.
    #Shows:
    #  - Green cell footprint
    #  - Cyan lysosome points inside cell
    #  - Cell ID text
    
    if not isinstance(cell_labels, np.ndarray) or cell_labels.max() == 0:
        print("No cells to export.")
        return
    if not isinstance(df, pd.DataFrame) or "cell_id_ch2" not in df.columns:
        print("Missing per-blob metadata; cannot filter by lysosome presence.")
        return

    cell_ids_with_lys = sorted(df.loc[
        (df["location_ch2"] == "cell") & (df["cell_id_ch2"] > 0),
        "cell_id_ch2"
    ].unique().astype(int))
    if len(cell_ids_with_lys) == 0:
        print("No cells contain lysosomes; no videos exported.")
        return

    os.makedirs(out_dir, exist_ok=True)
    COL_CYAN, COL_GREEN, TXT_COL = (255, 255, 0), (0, 255, 0), (240, 240, 240)
    props = [p for p in regionprops(cell_labels) if p.label in cell_ids_with_lys]
    axes_all = ["X", "Y", "Z"]
    Ztot, Ytot, Xtot = cell_labels.shape
    exported = 0

    for p in props:
        cid = p.label
        minz, miny, minx, maxz, maxy, maxx = p.bbox
        miny, maxy = max(0, miny - pad), min(Ytot, maxy + pad)
        minx, maxx = max(0, minx - pad), min(Xtot, maxx + pad)
        sub_lab = cell_labels[minz:maxz, miny:maxy, minx:maxx]
        if (sub_lab == cid).sum() == 0:
            continue
        subZ, subY, subX = sub_lab.shape
        center_zyx = np.array([(subZ-1)/2, (subY-1)/2, (subX-1)/2], dtype=np.float32)

        # lysosomes in this cell
        sub_blobs = df[(df["cell_id_ch2"] == cid) & (df["location_ch2"] == "cell")]
        if sub_blobs.empty:
            continue
        sel = sub_blobs.index.to_numpy()
        pts_local = np.stack([
            blobs[sel, 0]-minz,
            blobs[sel, 1]-miny,
            blobs[sel, 2]-minx
        ], axis=1).astype(np.float32)

        # iterate over each axis type
        for axis in axes_all:
            frames = []
            for ang in angles:
                # rotate volume around chosen axis
                if axis == "Y":
                    rot_lab = ndrotate(sub_lab, angle=ang, axes=(0,2), reshape=False, order=0, mode='nearest')
                elif axis == "X":
                    rot_lab = ndrotate(sub_lab, angle=ang, axes=(1,0), reshape=False, order=0, mode='nearest')
                else:  # Z
                    rot_lab = ndrotate(sub_lab, angle=ang, axes=(1,2), reshape=False, order=0, mode='nearest')

                mask_rot = (rot_lab > 0)
                footprint = mask_rot.max(axis=0).astype(np.uint8)
                frame = np.zeros((subY, subX, 3), dtype=np.uint8)

                # green overlay
                if footprint.any():
                    green = (footprint * 255)
                    overlay = frame.copy()
                    overlay[..., 1] = np.maximum(overlay[..., 1], green)
                    frame = cv2.addWeighted(frame, 1.0, overlay, 0.35, 0.0)
                    contours, _ = cv2.findContours((footprint*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cv2.drawContours(frame, contours, -1, COL_GREEN, 1, lineType=cv2.LINE_AA)

                # rotate and draw lysosome points
                pts_rot = _rotate_points(pts_local, center_zyx, ang, axis=axis)
                ys = np.clip(np.round(pts_rot[:, 1]).astype(int), 0, subY-1)
                xs = np.clip(np.round(pts_rot[:, 2]).astype(int), 0, subX-1)
                for k in range(len(xs)):
                    cv2.circle(frame, (int(xs[k]), int(ys[k])), draw_point_radius_px, COL_CYAN, -1, lineType=cv2.LINE_AA)

                frames.append(frame)

            out_path = os.path.join(out_dir, f"cell_{cid:03d}_spin{axis}.mp4")
            try:
                imageio.mimsave(out_path,
                                [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames],
                                fps=fps, format='FFMPEG')
                print(f"Saved: {out_path}")
            except TypeError:
                out_gif = out_path.replace(".mp4", ".gif")
                imageio.mimsave(out_gif,
                                [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames],
                                fps=fps)
                print(f"Saved (GIF): {out_gif}")
            exported += 1

    print(f"Exported {exported} per-cell rotation videos (X, Y, Z) → {out_dir}")


# ---- call it ----
save_cell_spin_videos_allaxes(cell_seg, blobs, df,
                              out_dir="cell_spin_videos",
                              angles=tuple(range(0, 360, 10)),
                              pad=8, fps=12)

# ==========================================
# 9) 3D screenshots (XY, YZ, XZ)
# ==========================================
def save_xy_3d_screenshot(viewer, path='ch2_segmentation_XY_3d.png'):
    viewer.dims.ndisplay = 3
    try:
        viewer.camera.angles = (90, 0, 0)
    except Exception:
        try:
            viewer.camera.elevation = 90
            viewer.camera.azimuth = 0
        except Exception:
            pass
    img_xy = viewer.screenshot(canvas_only=True)
    try:
        imageio.imwrite(path, img_xy)
    except Exception:
        cv2.imwrite(path, cv2.cvtColor(img_xy, cv2.COLOR_RGBA2BGRA))
    print(f"Saved 3D XY screenshot: {path}")

def save_yz_3d_screenshot(viewer, path='ch2_segmentation_YZ_3d.png'):
    viewer.dims.ndisplay = 3
    try:
        viewer.camera.angles = (0, 90, 0)
    except Exception:
        try:
            viewer.camera.elevation = 0
            viewer.camera.azimuth = 90
        except Exception:
            pass
    img_yz = viewer.screenshot(canvas_only=True)
    try:
        imageio.imwrite(path, img_yz)
    except Exception:
        cv2.imwrite(path, cv2.cvtColor(img_yz, cv2.COLOR_RGBA2BGRA))
    print(f"Saved 3D YZ screenshot: {path}")

def save_xz_3d_screenshot(viewer, path='ch2_segmentation_XZ_3d.png'):
    viewer.dims.ndisplay = 3
    try:
        viewer.camera.angles = (0, 0, 0)
    except Exception:
        try:
            viewer.camera.elevation = 0
            viewer.camera.azimuth = 0
        except Exception:
            pass
    img_xz = viewer.screenshot(canvas_only=True)
    try:
        imageio.imwrite(path, img_xz)
    except Exception:
        cv2.imwrite(path, cv2.cvtColor(img_xz, cv2.COLOR_RGBA2BGRA))
    print(f"Saved 3D XZ screenshot: {path}")

viewer.dims.ndisplay = 3
save_xy_3d_screenshot(viewer, path='cells_segmentation_lysosomes_XY_3d.png')
save_yz_3d_screenshot(viewer, path='cells_segmentation_lysosomes_YZ_3d.png')
save_xz_3d_screenshot(viewer, path='cells_segmentation_lysosomes_XZ_3d.png')

# ==========================================
# 10) EXTRA: Export Original stack (Ch1) as MP4
# ==========================================
try:
    raw_stack = np.array(image, dtype=np.float32)
    raw_norm = (255 * (raw_stack - raw_stack.min()) / (raw_stack.ptp() + 1e-8)).astype(np.uint8)

    frames_raw = []
    for z in range(raw_norm.shape[0]):
        frame_gray = raw_norm[z]
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        frames_raw.append(frame_bgr)

    mp4_name = "original_raw_ch1.mp4"
    imageio.mimsave(mp4_name, frames_raw, fps=8, format='FFMPEG')
    print(f"Saved: {mp4_name}")
except Exception as e:
    print("MP4 export of original stack failed:", e)

# ==========================================
# 11) EXTRA: Side-by-side (RAW | SEGMENTED) MP4
# ==========================================
try:
    def to_uint8_grayscale(vol):
        vol = vol.astype(np.float32)
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax <= vmin:
            return (np.zeros_like(vol, dtype=np.uint8))
        return (255.0 * (vol - vmin) / (vmax - vmin)).astype(np.uint8)

    raw_stack_u8 = to_uint8_grayscale(image)      # left panel (raw Ch1)
    seg_base_u8  = to_uint8_grayscale(image_2)    # right base (Ch2)

    Z, H, W = raw_stack_u8.shape
    fps = 8

    out_name = "raw_vs_segmented_side_by_side.mp4"
    writer = imageio.get_writer(out_name, fps=fps, format="FFMPEG")

    for z in range(Z):
        left_bgr = cv2.cvtColor(raw_stack_u8[z], cv2.COLOR_GRAY2BGR)
        base_bgr = cv2.cvtColor(seg_base_u8[z], cv2.COLOR_GRAY2BGR)

        if 'cell_mask' in globals():
            cell = (cell_mask[z].astype(np.uint8) * 255)
            overlay = base_bgr.copy()
            overlay[..., 1] = np.maximum(overlay[..., 1], cell)
            right_bgr = cv2.addWeighted(base_bgr, 1.0, overlay, 0.35, 0.0)
        else:
            right_bgr = base_bgr


        # --- draw ONLY in-cell lysosomes on the right panel ---
        if 'blobs' in globals() and len(blobs) > 0 and 'cell_mask' in globals():
            z_blobs = blobs[np.abs(blobs[:, 0] - z) < 0.5]
            for b in z_blobs:
                y, x = int(round(b[1])), int(round(b[2]))
                r = max(2, int(round(b[3])))
                # draw only if this blob center is inside the cell on this slice
                if 0 <= y < H and 0 <= x < W and cell_mask[z, y, x]:
                    cv2.circle(right_bgr, (x, y), r, (255, 255, 0), 2)  # yellow (in-cell only)
        # NOTE: no drawing at all for outside points


        if left_bgr.shape != right_bgr.shape:
            right_bgr = cv2.resize(right_bgr, (left_bgr.shape[1], left_bgr.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        divider = np.full((left_bgr.shape[0], 4, 3), 32, dtype=np.uint8)
        frame = cv2.hconcat([left_bgr, divider, right_bgr])

        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    writer.close()
    print(f"Saved: {out_name}")

except Exception as e:
    try:
        print("FFMPEG writer failed; attempting GIF fallback. Error:", e)
        frames = []
        for z in range(raw_stack_u8.shape[0]):
            left_bgr = cv2.cvtColor(raw_stack_u8[z], cv2.COLOR_GRAY2BGR)
            base_bgr = cv2.cvtColor(seg_base_u8[z], cv2.COLOR_GRAY2BGR)
            if 'cell_mask' in globals():
                cell = (cell_mask[z].astype(np.uint8) * 255)
                overlay = base_bgr.copy()
                overlay[..., 1] = np.maximum(overlay[..., 1], cell)
                right_bgr = cv2.addWeighted(base_bgr, 1.0, overlay, 0.35, 0.0)
            else:
                right_bgr = base_bgr

            # --- draw ONLY in-cell lysosomes on the right panel ---
            if 'blobs' in globals() and len(blobs) > 0 and 'cell_mask' in globals():
                z_blobs = blobs[np.abs(blobs[:, 0] - z) < 0.5]
                for b in z_blobs:
                    y, x = int(round(b[1])), int(round(b[2]))
                    r = max(2, int(round(b[3])))
                    # draw only if this blob center is inside the cell on this slice
                    if 0 <= y < H and 0 <= x < W and cell_mask[z, y, x]:
                        cv2.circle(right_bgr, (x, y), r, (255, 255, 0), 2)  # yellow (in-cell only)
            # NOTE: no drawing at all for outside points

            if left_bgr.shape != right_bgr.shape:
                right_bgr = cv2.resize(right_bgr, (left_bgr.shape[1], left_bgr.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            divider = np.full((left_bgr.shape[0], 4, 3), 32, dtype=np.uint8)
            frame = cv2.hconcat([left_bgr, divider, right_bgr])
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        imageio.mimsave("raw_vs_segmented_side_by_side.gif", frames, fps=8)
        print("Saved: raw_vs_segmented_side_by_side.gif")
    except Exception as e2:
        print("Side-by-side export failed:", e2)

# ==========================================
# 12) Run viewer
# ==========================================
napari.run()
