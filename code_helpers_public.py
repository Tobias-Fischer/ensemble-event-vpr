import cv2, os, json, sys, ast, re
import numpy as np
from os.path import join, isfile, basename, abspath
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F
import argparse

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_INT_RE = re.compile(r'^[+-]?\d+$')   # integer literal (no decimal, no exponent)
_FLOAT_RE = re.compile(r'^[+-]?(?:\d+\.\d*|\.\d+|\d+([eE][+-]?\d+))$')  # basic float/exponent

def _parse_numeric_token(token):
    """
    Parse a single token string into int or float, preserving ints.
    Raises argparse.ArgumentTypeError on failure.
    """
    t = token.strip()
    if not t:
        raise argparse.ArgumentTypeError("Empty numeric token")
    # Plain integer like "500" or "-42"
    if _INT_RE.match(t):
        try:
            return int(t)
        except ValueError:
            pass
    # Float-like tokens, including exponent forms like "1e3" or "3.14" or "300.0"
    try:
        # Try float conversion as fallback (handles exponents)
        val = float(t)
        # If float is integral and token looked like an integer with trailing .0,
        # we only keep float if user wrote decimals or exponent. Preserve int only for pure integer tokens.
        if _INT_RE.match(t):
            return int(val)
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid numeric token: {token!r}")

def _parse_number_list_string(s):
    """
    Parse a single string token which may be:
      - a python list literal: "[500, 1000]" -> preserves ints/floats from literal
      - a csv: "500,1000"
      - a single number: "500" or "3.14" or "1e3"
    Returns: list of ints/floats preserving integer type when input is integer.
    """
    s = s.strip()
    if not s:
        return []
    # python list literal (must both start and end with bracket)
    if s.startswith('[') and s.endswith(']'):
        try:
            v = ast.literal_eval(s)
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid python list literal: {e}")
        if not isinstance(v, (list, tuple)):
            raise argparse.ArgumentTypeError("Expected a list/tuple literal")
        parsed = []
        for x in v:
            if isinstance(x, int):
                parsed.append(x)
            elif isinstance(x, float):
                # If the literal was 300.0 it will be float; keep as float.
                parsed.append(x)
            else:
                # If it's a numeric string inside list: try parse it
                parsed.append(_parse_numeric_token(str(x)))
        return parsed
    # CSV: "500,1000" or "500, 1000"
    if ',' in s:
        tokens = [tok.strip() for tok in s.split(',') if tok.strip()]
        return [_parse_numeric_token(tok) for tok in tokens]
    # single numeric token
    return [_parse_numeric_token(s)]

class ListOrLiteral(argparse.Action):
    """
    Robust argparse Action that always puts a Python list on the namespace, preserving ints.
    Supports:
      - multiple separate tokens:  -w 500 1000
      - a single quoted python-list token: -w "[500, 1000]"
      - a single quoted csv token: -w "500,1000"
      - shell-split bracket tokens: -w [500, 1000] (passed as ['[500,', '1000]'])
    """
    def __call__(self, parser, namespace, values, option_string=None):
        # values will be a list when nargs='+'.
        # Examples:
        #   ['500', '1000']
        #   ['[500,', '1000]']  (shell split)
        #   ['[500,1000]']      (single token)
        #   ['500,1000']        (single token CSV)
        if isinstance(values, list):
            if len(values) > 1:
                # Try joining — covers shell-split bracket tokens like ['[500,', '1000]']
                joined = ' '.join(values).strip()
                if joined.startswith('[') and joined.endswith(']'):
                    try:
                        result = _parse_number_list_string(joined)
                    except argparse.ArgumentTypeError:
                        # fall back to parsing tokens individually
                        result = [_parse_numeric_token(v) for v in values]
                else:
                    # tokens are likely separate numeric tokens: parse each preserving ints
                    try:
                        result = [_parse_numeric_token(v) for v in values]
                    except argparse.ArgumentTypeError:
                        # fallback: try parsing joined as CSV or literal
                        result = _parse_number_list_string(joined)
            else:
                # single token: could be "[500,1000]" or "500,1000" or "500"
                result = _parse_number_list_string(values[0])
        else:
            # unlikely with nargs='+', but handle anyway
            result = _parse_number_list_string(values)
        setattr(namespace, self.dest, result)

def compare_images(image_paths1, image_paths2, image_num1, image_num2):
    if isinstance(image_paths1[image_num1], np.ndarray):
        img_A = image_paths1[image_num1]
    else:
        img_A = cv2.imread(image_paths1[image_num1])

    if isinstance(image_paths2[image_num2], np.ndarray):
        img_B = image_paths2[image_num2]
    else:
        img_B = cv2.imread(image_paths2[image_num2])

    fig, ax = plt.subplots(1, 2, figsize=(2 * 7.2, 5.4))

    if isinstance(image_paths1[image_num1], str):
        ax[0].set_title(basename(image_paths1[image_num1]) + ' (Place ' + str(image_num1) + ')')
    else:
        ax[0].set_title('Place ' + str(image_num1))
    if isinstance(image_paths2[image_num2], str):
        ax[1].set_title(basename(image_paths2[image_num2]) + ' (Place ' + str(image_num2) + ')')
    else:
        ax[1].set_title('Place ' + str(image_num2))
    if len(img_A.shape) == 3 and img_A.shape[2] == 3:
        ax[0].imshow(img_A[..., ::-1])
    else:
        ax[0].imshow(img_A, 'gray', vmin=0, vmax=255)
    if len(img_B.shape) == 3 and img_B.shape[2] == 3:
        ax[1].imshow(img_B[..., ::-1])
    else:
        ax[1].imshow(img_B, 'gray', vmin=0, vmax=255)


def get_timestamps(comparison_folder1, comparison_folder2):
    def read_timestamps(comparison_folder):
        with open(join(comparison_folder, 'timestamps.txt'), "r") as file_in:
            timestamps_in = [float(line.rstrip()) for line in file_in]
        return timestamps_in
    return np.array(read_timestamps(comparison_folder1)), np.array(read_timestamps(comparison_folder2))


def get_timestamp_matches(timestamps, timestamps_to_match):
    timestamps_matched = np.array([np.abs(timestamps - ts).argmin() for ts in timestamps_to_match])
    return timestamps_matched

def get_image_paths(folder1):
    return sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.png')])

def load_e2vid_abs_times(metadata_json_path, timestamps_txt_path):
    """Return absolute frame times (seconds) from metadata.json + e2vid timestamps.txt."""
    with open(metadata_json_path, 'r') as f:
        md = json.load(f)
    if 'start_time_ns' not in md:
        raise KeyError(f"{metadata_json_path} missing 'start_time_ns'")
    start_s = float(md['start_time_ns']) * 1e-9
    # timestamps.txt contains per-frame times in seconds (E2VID output)
    ts = np.loadtxt(timestamps_txt_path, dtype=float)
    abs_ts = start_s + ts
    if not np.all(np.diff(abs_ts) >= 0):
        abs_ts = np.sort(abs_ts)
    return abs_ts

# --- helpers ---
def _abs_times_for_dir(frames_dir: str) -> np.ndarray:
    ts_path   = os.path.join(frames_dir, 'timestamps.txt')
    meta_path = os.path.join(Path(frames_dir).parent, 'metadata.json')
    ts = load_e2vid_abs_times(str(meta_path), str(ts_path))  # seconds, sorted
    return ts.astype(float)

def _pair_canonical(q_times: np.ndarray, r_times: np.ndarray, max_dt: float):
    """
    Robust time pairing between two streams:
      1) fit linear map r ≈ a*q + b,
      2) nearest-neighbour under a cadence-based tolerance,
      3) enforce monotonicity,
      4) one fallback with a looser tol.
    Returns (t_q, t_r) with equal length.
    """
    q = np.asarray(q_times, float)
    r = np.asarray(r_times, float)
    if q.size == 0 or r.size == 0:
        return q[:0], r[:0]
    q.sort(); r.sort()

    # --- 1) linear time map r ≈ a*q + b (handles different window cadences) ---
    if q[-1] > q[0]:
        a = (r[-1] - r[0]) / (q[-1] - q[0])
    else:
        a = 1.0
    b = r[0] - a * q[0]
    r_pred = a * q + b

    # cadence-based tolerance (use median step); don’t trust tiny fixed max_dt
    dq = (np.median(np.diff(q)) if q.size > 1 else max_dt)
    dr = (np.median(np.diff(r)) if r.size > 1 else max_dt)
    tol = max(max_dt, 0.6 * max(dq, dr))

    def _nn_monotonic(q_hat, r_arr, tol_):
        idx = np.searchsorted(r_arr, q_hat, side='left')
        best = np.full(q_hat.size, -1, dtype=int)
        for i, (t, j) in enumerate(zip(q_hat, idx)):
            c = []
            if j > 0:           c.append(j - 1)
            if j < r_arr.size:  c.append(j)
            if not c:           continue
            c = np.asarray(c, int)
            k = int(c[np.argmin(np.abs(r_arr[c] - t))])
            if abs(r_arr[k] - t) <= tol_:
                best[i] = k
        # monotonic non-decreasing
        out = np.full_like(best, -1)
        prev = -1
        for i, j in enumerate(best):
            if j < 0: continue
            if j >= prev:
                out[i] = j; prev = j
            elif prev >= 0 and abs(r_arr[prev] - q_hat[i]) <= tol_:
                out[i] = prev
        valid = out >= 0
        return valid, out

    # --- 2) NN + monotonic with cadence tol ---
    valid, out = _nn_monotonic(r_pred, r, tol)

    # --- 3) fallback once with looser tol if empty ---
    if not np.any(valid):
        tol2 = max(tol * 2.0, 1.5 * max(dq, dr), 0.25)
        valid, out = _nn_monotonic(r_pred, r, tol2)
        if not np.any(valid):
            return q[:0], r[:0]

    return q[valid], r[out[valid]]

class ListOnDemand(object):
    def __init__(self):
        self.path_list = []
        self.image_list = []

    def append(self, item):
        self.path_list.append(item)
        self.image_list.append(None)

    def get_path_list(self):
        return self.path_list

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            #Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*idx.indices(len(self)))]
        elif isinstance(idx, list):
            return [self[ii] for ii in idx]
        elif isinstance(idx, np.ndarray):
            return self[idx.tolist()]
        elif isinstance(idx, str):
            return self[int(idx)]
        elif isinstance(idx, int) or isinstance(idx, np.generic):
            if idx < 0 : #Handle negative indices
                idx += len( self )
            if idx < 0 or idx >= len( self ) :
                raise IndexError("Index %d is out of range."%idx)

            if self.image_list[idx] is None:
                self.image_list[idx] = cv2.imread(self.path_list[idx])

            return self.image_list[idx]
        else:
            raise TypeError("Invalid argument type:", type(idx))

def get_image_sets_on_demand(image_paths1, image_paths2, matches1, matches2):
    images_set1 = ListOnDemand()
    images_set2 = ListOnDemand()
    for idx1 in range(len(matches1)):
        image_num1 = matches1[idx1]
        images_set1.append(image_paths1[image_num1])
    for idx2 in range(len(matches2)):
        image_num2 = matches2[idx2]
        images_set2.append(image_paths2[image_num2])
    return images_set1, images_set2


class L2AcrossChannels(nn.Module):
    """Match TF: tf.nn.l2_normalize(x, axis=-1) on NCHW -> dim=1."""
    def forward(self, x):  # x: [B, C, H, W]
        return F.normalize(x, p=2, dim=1)

class L2Normalize1d(nn.Module):
    """Final L2 on descriptor vector."""
    def forward(self, x):  # x: [B, F]
        return F.normalize(x, p=2, dim=1)

def build_model(num_clusters = 64, add_wpca = False, wpca_dim: int = 4096, freeze_backbone = True, device = None, netvlad_folder=None):
    sys.path.append(netvlad_folder)
    from netvlad import NetVLAD
    """
    Returns:
        model: nn.Module producing [B, F] descriptors
        feat_dim: int (F)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # VGG16 conv1..conv5_3 with ImageNet feature weights
    backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    # NetVLAD layer: we L2-normalize features ourselves, so keep normalize_input=False
    vlad = NetVLAD(num_clusters=num_clusters, dim=512, normalize_input=False)

    layers = [
        backbone,             # conv features (C=512)
        L2AcrossChannels(),   # TF-style channel-wise L2 BEFORE NetVLAD
        vlad,                 # -> [B, K*512]
    ]

    feat_dim = num_clusters * 512
    if add_wpca:
        # TF does 1x1 conv on the flattened vector; Linear is equivalent here
        layers += [nn.Linear(feat_dim, wpca_dim, bias=True)]
        feat_dim = wpca_dim

    layers += [L2Normalize1d()]  # final L2 on descriptor vector

    model = nn.Sequential(*layers).to(device).eval()
    return model, feat_dim

class _ImageSetDataset(Dataset):
    """
    Accepts a list of file paths OR np.ndarray images (HWC, BGR or RGB, uint8/float).
    Resizes to a fixed target_size for batching, converts to RGB, normalizes, CHW tensor.
    """
    def __init__(self, images_set, target_size=(480, 640)):
        self.items = images_set
        self.h, self.w = target_size

    def __len__(self):
        return len(self.items)

    def _load_image(self, item):
        # item can be a path or ndarray
        if isinstance(item, (np.ndarray, np.generic)):
            img = item
        else:
            img = cv2.imread(item, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {item}")

        # Ensure 3-channel RGB
        if img.ndim == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA/BGRA -> BGR first then to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Assume BGR if from cv2, otherwise RGB — safest is to coerce via BGR2RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Fixed-size resize for consistent batching
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)

        # To float32, [0,1], normalize
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        # HWC -> CHW tensor
        img = np.transpose(img, (2, 0, 1))  # (3, H, W)
        return torch.from_numpy(img)

    def __getitem__(self, idx):
        return self._load_image(self.items[idx])

def get_vlad_features(sess, net_out, image_batch, images_set, save_name=None, batch_size=4, tqdm_position=1):
    if save_name and isfile(save_name):
        pre_extracted_features = np.load(save_name)
        if pre_extracted_features.shape[0] == len(images_set):
            return pre_extracted_features
        else:
            print('Warning: shapes of pre_extracted_features and images_set do not match, re-compute!')

    descs = []
    for batch_offset in tqdm(range(0, len(images_set), batch_size), position=tqdm_position, leave=False):
    # for batch_offset in range(0, len(images_set), batch_size):
        images = []
        for i in range(batch_offset, batch_offset + batch_size):
            if i == len(images_set):
                break

            image = images_set[i]
            if not isinstance(image, (np.ndarray, np.generic) ):
                image = cv2.imread(image)

            if image_batch.shape[3] == 1:  # grayscale
                images.append(np.expand_dims(np.expand_dims(image, axis=0), axis=-1))
            else:  # color
                image_inference = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(np.expand_dims(image_inference, axis=0))

        batch = np.concatenate(images, 0)
        descs = descs + list(sess.run(net_out, feed_dict={image_batch: batch}))

    netvlad_feature_list = np.array(descs)

    if save_name:
        np.save(save_name, netvlad_feature_list)

    return netvlad_feature_list

def get_vlad_features_pytorch(
    model,
    images_set,
    save_name=None,
    batch_size=8,
    num_workers=4,
    tqdm_position=1,
    device=None,
    target_size=(480, 640),
    mmap_safely=True,
):
    """
    PyTorch replacement for TF get_vlad_features.

    Args:
        model: torch.nn.Module returning NetVLAD descriptors of shape [B, K*D].
               (e.g., nn.Sequential(vgg16.features.eval(), NetVLAD(...)).eval())
        images_set: list of image paths or np.ndarray images (HWC).
        save_name: optional .npy path to cache descriptors; if shape matches, loads & returns.
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers (set 0 on Windows if issues).
        tqdm_position: progress bar row for nested tqdm.
        device: torch device (auto-detect if None).
        target_size: (H, W) resize for batching.
        mmap_safely: if True, write directly to an on-disk .npy using open_memmap (no big RAM spike).

    Returns:
        np.ndarray of descriptors with shape (N, K*D), dtype float32.
    """
    # ----- Cache short-circuit -----
    if save_name and isfile(save_name):
        cached = np.load(save_name, mmap_mode="r")
        if cached.shape[0] == len(images_set):
            return np.asarray(cached)  # zero-copy view
        else:
            print("Warning: cached feature count mismatch; recomputing…")

    # ----- Device & model prep -----
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    model = model.to(device).eval()

    # ----- Data pipeline -----
    ds = _ImageSetDataset(images_set, target_size=target_size)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # ----- Allocate output (memmap after first batch when we know feat_dim) -----
    use_memmap = bool(save_name and mmap_safely)
    feats = None  # will be np.ndarray (or memmap) of shape (N, feat_dim)
    write_ptr = 0

    # ----- Extract -----
    with torch.inference_mode():
        for batch in tqdm(dl, position=tqdm_position, leave=False, desc="NetVLAD"):
            batch = batch.to(device, non_blocking=True)
            out = model(batch)               # [B, K*D]
            out_cpu = out.detach().cpu().to(torch.float32).numpy()  # (B, F)

            if feats is None:
                feat_dim = out_cpu.shape[1]
                if use_memmap:
                    # Write directly to .npy on disk; safe for huge N
                    feats = np.lib.format.open_memmap(
                        save_name, mode="w+", dtype="float32", shape=(len(ds), feat_dim)
                    )
                else:
                    feats = np.empty((len(ds), feat_dim), dtype=np.float32)

            bsz = out_cpu.shape[0]
            feats[write_ptr:write_ptr + bsz, :] = out_cpu
            write_ptr += bsz

    # ----- Finalize -----
    if use_memmap:
        feats.flush()  # ensure data is written

    return feats