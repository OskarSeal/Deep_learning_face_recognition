from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image
import torch
import numpy as np
import time
import os

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Running on device: {device}")

# ── Models ────────────────────────────────────────────────────────────────────
# MTCNN must stay on CPU — MPS has a bug with adaptive pooling inside MTCNN
mtcnn = MTCNN(keep_all=True, device='cpu', post_process=False,
              min_face_size=40,
              thresholds=[0.6, 0.7, 0.7])

print("Loading InceptionResnetV1 model pretrained on VGGFace2...")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("✓ Face recognition model loaded successfully")

# ── Webcam ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ── Known-face database ───────────────────────────────────────────────────────
known_embeddings_matrix: torch.Tensor | None = None  # [N, 512]
known_names: list[str] = []
_embeddings_list: list[torch.Tensor] = []            # each [1, 512]
person_thresholds: dict[str, float] = {}             # per-person thresholds


def _rebuild_matrix() -> None:
    """Rebuild the stacked embedding matrix and per-person thresholds."""
    global known_embeddings_matrix
    if _embeddings_list:
        known_embeddings_matrix = torch.cat(_embeddings_list, dim=0)
    else:
        known_embeddings_matrix = None


def _update_person_threshold(name: str) -> None:
    """
    Set a per-person threshold based on intra-class embedding spread.
    Uses mean pairwise distance + margin. Falls back to 0.4 for single photos.
    """
    indices = [i for i, n in enumerate(known_names) if n == name]
    if len(indices) < 2:
        person_thresholds[name] = 0.4
        return

    embs = known_embeddings_matrix[indices]
    embs_norm = embs / embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    dist_matrix = 1.0 - torch.mm(embs_norm, embs_norm.T)
    n = len(indices)
    upper = [dist_matrix[i, j].item() for i in range(n) for j in range(i + 1, n)]
    mean_dist = sum(upper) / len(upper)
    person_thresholds[name] = round(mean_dist + 0.1, 3)
    print(f"  Adaptive threshold for '{name}': {person_thresholds[name]:.3f}")


def apply_clahe(pil_image: Image.Image) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve
    recognition under poor or uneven lighting conditions.
    Only the luminance channel is enhanced to avoid colour distortion.
    """
    img_np = np.array(pil_image)
    lab    = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


def add_known_face(image: Image.Image, name: str) -> bool:
    """
    Extract a face embedding and add/average it into the known-faces database.
    - Applies CLAHE preprocessing for lighting robustness.
    - Normalizes pixel values before passing to resnet (fixes false positives
      caused by post_process=False).
    - Averages embeddings if the person already exists in the database.
    - Updates the adaptive per-person threshold after each addition.
    """
    image = apply_clahe(image)
    face  = mtcnn(image)

    if face is None:
        print(f"  ✗ No face detected for '{name}'")
        return False

    face_tensor = face[0:1]                                # first face [1,C,H,W]
    face_tensor = (face_tensor.float() - 127.5) / 128.0   # normalize to ~[-1, 1]

    with torch.no_grad():
        emb = resnet(face_tensor.to(device)).cpu().float()  # [1, 512]

    if name in known_names:
        idx = known_names.index(name)
        _embeddings_list[idx] = (_embeddings_list[idx] + emb) / 2
        _rebuild_matrix()
        _update_person_threshold(name)
        print(f"  ✓ Updated embedding for '{name}' (averaged)")
    else:
        _embeddings_list.append(emb)
        known_names.append(name)
        _rebuild_matrix()
        _update_person_threshold(name)
        print(f"  ✓ Added '{name}' to database")

    return True


def recognize_faces_batch(face_tensors: torch.Tensor) -> list[tuple[str, float]]:
    """
    Recognise multiple faces in a single resnet forward pass.
    Uses vectorised cosine distance (one matmul) instead of a Python loop.
    Applies per-person adaptive thresholds for better discrimination.

    Parameters
    ----------
    face_tensors : [N, C, H, W]  — already normalized to ~[-1, 1]

    Returns
    -------
    List of (name, distance) tuples, one per input face.
    """
    with torch.no_grad():
        embeddings = resnet(face_tensors.to(device)).cpu().float()  # [N, 512]

    if known_embeddings_matrix is None:
        return [("Unknown", 1.0)] * embeddings.shape[0]

    # Vectorised cosine distance
    emb_norm   = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    known_norm = known_embeddings_matrix / \
                 known_embeddings_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
    dist_matrix = 1.0 - torch.mm(emb_norm, known_norm.T)  # [N, M]

    results: list[tuple[str, float]] = []
    for row in dist_matrix:
        min_val, min_idx = row.min(dim=0)
        dist      = min_val.item()
        candidate = known_names[min_idx.item()]
        threshold = person_thresholds.get(candidate, 0.4)
        name      = candidate if dist < threshold else "Unknown"
        results.append((name, dist))

    return results


def load_known_faces_from_folder(folder_path: str = "known_faces") -> None:
    """
    Load known faces from a folder of images.

    Naming convention
    -----------------
    Single photo  :  John_Doe.jpg          → label "John Doe"
    Multiple photos: John_Doe__1.jpg       → label "John Doe"
                     John_Doe__side.jpg    → label "John Doe"  (averaged)

    Rules
    -----
    - Use __ to separate the name from any suffix (suffix is ignored).
    - Use _ within the name for spaces  (John_Doe → "John Doe").
    - Accepted formats: .jpg  .jpeg  .png  .bmp
    - One face per photo (MTCNN takes the first detected face).
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder '{folder_path}'. Add photos there.")
        return

    print(f"\nLoading faces from '{folder_path}' …")
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    for filename in sorted(os.listdir(folder_path)):
        if os.path.splitext(filename)[1].lower() not in extensions:
            continue
        raw_name = os.path.splitext(filename)[0]   # strip extension
        name     = raw_name.split('__')[0]          # strip __suffix
        name     = name.replace('_', ' ')           # underscores → spaces
        path     = os.path.join(folder_path, filename)
        try:
            image = Image.open(path).convert('RGB')
            add_known_face(image, name)
        except Exception as exc:
            print(f"  ✗ Error loading '{filename}': {exc}")


# ── Load existing database ────────────────────────────────────────────────────
try:
    data = torch.load('face_database.pt')
    _embeddings_list = list(data['embeddings'])
    known_names      = list(data['names'])
    _rebuild_matrix()
    # Rebuild per-person thresholds from loaded data
    for name in set(known_names):
        _update_person_threshold(name)
    print(f"✓ Loaded {len(known_names)} faces from saved database")
except FileNotFoundError:
    print("ℹ  No existing database found – starting fresh")
    load_known_faces_from_folder("known_faces")

print("\n=== Controls ===")
print("Q  quit   |  A  add face   |  S  save DB   |  L  load DB   |  C  clear DB")
print("================\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
PROCESS_EVERY_N_FRAMES = 2   # increase for higher FPS, decrease for lower latency
frame_count    = 0
fps            = 0.0
prev_time      = time.time()
cached_results: list[tuple[int, int, int, int, str, float]] = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count   += 1
    frame_display  = frame.copy()

    now       = time.time()
    fps       = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
    prev_time = now

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_pil = apply_clahe(frame_pil)

        # ── ADD: shrink for detection only ────────────────────────────
        DETECTION_SIZE = (320, 240)
        small_pil = frame_pil.resize(DETECTION_SIZE)

        # ── REPLACE: use small_pil for detect, frame_pil for extract ──
        t0 = time.time()
        boxes, probs = mtcnn.detect(small_pil)          # ← was frame_pil
        t1 = time.time()
        cached_results = []

        if boxes is not None:
            # ── Scale boxes back to original resolution ───────────────
            scale_x = frame_pil.width  / DETECTION_SIZE[0]
            scale_y = frame_pil.height / DETECTION_SIZE[1]
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            face_list = mtcnn.extract(frame_pil, boxes, save_path=None)
            t2 = time.time()

            if face_list is not None:
                valid = [(f, b) for f, b in zip(face_list, boxes) if f is not None]

                if valid:
                    faces_tensor = torch.stack([f for f, _ in valid])
                    faces_tensor = (faces_tensor.float() - 127.5) / 128.0

                    results = recognize_faces_batch(faces_tensor)
                    t3 = time.time()

                    # ── Benchmark print ───────────────────────────────
                    print(f"detect={1000*(t1-t0):.1f}ms  "
                        f"extract={1000*(t2-t1):.1f}ms  "
                        f"resnet={1000*(t3-t2):.1f}ms")

                    for (_, box), (name, conf) in zip(valid, results):
                        x1, y1, x2, y2 = (int(b) for b in box)
                        cached_results.append((x1, y1, x2, y2, name, conf))
            else:
                for box in boxes:
                    x1, y1, x2, y2 = (int(b) for b in box)
                    cached_results.append((x1, y1, x2, y2, "Unknown", 1.0))

    # ── Draw cached results ───────────────────────────────────────────────
    for (x1, y1, x2, y2, name, conf) in cached_results:
        color = (0, 255, 0) if name != "Unknown" else (0, 255, 255)
        cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_display, f"{name} ({conf:.2f})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if cached_results:
        cv2.putText(frame_display, f"Faces: {len(cached_results)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ── HUD ───────────────────────────────────────────────────────────────
    h = frame_display.shape[0]
    cv2.putText(frame_display, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_display, f"Known faces: {len(set(known_names))}",
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame_display,
                "Q:Quit | A:Add face | S:Save DB | L:Load DB | C:Clear DB",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Face Recognition – InceptionResnetV1', frame_display)

    # ── Key handling ──────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('a'):
        if cached_results:
            name_input = input("\nEnter name for this face: ").strip()
            if name_input:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                if add_known_face(frame_pil, name_input):
                    print(f"✓ Face saved as '{name_input}'")
                else:
                    print("✗ Failed to add face")
        else:
            print("✗ No face detected to add")

    elif key == ord('c'):
        _embeddings_list.clear()
        known_names.clear()
        person_thresholds.clear()
        known_embeddings_matrix = None
        print("✓ Database cleared")

    elif key == ord('s'):
        if _embeddings_list:
            torch.save({'embeddings': _embeddings_list, 'names': known_names},
                       'face_database.pt')
            print(f"✓ Database saved with {len(set(known_names))} people "
                  f"({len(known_names)} embeddings)")
        else:
            print("✗ No faces to save")

    elif key == ord('l'):
        try:
            data = torch.load('face_database.pt')
            _embeddings_list = list(data['embeddings'])
            known_names      = list(data['names'])
            _rebuild_matrix()
            for name in set(known_names):
                _update_person_threshold(name)
            print(f"✓ Loaded {len(set(known_names))} people from database")
        except FileNotFoundError:
            print("✗ No saved database found")

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("\n✓ Face recognition ended")
