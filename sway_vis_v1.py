"""

Bout this code--------------
- Uses MediaPipe Pose to estimate 3D world landmarks in meters.
- Approximates Center of Mass as the mean of all pose world landmarks.
- Lets the user choose an axis:
    lr = mediolateral sway (world X)   -> labeled ML
    bf = anteroposterior sway (world Z) -> labeled AP
- Records the chosen axis for a set duration, then classifies SD (≈ RMS) vs 95th-percentile cut-offs.
- Draws a line during the recrding so the user can see the sway.

Two input modes
---------------
- Live camera : --mode live --camera 0
- Video file playback: --mode file --video /path/to/file.mp4

Controls
--------
- p = start prompt -> type lr/bf -> press 1/2/3 for stance (Narrow/Hip/Shoulder)
- c = cancel countdown, q = quit
- After results: press any key to reset

"""

import cv2
import mediapipe as mp
import numpy as np
import time, argparse, math
from pathlib import Path

# ---------- args (kinda basic, but does the job) ------------------------
ap = argparse.ArgumentParser(description="CoM sway (live/file) w/ path trace")
ap.add_argument('--mode', choices=['auto','live','file'], default='auto',
                help="Pick input: live, file, or auto (ask at start).")
ap.add_argument('--camera', type=int, default=0, help="Camera index (default 0).")
ap.add_argument('--video', type=str, default=None, help="Path to a video file.")
ap.add_argument('--secs', type=float, default=120.0, help="Record seconds (default 120).")
ap.add_argument('--prep', type=float, default=3.0, help="Countdown seconds (default 3).")
ap.add_argument('--trace-thick', type=int, default=2, help="Polyline thickness.")
# mirror helps w/ webcams; off for files if you prefer
mx = ap.add_mutually_exclusive_group()
mx.add_argument('--mirror', dest='mirror', action='store_true', help="Mirror the view (selfie).")
mx.add_argument('--no-mirror', dest='mirror', action='store_false', help="Do not mirror.")
ap.set_defaults(mirror=True)
args, _ = ap.parse_known_args()

# ---------- pick input source (ask once if auto) ------------------------
mode_in = args.mode
if mode_in == 'auto':
    try:
        choice = input("Input source? [l]ive / [f]ile (default live): ").strip().lower()
    except Exception:
        choice = 'l'
    mode_in = 'file' if choice.startswith('f') else 'live'

if mode_in == 'file' and not args.video:
    try:
        args.video = input("Enter path to video (mp4/avi/mov…): ").strip().strip('"').strip("'")
    except Exception:
        args.video = ""

# ---------- colors & tiny UI constants (meh) ----------------------------
RED, CYAN, GRN = (0,0,255), (255,255,0), (0,255,0)
TRACE_COL = (0,215,255)     # kinda goldish for the path line
TOP_TXT  = (20,40)
RESULT_TXT = (20,82)
SHADOW = (0,0,0)
FS, FS_BIG, FT = 0.5, 1.0, 1
R_DOT, R_COM = 2, 6

# ---------- pose init (leave defaults; this works ok) -------------------
mp_p = mp.solutions.pose
pose = mp_p.Pose(
    model_complexity=1,              # 2 is slower, little more stable maybe
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- axis/stance maps (dont mess with labels) --------------------
# LR = mediolateral (X), BF = anteroposterior (Z)
EXERCISE = {'lr': (0,'LR'), 'bf': (2,'BF')}
STANCE_KEYS = {'1': ('narrow','Narrow'),
               '2': ('hip','Hip‑width'),
               '3': ('shoulder','Shoulder‑width')}
# 95th percentile SD (cm) healthy young, eyes open, ~120s
CUTOFF95_CM = {
    'narrow':   {'AP': 0.695, 'ML': 0.57},
    'hip':      {'AP': 0.584, 'ML': 0.324},
    'shoulder': {'AP': 0.579, 'ML': 0.254}
}
LABEL_TO_DIR = {'LR':'ML','BF':'AP'}  # show ML for LR, AP for BF etc

# ---------- open source -------------------------------------------------
is_live = (mode_in == 'live')
cap = None
first_frame = None
file_fps = 0.0
delay_ms = 1  # live wants small wait to keep UI reactive

if is_live:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")
else:
    if not args.video:
        raise SystemExit("No video path given. Use --video or pick live mode.")
    video_path = str(Path(args.video))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1, int(round(1000.0 / file_fps)))
    ok_first, first_frame = cap.read()
    if not ok_first:
        raise SystemExit("Could not read first frame from file.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind for clean start

print("Controls: p=start • type lr/bf • press 1/2/3 for stance • h=hide dots • c=cancel • q=quit")

# ---------- state (simple but works) ------------------------------------
IDLE, PROMPT_AX, PROMPT_STANCE, READY, RECORD, SHOW = range(6)
mode, typed = IDLE, ''
axis, label = None, ''
stance_key, stance_label = None, ''
t_prep, t_start = 0.0, None
trace_vals, path_px = [], []
std_cm, cutoff_used = 0.0, 0.0
status_text = '—'
frozen_disp = None
show_points = True  # toggle with 'h'

# ---------- lil helpers -------------------------------------------------
def draw_top_bar(img):
    """dark strip on top so text is readable (yea, it’s ugly but okay)"""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0,0), (w,48), (0,0,0), -1)

def draw_text(img, text, org, color, big=False):
    """text with a lazy shadow so it pops a bit (not fancy)"""
    fs = FS_BIG if big else FS
    cv2.putText(img, text, (org[0]+1, org[1]+1), cv2.FONT_HERSHEY_SIMPLEX, fs, SHADOW, FT, cv2.LINE_AA)
    cv2.putText(img, text, org,                      cv2.FONT_HERSHEY_SIMPLEX, fs, color,  FT, cv2.LINE_AA)

def draw_trace(img, pts, color, thickness=2):
    """whole CoM path so far. if only 1 pt then meh."""
    if len(pts) > 1:
        poly = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [poly], False, color, thickness, lineType=cv2.LINE_AA)

def classify(vals_m, label, stance_key):
    """compute SD (cm) then compare to stance cut‑off. simple calc, not robust"""
    if len(vals_m) == 0:
        return 0.0, 0.0, 'NO DATA'
    v = np.asarray(vals_m, dtype=np.float32)
    v = v - v.mean()                      # remove mean -> SD ≈ RMS of sway
    sd_cm = float(v.std(ddof=0) * 100.0)  # meters -> cm
    direction = LABEL_TO_DIR[label]       # ML if LR, AP if BF (ye)
    cutoff = CUTOFF95_CM[stance_key][direction]
    status = 'HEALTHY' if sd_cm <= cutoff else 'UNHEALTHY (above 95th)'
    return sd_cm, cutoff, status

# ---------- main loop ---------------------------------------------------
try:
    while True:

        # choose frame (freeze on SHOW)
        if mode == SHOW:
            disp = frozen_disp.copy() if frozen_disp is not None else np.zeros((480,640,3), np.uint8)
        else:
            if is_live:
                ok, frm = cap.read()
                if not ok:
                    time.sleep(0.005)   # cam hiccup, just chill a tiny bit ( dont evven rember y but  deepsick told me to do it previously)
                    continue
                disp = cv2.flip(frm, 1) if args.mirror else frm
            else:
                if mode == RECORD:
                    ok, frm = cap.read()
                    if not ok:
                        # EOF during record -> just finalize with what we got
                        std_cm, cutoff_used, status_text = classify(trace_vals, label, stance_key)
                        frozen_disp = disp.copy() if 'disp' in locals() else first_frame.copy()
                        mode = SHOW
                        disp = frozen_disp.copy()
                    else:
                        disp = cv2.flip(frm, 1) if args.mirror else frm
                else:
                    disp = first_frame.copy()  # keep it static until we start

        # pose on current disp
        res = pose.process(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))

        # landmarks + CoM
        com_px = None
        if res.pose_landmarks and res.pose_world_landmarks:
            im_lm = res.pose_landmarks.landmark
            wd_lm = res.pose_world_landmarks.landmark

            if show_points:
                # tiny red dots to make sure tracking is alive (yeah a bit noisy)
                for im in im_lm:
                    x, y = int(im.x*disp.shape[1]), int(im.y*disp.shape[0])
                    cv2.circle(disp, (x,y), R_DOT, RED, -1, cv2.LINE_AA)

            # “CoM” ~ mean of world landmarks (very rough aproximation)
            wd_xyz = np.array([[p.x, p.y, p.z] for p in wd_lm], dtype=np.float32)
            com_w  = wd_xyz.mean(axis=0)  # [x,y,z] meters in cam space
            im_xy  = np.array([[p.x, p.y] for p in im_lm], dtype=np.float32).mean(axis=0)
            com_px = (int(im_xy[0]*disp.shape[1]), int(im_xy[1]*disp.shape[0]))

            # green dot + label so folks see where CoM is right now
            cv2.circle(disp, com_px, R_COM, GRN, -1, cv2.LINE_AA)
            draw_text(disp, "CoM", (com_px[0]+6, com_px[1]-6), GRN)

            if mode == RECORD and axis is not None:
                trace_vals.append(float(com_w[axis]))   # ML=X (0) or AP=Z (2)
                path_px.append(com_px)
                draw_trace(disp, path_px, TRACE_COL, thickness=args.trace_thick)

        # banner + key handling
        draw_top_bar(disp)

        # UI pacing: for file during RECORD pace by fps, else be snappy
        wait = delay_ms if (not is_live and mode == RECORD) else 1
        k = cv2.waitKey(wait) & 0xFF
        if k == ord('q'):
            break
        if k == ord('h'):
            show_points = not show_points   # handy if dots are too busy

        if mode == IDLE:
            draw_text(disp, "Press p to select axis & stance (q quits)", TOP_TXT, CYAN)
            if k == ord('p'):
                mode, typed = PROMPT_AX, ''

        elif mode == PROMPT_AX:
            draw_text(disp, "Type: lr (left-right / ML)  or  bf (back-front / AP)", TOP_TXT, CYAN)
            if k != 255 and k not in (13,10):
                try:
                    typed += chr(k).lower()
                except ValueError:
                    pass
                if typed in EXERCISE:
                    axis, label = EXERCISE[typed]
                    typed = ''
                    mode  = PROMPT_STANCE
                elif len(typed) > 2:  # user typed junk, just bail
                    typed = ''
                    mode  = IDLE

        elif mode == PROMPT_STANCE:
            draw_text(disp, "Select stance: 1=Narrow  2=Hip  3=Shoulder   (c cancels)", TOP_TXT, CYAN)
            if k == ord('c'):
                mode = IDLE
            elif k != 255 and chr(k) in STANCE_KEYS:
                stance_key, stance_label = STANCE_KEYS[chr(k)]
                trace_vals, path_px = [], []
                t_start = None
                if not is_live:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # start from top for files
                t_prep = time.time()
                mode   = READY

        elif mode == READY:
            left = args.prep - (time.time() - t_prep)
            if left <= 0:
                trace_vals, path_px = [], []
                if not is_live:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                t_start = time.time()
                mode = RECORD
            else:
                draw_text(
                    disp,
                    f"{label}/{stance_label}: starting in {int(math.ceil(left))}…  (c cancels)",
                    TOP_TXT, CYAN, big=True
                )
                if k == ord('c'):
                    mode = IDLE

        elif mode == RECORD:
            elapsed = max(0.0, time.time() - (t_start or time.time()))
            left = max(0, int(args.secs - elapsed + 0.999))
            draw_text(disp, f"{label}/{stance_label} recording: {left}s", TOP_TXT, CYAN)
            if elapsed >= args.secs:
                std_cm, cutoff_used, status_text = classify(trace_vals, label, stance_key)
                print(f"STD {label} {stance_label} = {std_cm:.2f} cm  |  cut-off {cutoff_used:.2f} cm  =>  {status_text}")
                frozen_disp = disp.copy()
                mode = SHOW

        elif mode == SHOW:
            colour = GRN if "HEALTHY" in status_text else RED
            draw_text(
                disp,
                f"{label}/{stance_label}  SD = {std_cm:.2f} cm  (95th {cutoff_used:.2f} cm)  ->  {status_text}",
                RESULT_TXT, colour
            )
            draw_text(disp, "Press any key to reset; q to quit", (20,112), CYAN)
            if k != 255:
                # reset for another try. TODO: consider keeping last trace, idk.
                trace_vals, path_px = [], []
                std_cm, cutoff_used, status_text = 0.0, 0.0, '—'
                axis, label = None, ''
                stance_key, stance_label = None, ''
                frozen_disp = None
                if not is_live:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                mode = IDLE

        cv2.imshow("Sway viz (live/file) — q quits", disp)

finally:
    if cap: cap.release()
    cv2.destroyAllWindows()
    pose.close()
