"""
MARS HILL ROVER (endless, with astronaut)  -  Python + OpenCV + NumPy (+ MediaPipe, + pygame for sound)

python mars_rover.py          # hand gestures: MediaPipe if available, else OpenCV-only
python mars_rover.py --cv     # force OpenCV-only hand tracking
python mars_rover.py --keys   # keyboard only

Hand:  open palm = gas | fist = brake/reverse | index+middle = BOOST
       tilt hand = rotate rover in the air | C = redo hand set-up (hand out of box, then open palm in box)
Keys:  D/W/Right gas | A/S/Left brake | Space boost | Q/E (Up/Down) tilt
       M mute | R restart | B maps | N new player (after a crash) | ESC quit    (pip install pynput = true held-key detection)
Install: pip install opencv-python numpy pygame mediapipe==0.10.14
"""
import os, sys, math, time, threading, io, wave, json
from collections import deque
import cv2
import numpy as np

W, H = 1440, 810
WORLD = 0                                # grows forever: terrain is generated in chunks as you drive
CAM_OFFSET = int(W * 0.3)
GRAV = 0.40
R = 18
LA = cv2.LINE_AA
rng = np.random.default_rng(7)
IDX = np.zeros(0, np.float32)


# ---------- maps ----------
BODY = {  # planet looks: (base colour, bands[(y_offset, height, colour)]) in units of the planet radius
    "earth": ((170, 110, 40), [(-0.45, 0.15, (235, 235, 235)), (0.05, 0.25, (80, 150, 70)), (0.5, 0.12, (235, 235, 235))]),
    "jupiter": ((130, 180, 225), [(-0.7, 0.18, (90, 140, 200)), (-0.3, 0.15, (200, 225, 245)),
                                  (0.1, 0.2, (70, 110, 170)), (0.55, 0.15, (170, 205, 235))]),
}
MAPS = [
    dict(name="MARS", sub="Rusty hills. Learn the ropes.", gtxt="NORMAL", lvl=1,
         grav=0.40, d0=0.25, span=20000, ridge_at=1500, burn=1.0, gap=2200, rescue=20, stars=170,
         sky=((75, 40, 70), (110, 110, 200), (170, 200, 250)), sun=((60, 120, 180), (215, 235, 255), 30),
         soil=((75, 120, 205), (30, 48, 98)), crust=((120, 170, 245), (40, 70, 140)),
         mtn=((130, 100, 150), (105, 90, 145), (80, 80, 130)), bodies=[("dot", W - 520, 110, 9)]),
    dict(name="MOON", sub="Low gravity. Huge jumps, risky landings.", gtxt="LOW", lvl=2,
         grav=0.22, d0=0.45, span=16000, ridge_at=800, burn=1.15, gap=2600, rescue=14, stars=420,
         sky=((12, 10, 10), (30, 26, 26), (60, 55, 55)), sun=((90, 90, 90), (235, 240, 245), 24),
         soil=((150, 150, 155), (60, 60, 66)), crust=((215, 215, 220), (90, 90, 95)),
         mtn=((95, 95, 100), (75, 75, 80), (55, 55, 60)), bodies=[("earth", 260, 170, 58)]),
    dict(name="Volcano", sub="Volcanic, steep and heavy. Fuel is scarce.", gtxt="HIGH", lvl=3,
         grav=0.4, d0=0.70, span=12000, ridge_at=300, burn=1.3, gap=3000, rescue=8, stars=120,
         sky=((20, 10, 35), (30, 60, 120), (70, 170, 230)), sun=((50, 150, 220), (200, 240, 255), 22),
         soil=((40, 200, 240), (25, 40, 120)), crust=((120, 255, 255), (20, 60, 140)),
         mtn=((60, 70, 160), (45, 50, 120), (35, 35, 85)), bodies=[("jupiter", 330, 200, 120)]),
]
MAPI, M = 0, MAPS[0]


def set_map(i):
    """Switch map: rebuild terrain + background cache."""
    global MAPI, M, WORLD, terrain, IDX
    MAPI, M = i, MAPS[i]
    WORLD, terrain = 0, np.zeros(0, np.float32)
    extend_world(12000)
    build_cache()


# ---------- terrain ----------
BASE_Y = 380


def difficulty(x):
    """Starts at the map's d0, climbs to 1.0 over the map's span, then stays brutal."""
    return M["d0"] + (1 - M["d0"]) * np.clip(np.asarray(x, dtype=np.float64) / M["span"], 0.0, 1.0)


def get_terrain_height(x):
    """Endless, never-repeating hills. Hills get taller, bumpier and steeper with distance,
    and sharp ramp-like ridges (jump kickers) appear after the first 150 m."""
    x = np.asarray(x, dtype=np.float64)
    d = difficulty(x)
    wave1 = np.sin(x * 0.002 + 1.6 * np.sin(x * 0.00037)) * (105 + 100 * d)        # big hills (drifting length)
    wave2 = np.sin(x * 0.007 + 2.0 * np.sin(x * 0.0011)) * (35 + 60 * d)          # medium bumps
    wave3 = np.sin(x * 0.02) * (8 + 9 * d)                                        # roughness
    fade = np.clip((x - M["ridge_at"]) / 1000.0, 0.0, 1.0)                                  # no ridges at the very start
    ridge = np.clip(np.sin(x * 0.0043 + 1.3), 0.0, 1.0) ** 6 * (55 * d) * fade    # steep kicker ramps
    wave4 = np.sin(x * 0.0135 + 1.7 * np.sin(x * 0.0023)) * (18 * d) * fade        # choppy mid-size bumps
    return BASE_Y - (wave1 + wave2 + wave3 + wave4 + ridge)


terrain = np.zeros(0, np.float32)
CHUNK = 6000


def extend_world(upto):
    """Generate more ground (in 6000 px chunks) whenever the rover gets close to the edge."""
    global WORLD, terrain, IDX
    if WORLD >= upto:
        return
    end = int(upto // CHUNK + 1) * CHUNK
    xs = np.arange(WORLD, end, dtype=np.float64)
    terrain = np.concatenate([terrain, get_terrain_height(xs).astype(np.float32)])
    WORLD = end
    IDX = np.arange(WORLD, dtype=np.float32)


extend_world(12000)


def gy(x):
    x = min(max(x, 0), WORLD - 2)
    i = int(x)
    f = x - i
    return terrain[i] * (1 - f) + terrain[i + 1] * f


def slope(x):
    return (gy(x + 6) - gy(x - 6)) / 12


def populate(st, upto):
    """Endlessly lay coins and fuel cans ahead of the rover (fuel gets scarcer with distance)."""
    extend_world(upto + 600)
    while st["nx_coin"] < upto:                      # arcs of coins following the terrain
        x = st["nx_coin"]
        n = int(rng.integers(4, 8))
        for i in range(n):
            cx = x + i * 52
            st["coins"].append((float(cx), gy(cx) - 62 - math.sin(i / (n - 1) * math.pi) * 28))
        st["nx_coin"] = x + n * 52 + int(rng.integers(500, 900))
    while st["nx_can"] < upto:
        x = st["nx_can"]
        st["cans"].append((float(x), gy(x) - 58))
        st["nx_can"] = x + 1500 + M["gap"] * (float(difficulty(x)) - M["d0"]) / (1 - M["d0"])


# ---------- input ----------
GAS, BRAKE = ("d", "w", "right"), ("a", "s", "left")


class Keys:
    """True key-held state via pynput if installed; otherwise latched OpenCV keys."""

    def __init__(self):
        self.down, self.t, self.native = set(), {}, False
        try:
            from pynput import keyboard
            nm = lambda k: (k.char.lower() if getattr(k, "char", None) else str(k).replace("Key.", ""))
            self.l = keyboard.Listener(on_press=lambda k: self.down.add(nm(k)),
                                       on_release=lambda k: self.down.discard(nm(k)))
            self.l.daemon = True
            self.l.start()
            self.native = True
        except Exception:
            print("Tip: pip install pynput for smoother held-key controls")

    def feed(self, k):
        if 0 <= k < 255:
            ch = chr(k).lower()
            self.t[ch] = time.time()
            for grp, opp in ((GAS, BRAKE), (BRAKE, GAS)):
                if ch in grp:
                    for o in opp:
                        self.t.pop(o, None)

    def axis(self, names):
        if self.native:
            return 1.0 if any(n in self.down for n in names) else 0.0
        now = time.time()
        return 1.0 if any(now - self.t.get(n, 0) < 0.6 for n in names) else 0.0


class Gestures:
    """Webcam + MediaPipe hands in a background thread; the game just reads the latest state."""
    name = "HAND-MP"

    def __init__(self):
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError("this mediapipe build has no legacy 'solutions' API (try mediapipe==0.10.14)")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("webcam not found")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0,
                                              min_detection_confidence=0.6,
                                              min_tracking_confidence=0.5)
        self.draw, self.conn = mp.solutions.drawing_utils, mp.solutions.hands.HAND_CONNECTIONS
        self.hist = deque(maxlen=5)          # majority vote -> no flicker
        self.offset = 0.0                    # tilt calibration (degrees)
        self.raw_ang = 0.0
        self.out = (0.0, 0.0, 0.0, None, "no hand", 0.0)
        self.run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def calibrate(self):
        self.offset = self.raw_ang

    def stop(self):
        self.run = False
        time.sleep(0.1)
        self.cap.release()

    def _loop(self):
        tilt_s = 0.0
        while self.run:
            ok, frame = self.cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            state, tilt = "none", 0.0
            if res.multi_hand_landmarks:
                h = res.multi_hand_landmarks[0]
                lm = h.landmark
                self.draw.draw_landmarks(frame, h, self.conn)
                # fingers up: index, middle, ring, pinky (tip above knuckle)
                f = [lm[t].y < lm[p].y for t, p in zip([8, 12, 16, 20], [6, 10, 14, 18])]
                if all(f):
                    state = "gas"
                elif not any(f):
                    state = "brake"
                elif f[0] and f[1] and not f[2] and not f[3]:
                    state = "boost"
                # hand roll: wrist -> middle knuckle angle from vertical
                self.raw_ang = math.degrees(math.atan2(lm[9].x - lm[0].x, -(lm[9].y - lm[0].y)))
                tilt = float(np.clip((self.raw_ang - self.offset) / 35, -1, 1))
                tilt = 0.0 if abs(tilt) < 0.2 else tilt
            self.hist.append(state)
            vote = max(set(self.hist), key=list(self.hist).count)
            tilt_s += (tilt - tilt_s) * 0.3
            small = cv2.resize(frame, (320, 240))
            cv2.putText(small, vote.upper(), (8, 26), 0, 0.7, (0, 255, 0), 2)
            self.out = (float(vote in ("gas", "boost")), float(vote == "brake"),
                        tilt_s, small, vote, float(vote == "boost"))

    def read(self):
        return self.out


def classify_hand(mask):
    """OpenCV-only gesture reader (palm-centre + finger-ring method).
    1. palm centre = max of the distance transform (largest inscribed circle)
    2. draw a ring ~1.8x palm radius around it; every hand piece crossing the ring is a finger
       (the piece pointing down is the wrist)
    returns (state, tilt_angle_deg|None, contour|None, (px, py, r, fingers))"""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return "none", None, None, None
    c = max(cs, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 3000:
        return "none", None, None, None
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [c], -1, 255, -1)
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    bx, by, bw, bh = cv2.boundingRect(c)
    wgt = (1 - 0.35 * np.clip((np.arange(mask.shape[0]) - by) / max(bh, 1), 0, 1))[:, None].astype(np.float32)
    _, _, _, (px, py) = cv2.minMaxLoc(dist * wgt)          # prefer the palm over the forearm
    r = float(dist[py, px])
    if r < 12:
        return "none", None, c, None

    ring = np.zeros_like(mask)
    cv2.circle(ring, (px, py), int(1.8 * r), 255, max(3, int(0.15 * r)))
    pieces, _ = cv2.findContours(cv2.bitwise_and(ring, filled), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fingers, wrist = [], None
    for q in pieces:
        x, y, w, h = cv2.boundingRect(q)
        if max(w, h) < 0.3 * r:
            continue                                   # tiny noise
        M = cv2.moments(q)
        if M["m00"] == 0:
            continue
        qx, qy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        if qy > py + 0.6 * r:
            wrist = (qx, qy)
        else:
            fingers.append((qx, qy))
    n = len(fingers)
    hand_area = float(np.count_nonzero(filled[:int(py + 1.3 * r)]))   # ignore forearm
    ratio = hand_area / (math.pi * r * r)                   # fist ~1-2, open hand much larger

    if n >= 4:
        state = "gas"                                  # open palm
    elif n == 2:
        state = "boost"                                # index + middle
    elif n <= 1 and ratio < 2.4:
        state = "brake"                                # fist (thumb may poke out)
    else:
        state = "none"

    ang = None
    if n >= 2:
        mx = sum(f[0] for f in fingers) / n
        my = sum(f[1] for f in fingers) / n
        ang = math.degrees(math.atan2(mx - px, -(my - py)))
    elif wrist is not None:
        ang = math.degrees(math.atan2(px - wrist[0], -(py - wrist[1])))
    return state, ang, c, (px, py, r, n)


class CVGestures:
    """OpenCV-only hand control with guided set-up (so walls / face / arm are ignored):
       STEP 1  take your hand OUT of the green box  -> learns the empty background
       STEP 2  show an OPEN PALM in the box         -> learns your skin colour + neutral tilt
    Then: hand = (differs from background) AND (skin coloured). Press C to redo the set-up."""
    name = "HAND-CV"

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("webcam not found")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.hist = deque(maxlen=7)
        self.offset, self.raw_ang, self._zero = 0.0, 0.0, False
        self.k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.out = (0.0, 0.0, 0.0, None, "no hand", 0.0)
        self.status = ""
        self.restart()
        self.run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def restart(self):
        self.stage, self.t_stage = "bg", time.time()
        self.bg_f, self.bg_n, self.big = None, 0, 0
        self.lo = np.array([0, 133, 77], np.uint8)
        self.hi = np.array([255, 173, 127], np.uint8)
        self.hist.clear()

    calibrate = restart

    def stop(self):
        self.run = False
        time.sleep(0.1)
        self.cap.release()

    def _loop(self):
        while self.run:
            ok, frame = self.cap.read()
            if ok:
                self.out = self.process(cv2.flip(frame, 1))

    def process(self, frame):
        fh, fw = frame.shape[:2]
        x0, y0, x1, y1 = int(fw * 0.5), int(fh * 0.1), int(fw * 0.98), int(fh * 0.9)
        roi = frame[y0:y1, x0:x1]
        blur = cv2.GaussianBlur(roi, (7, 7), 0)
        now, el = time.time(), time.time() - self.t_stage
        mask, state, tilt_in, cnt, palm, fg = None, "none", None, None, None, None

        if self.stage == "bg":
            self.status = "STEP 1/2: take your hand OUT of the green box"
            if el > 1.0:                                              # average several empty frames
                if self.bg_f is None:
                    self.bg_f = blur.astype(np.float32)
                else:
                    cv2.accumulateWeighted(blur, self.bg_f, 0.1)
                self.bg_n += 1
            if el > 2.6 and self.bg_n >= 8:
                self.stage, self.t_stage = "skin", now
        else:
            bg8 = cv2.convertScaleAbs(self.bg_f)
            d = cv2.absdiff(blur, bg8).astype(np.int16)
            fg = ((d[:, :, 0] + d[:, :, 1] + d[:, :, 2]) > 42).astype(np.uint8) * 255
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k5)
            ycc = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
            if self.stage == "skin":
                self.status = "STEP 2/2: show an OPEN PALM in the box"
                mask = fg
                if el > 1.5:
                    ph, pw = fg.shape
                    sel = np.zeros_like(fg)
                    sel[ph // 2 - 70:ph // 2 + 70, pw // 2 - 60:pw // 2 + 60] = 255
                    pix = ycc[(sel > 0) & (fg > 0)]
                    if len(pix) > 800:                                # hand really is there
                        m = np.median(pix, axis=0)
                        mad = np.median(np.abs(pix - m), axis=0)
                        sp = np.clip(2.5 * mad[1:], 14, 30)
                        self.lo = np.array([0, max(m[1] - sp[0], 0), max(m[2] - sp[1], 0)], np.uint8)
                        self.hi = np.array([255, min(m[1] + sp[0], 255), min(m[2] + sp[1], 255)], np.uint8)
                        self.stage, self._zero = "run", True
                        self.status = ""
            else:
                skin = cv2.bitwise_and(cv2.inRange(ycc, self.lo, self.hi),
                                       cv2.inRange(cv2.cvtColor(blur, cv2.COLOR_BGR2HSV), (0, 15, 30), (180, 255, 255)))
                mask = cv2.bitwise_and(fg, skin)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k5)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k9, iterations=2)
                # keep the empty-background model fresh away from the hand
                cv2.accumulateWeighted(blur, self.bg_f, 0.02, cv2.bitwise_not(cv2.dilate(fg, self.k9, iterations=2)))
                self.big = self.big + 1 if fg.mean() / 255 > 0.55 else 0   # camera moved / lights changed
                if self.big > 45:
                    self.restart()
                state, tilt_in, cnt, palm = classify_hand(mask)

        tilt = 0.0
        if tilt_in is not None:
            self.raw_ang = tilt_in
            if self._zero:
                self.offset, self._zero = tilt_in, False
            tilt = float(np.clip((tilt_in - self.offset) / 35, -1, 1))
            tilt = 0.0 if abs(tilt) < 0.2 else tilt
        self.hist.append(state)
        vote = max(set(self.hist), key=list(self.hist).count)
        self.tilt_s = getattr(self, "tilt_s", 0.0) + (tilt - getattr(self, "tilt_s", 0.0)) * 0.3

        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        sub = frame[y0:y1, x0:x1]
        if cnt is not None:
            cv2.drawContours(sub, [cnt], -1, (0, 200, 255), 2)
        if palm is not None:
            px, py, r, n = palm
            cv2.circle(sub, (px, py), int(r), (255, 120, 0), 2)
            cv2.circle(sub, (px, py), int(1.8 * r), (255, 255, 0), 1)
            cv2.putText(sub, f"fingers:{n}", (8, 22), 0, 0.6, (0, 255, 255), 2)
        small = cv2.resize(frame, (320, 240))
        if mask is not None:
            small[165:240, 0:100] = cv2.cvtColor(cv2.resize(mask, (100, 75)), cv2.COLOR_GRAY2BGR)
        label = vote.upper() if self.stage == "run" else "SET-UP"
        cv2.putText(small, label, (8, 26), 0, 0.7, (0, 255, 0), 2)
        gas = float(vote in ("gas", "boost") and self.stage == "run")
        return (gas, float(vote == "brake" and self.stage == "run"), self.tilt_s, small, label.lower(),
                float(vote == "boost" and self.stage == "run"))

    def read(self):
        return self.out


# ---------- sound (synthesised, no asset files) ----------
SR = 22050


def _i16(x):
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def _tone(f, d, decay=8.0, vol=0.5):
    t = np.arange(int(SR * d)) / SR
    return vol * np.sin(2 * np.pi * f * t) * np.exp(-decay * t)


def _sweep(f0, f1, d, vol=0.4):
    t = np.arange(int(SR * d)) / SR
    f = f0 + (f1 - f0) * t / d
    return vol * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, (d - t) * 20) * np.minimum(1, t * 60)


def _periodic_noise(lo, hi, vol):
    spec = np.zeros(SR // 2 + 1, complex)
    k = np.arange(lo, hi)
    spec[lo:hi] = (np.random.default_rng(1).normal(size=len(k)) + 1j * np.random.default_rng(2).normal(size=len(k))) / np.sqrt(k)
    n = np.fft.irfft(spec, SR)
    return vol * n / np.abs(n).max()


def make_sounds():
    t1 = np.arange(SR) / SR
    saw = lambda f: sum(np.sin(2 * np.pi * f * h * t1) / h for h in range(1, 9))
    coin = np.pad(_tone(988, 0.07, 6), (0, int(SR * 0.3)))
    coin[int(SR * 0.06):int(SR * 0.06) + int(SR * 0.3)] += _tone(1319, 0.3, 9)[:int(SR * 0.3)]
    n = np.random.default_rng(3).normal(size=int(SR * 0.4))
    tt = np.arange(len(n)) / SR
    land = 0.7 * np.sin(2 * np.pi * 75 * tt) * np.exp(-14 * tt) + 0.25 * n * np.exp(-30 * tt)
    tc = np.arange(int(SR * 1.0)) / SR
    crash = (0.6 * np.random.default_rng(4).normal(size=len(tc)) * np.exp(-5 * tc)
             + 0.5 * np.sin(2 * np.pi * 90 * tc) * np.exp(-6 * tc) + _sweep(300, 40, 1.0, 0.3))
    return {
        "coin": _i16(coin * 0.8),
        "fuel": _i16(_sweep(380, 1100, 0.35, 0.45)),
        "land": _i16(land),
        "crash": _i16(crash),
        "beep": _i16(_tone(880, 0.12, 5, 0.35)),
        "idle": _i16(0.22 * saw(46) * (0.85 + 0.15 * np.sin(2 * np.pi * 4 * t1))),
        "rev": _i16(0.22 * saw(92) + 0.08 * np.sin(2 * np.pi * 400 * t1)),
        "boost": _i16(_periodic_noise(80, 2500, 0.5)),
    }


class Audio:
    """pygame mixer (best) -> winsound one-shots (Windows fallback) -> silent."""

    def __init__(self):
        self.mode, self.muted, self.snd, self.ch = None, False, {}, {}
        try:
            os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
            import pygame
            pygame.mixer.pre_init(SR, -16, 1, 512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(12)
            self.pg = pygame
            for k, v in make_sounds().items():
                self.snd[k] = pygame.mixer.Sound(buffer=v.tobytes())
            for i, k in enumerate(("idle", "rev", "boost")):
                self.ch[k] = pygame.mixer.Channel(i)
                self.ch[k].play(self.snd[k], loops=-1)
                self.ch[k].set_volume(0)
            self.mode = "pygame"
        except Exception:
            try:
                import winsound
                self.ws = winsound
                self.wav = {}
                for k in ("coin", "fuel", "land", "crash", "beep"):
                    b = io.BytesIO()
                    with wave.open(b, "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                        w.writeframes(make_sounds()[k].tobytes())
                    self.wav[k] = b.getvalue()
                self.mode = "winsound"
                print("Sound: basic effects only (pip install pygame for engine sound)")
            except Exception:
                print("Sound off (pip install pygame)")

    def play(self, name, vol=1.0):
        if self.muted:
            return
        if self.mode == "pygame":
            self.snd[name].set_volume(vol)
            self.snd[name].play()
        elif self.mode == "winsound":
            try:
                self.ws.PlaySound(self.wav[name], self.ws.SND_MEMORY | self.ws.SND_ASYNC)
            except Exception:
                pass

    def engine(self, thr, speed, boost, air, alive):
        if self.mode != "pygame":
            return
        m = 0.0 if (self.muted or not alive) else 1.0
        sp = min(abs(speed) / 10, 1)
        a = 0.45 if air else 1.0
        self.ch["idle"].set_volume(m * a * (0.35 + 0.25 * thr) * (1 - 0.5 * sp))
        self.ch["rev"].set_volume(m * a * min(1, 0.6 * sp + 0.5 * thr) * (0.15 + 0.85 * sp) * 0.9)
        self.ch["boost"].set_volume(m * boost * 0.35)

    def toggle(self):
        self.muted = not self.muted

    def stop(self):
        if self.mode == "pygame":
            self.pg.mixer.stop()


# ---------- visuals ----------
_SKY = _SOIL = _MASK = None


def planet(img, cx, cy, r, base, bands=()):
    cv2.circle(img, (cx, cy), r, base, -1, LA)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1, LA)
    ov = img.copy()
    for dy, hh, col in bands:
        cv2.rectangle(ov, (cx - r, int(cy + dy * r)), (cx + r, int(cy + (dy + hh) * r)), col, -1)
    cv2.copyTo(ov, m, img)
    cv2.circle(img, (cx, cy), r, tuple(int(c * 0.6) for c in base), 2, LA)


def build_cache():
    global _SKY, _SOIL, _MASK
    y = np.linspace(0, 1, H)[:, None, None]
    top, mid, hor = (np.array(c, np.float64) for c in M["sky"])
    k = np.clip(y / 0.55, 0, 1)
    sky = np.where(y < 0.55, top * (1 - k) + mid * k, mid * (1 - (y - 0.55) / 0.45) + hor * ((y - 0.55) / 0.45))
    _SKY = np.repeat(sky, W, axis=1).astype(np.uint8)
    r = np.random.default_rng(11)
    for _ in range(M["stars"]):
        x, yy = int(r.integers(0, W)), int(r.integers(0, int(H * 0.32)))
        fade = max(0.25, 1 - yy / (H * 0.32)) * r.uniform(0.6, 1.0)
        col = tuple(int(v + (255 - v) * fade) for v in _SKY[yy, x])
        cv2.circle(_SKY, (x, yy), int(r.integers(0, 2)), col, -1, LA)
    sg, sc, sr = M["sun"]
    glow = np.zeros((H, W, 3), np.float32)
    cv2.circle(glow, (W - 220, 150), 70, sg, -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 55)
    _SKY = np.clip(_SKY.astype(np.float32) + glow, 0, 255).astype(np.uint8)
    cv2.circle(_SKY, (W - 220, 150), sr, sc, -1, LA)
    for kind, bx, by, br in M["bodies"]:
        if kind == "dot":
            cv2.circle(_SKY, (bx, by), br, (150, 150, 165), -1, LA)
        else:
            planet(_SKY, bx, by, br, *BODY[kind])
    ys = np.arange(H, dtype=np.float32)[:, None]
    f = np.clip((ys - 250) / (H - 250), 0, 1)
    c0, c1 = (np.array(c, np.float32) for c in M["soil"])
    soil = c0 * (1 - f[..., None]) + c1 * f[..., None] + 4 * np.sin(ys / 7)[..., None]
    _SOIL = np.repeat(soil, W, axis=1).clip(0, 255).astype(np.uint8)
    _MASK = np.zeros((H, W), np.uint8)


def draw_background(img, cam):
    img[:] = _SKY
    xs = np.arange(0, W + 12, 12, dtype=np.float32)
    for (par, base, amp), col in zip(((0.06, 0.58, 80), (0.15, 0.66, 60), (0.30, 0.75, 42)), M["mtn"]):
        x = xs + cam * par
        y = H * base - amp * (0.5 * np.sin(x / 310) + 0.3 * np.sin(x / 130 + 1) + 0.2 * np.sin(x / 57 + 2))
        pts = np.vstack([[0, H], np.stack([xs, y], 1), [W + 12, H]]).astype(np.int32)
        cv2.fillPoly(img, [pts], col, LA)


def draw_ground(img, cam, cy):
    xs = np.arange(0, W + 2, 2, dtype=np.float32)
    ys = np.interp(cam + xs, IDX, terrain) - cy
    top = np.stack([xs, ys], 1).astype(np.int32)
    _MASK.fill(0)
    cv2.fillPoly(_MASK, [np.vstack([[0, H + 5], top, [W + 2, H + 5]]).astype(np.int32)], 255)
    cv2.copyTo(_SOIL, _MASK, img)
    for k in range(1, 6):
        lc = tuple(int(c * (0.7 - 0.1 * k)) for c in M["soil"][0])
        cv2.polylines(img, [top + [0, 42 * k]], False, lc, 1, LA)
    cv2.polylines(img, [top + [0, 2]], False, M["crust"][1], 5, LA)
    cv2.polylines(img, [top], False, M["crust"][0], 3, LA)


def make_card(m, cw=400, ch=300):
    t = np.linspace(0, 1, ch)[:, None, None]
    top, mid, hor = (np.array(c, np.float32) for c in m["sky"])
    img = np.repeat(np.where(t < 0.5, top * (1 - 2 * t) + mid * 2 * t, mid * (2 - 2 * t) + hor * (2 * t - 1)),
                    cw, axis=1).astype(np.uint8)
    s = cw / W
    for kind, bx, by, br in m["bodies"]:
        if kind != "dot":
            planet(img, int(bx * s), int(by * s), max(int(br * s), 4), *BODY[kind])
    xs = np.arange(0, cw + 8, 8)
    ys = ch * 0.7 - 26 * np.sin(xs / 55.0) - 12 * np.sin(xs / 21.0 + 1)
    pts = np.vstack([[0, ch], np.stack([xs, ys], 1), [cw + 8, ch]]).astype(np.int32)
    cv2.fillPoly(img, [pts], m["soil"][0], LA)
    cv2.polylines(img, [np.stack([xs, ys], 1).astype(np.int32)], False, m["crust"][0], 3, LA)
    return img


def show(img):
    """Letterboxed fullscreen display (keeps the 16:9 game picture on any monitor)."""
    try:
        _, _, ww, wh = cv2.getWindowImageRect(WIN)
    except Exception:
        ww = wh = 0
    if ww > 0 and wh > 0 and (ww, wh) != (W, H):
        s = min(ww / W, wh / H)
        nw, nh = int(W * s), int(H * s)
        out = np.zeros((wh, ww, 3), np.uint8)
        oy, ox = (wh - nh) // 2, (ww - nw) // 2
        out[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh))
        img = out
    cv2.imshow(WIN, img)


def choose_map(cur):
    """Map select screen. Returns map index, or None on ESC."""
    cards = [make_card(m) for m in MAPS]
    sel, cw, ch, gap = cur, 400, 300, 40
    x0 = (W - 3 * cw - 2 * gap) // 2
    while True:
        img = np.full((H, W, 3), (28, 20, 18), np.uint8)
        for s_, y_, sz, col in (("SELECT MAP", 110, 1.6, (60, 215, 255)),
                                ("A / D or 1-3 = choose     ENTER = go     ESC = cancel", 160, 0.6, (190, 190, 205))):
            th = 2 if sz > 1 else 1
            (tw, _), _ = cv2.getTextSize(s_, cv2.FONT_HERSHEY_DUPLEX, sz, th)
            text(img, s_, (W // 2 - tw // 2, y_), sz, col, th)
        for i, m in enumerate(MAPS):
            x, y = x0 + i * (cw + gap), 220
            img[y:y + ch, x:x + cw] = cards[i]
            on = i == sel
            cv2.rectangle(img, (x - 4, y - 4), (x + cw + 4, y + ch + 150), (60, 215, 255) if on else (110, 110, 125),
                          3 if on else 1, LA)
            text(img, f"{i + 1}. {m['name']}", (x + 16, y + ch + 40), 0.9, (255, 255, 255), 2)
            text(img, m["sub"], (x + 16, y + ch + 72), 0.5, (190, 190, 205))
            text(img, "GRAVITY: " + m["gtxt"], (x + 16, y + ch + 100), 0.5, (190, 190, 205))
            text(img, "DIFFICULTY", (x + 16, y + ch + 130), 0.5, (190, 190, 205))
            for j in range(3):
                cv2.rectangle(img, (x + 140 + j * 26, y + ch + 116), (x + 162 + j * 26, y + ch + 132),
                              (60, 215, 255) if j < m["lvl"] else (65, 60, 70), -1)
        show(img)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            return None
        if k in (13, 10):
            return sel
        if k == ord("a"):
            sel = (sel - 1) % 3
        elif k == ord("d"):
            sel = (sel + 1) % 3
        elif ord("1") <= k <= ord("3"):
            sel = k - ord("1")


def aa_poly(img, pts, fill, edge=None, th=1):
    p = np.array(pts, np.int32)
    cv2.fillPoly(img, [p], fill, LA)
    if edge is not None:
        cv2.polylines(img, [p], True, edge, th, LA)


def alpha_poly(img, pts, color, alpha):
    p = np.array(pts, np.int32)
    x, y, w, h = cv2.boundingRect(p)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    ov = roi.copy()
    cv2.fillPoly(ov, [p - [x0, y0]], color, LA)
    cv2.addWeighted(ov, alpha, roi, 1 - alpha, 0, roi)


def draw_coin(img, sx, sy, spin, r=13):
    w = max(int(abs(math.cos(spin)) * r), 2)
    cv2.ellipse(img, (sx, sy + 2), (w + 1, r + 1), 0, 0, 360, (25, 70, 120), -1, LA)        # soft shadow
    cv2.ellipse(img, (sx, sy), (w + 1, r + 1), 0, 0, 360, (20, 130, 210), -1, LA)           # rim
    cv2.ellipse(img, (sx, sy), (w, r), 0, 0, 360, (45, 205, 255), -1, LA)                   # face
    if w > 5:
        cv2.ellipse(img, (sx, sy), (max(w - 4, 1), r - 4), 0, 0, 360, (30, 165, 240), 1, LA)
        cv2.line(img, (sx, sy - r // 2), (sx, sy + r // 2), (30, 160, 240), 2, LA)
        cv2.line(img, (sx - w // 2, sy - r // 2), (sx - w // 2, sy - r // 4), (215, 248, 255), 2, LA)


def draw_jerrycan(img, cx, cy, s=1.0, color=(50, 50, 215)):
    P = lambda dx, dy: (int(cx + dx * s), int(cy + dy * s))
    t = max(int(2 * s), 1)
    aa_poly(img, [P(-11, -11), P(5, -11), P(11, -5), P(11, 14), P(-11, 14)], color, (20, 20, 120), t)
    cv2.rectangle(img, P(-9, -17), P(-3, -11), (40, 40, 160), -1)
    cv2.rectangle(img, P(-9, -20), P(-3, -17), (40, 210, 255), -1)
    cv2.polylines(img, [np.array([P(-1, -11), P(-1, -17), P(7, -17), P(7, -9)])], False, (20, 20, 120), t, LA)
    cv2.line(img, P(-8, -7), P(-8, 11), (120, 120, 250), t, LA)
    cv2.circle(img, P(1, 5), max(int(4.5 * s), 2), (255, 255, 255), -1, LA)
    aa_poly(img, [P(1, -4), P(-3, 3), P(5, 3)], (255, 255, 255))


def panel(img, x0, y0, x1, y1, alpha=0.55):
    sub = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(sub, 1 - alpha, np.zeros_like(sub), alpha, 0)
    cv2.rectangle(img, (x0, y0), (x1, y1), (205, 205, 215), 1, LA)
    cv2.line(img, (x0, y0), (x1, y0), (60, 190, 255), 2, LA)                    # accent line


def text(img, s, pos, size=0.5, col=(255, 255, 255), th=1):
    cv2.putText(img, s, pos, cv2.FONT_HERSHEY_DUPLEX, size, col, th, LA)


def draw_hud(img, st, fr, speed):
    panel(img, 12, 12, 372, 134)
    draw_coin(img, 42, 46, 0.0, 16)
    text(img, "COINS", (70, 32), 0.42, (190, 190, 205))
    text(img, str(st["coin_count"]), (70, 66), 1.0, (255, 255, 255), 2)
    text(img, "DISTANCE", (200, 32), 0.42, (190, 190, 205))
    text(img, f"{int(st['best_x'] - 200) // 10} m", (200, 66), 1.0, (255, 255, 255), 2)
    f = max(st["fuel"], 0)
    if not (f < 25 and (fr // 15) % 2):
        draw_jerrycan(img, 40, 106, 1.0)
    col = (90, 210, 90) if f > 50 else (40, 190, 250) if f > 25 else (60, 60, 235)
    for i in range(20):
        x = 66 + i * 13
        cv2.rectangle(img, (x, 94), (x + 10, 118), col if f > i * 5 else (65, 60, 70), -1)
    text(img, f"{int(f)}%", (332, 113), 0.45)
    # speed gauge
    panel(img, 12, H - 142, 204, H - 22)
    c, r = (108, H - 56), 62
    cv2.ellipse(img, c, (r, r), 0, 180, 360, (75, 70, 80), 9, LA)
    frac = min(abs(speed) / 13, 1)
    gc = (90, 210, 90) if frac < .6 else (40, 190, 250) if frac < .85 else (60, 60, 235)
    if frac > 0.01:
        cv2.ellipse(img, c, (r, r), 0, 180, 180 + 180 * frac, gc, 9, LA)
    text(img, str(int(abs(speed) * 14)), (c[0] - 26, c[1] - 6), 0.9, (255, 255, 255), 2)
    text(img, "KM/H", (c[0] - 20, c[1] + 16), 0.4, (190, 190, 205))


def draw_rover(img, cam, cy, st, boost, fr):
    wr, xs = 15, (-46, 0, 46)
    ang = st["ang"]
    c, s_ = math.cos(ang), math.sin(ang)
    tg, wxs = [], []
    for wx in xs:
        px = st["x"] + wx * c - COM_H * s_
        py = st["y"] + wx * s_ + COM_H * c
        wxs.append(px)
        tg.append(min(py, gy(px) - wr))               # wheel rests on the ground, hangs freely in the air
    if st["wy"] is None:
        st["wy"] = tg[:]
    for i in range(3):                                   # spring-smoothed wheels
        st["wy"][i] += (tg[i] - st["wy"][i]) * (0.2 if st["air"] else 0.6)
    cen = [(wxs[i] - cam, st["wy"][i] - cy) for i in range(3)]
    mx, my = sum(p[0] for p in cen) / 3, sum(p[1] for p in cen) / 3
    ba = 0.7 * math.atan2(cen[2][1] - cen[0][1], cen[2][0] - cen[0][0]) + 0.3 * ang
    if st["air"] or st["dead"] or abs(ang) > 0.7:          # show the real rotation so flips look (and are) fair
        ba = ang
    bc, bs = math.cos(ba), math.sin(ba)
    B = lambda x, y: (int(mx + x * bc - y * bs), int(my + x * bs + y * bc))

    gyl = gy(st["x"]) - cy                                # ground shadow
    h = max(0.0, gyl - my - 10)
    cv2.ellipse(img, (int(st["x"] - cam), int(gyl + 3)), (int(66 - h * 0.2), 6), math.degrees(math.atan(slope(st["x"]))),
                0, 360, (25, 45, 95), -1, LA)

    if boost > 0.3:                                       # boost flame
        fl = 26 + 14 * ((fr % 3) / 2) * boost
        aa_poly(img, [B(-60, -22), B(-60 - fl, -17), B(-60, -12)], (30, 120, 255))
        aa_poly(img, [B(-60, -20), B(-60 - fl * 0.55, -17), B(-60, -14)], (170, 235, 255))

    for p, q in ((B(0, -8), cen[1]), (B(-30, -8), cen[0]), (B(30, -8), cen[2])):      # rocker-bogie
        cv2.line(img, (int(q[0]), int(q[1])), p, (40, 40, 48), 7, LA)
        cv2.line(img, (int(q[0]), int(q[1])), p, (150, 155, 168), 4, LA)
    for p in cen:                                          # wheels
        p = (int(p[0]), int(p[1]))
        cv2.circle(img, p, wr + 1, (20, 20, 24), -1, LA)
        for k in range(14):
            a_ = st["spin"] + k * math.pi / 7
            cv2.line(img, (int(p[0] + (wr - 3) * math.cos(a_)), int(p[1] + (wr - 3) * math.sin(a_))),
                     (int(p[0] + (wr + 1) * math.cos(a_)), int(p[1] + (wr + 1) * math.sin(a_))), (70, 70, 78), 2, LA)
        cv2.circle(img, p, wr - 5, (150, 155, 168), -1, LA)
        cv2.circle(img, p, wr - 5, (60, 60, 70), 1, LA)
        for k in range(5):
            a_ = st["spin"] + k * 2 * math.pi / 5
            cv2.line(img, p, (int(p[0] + (wr - 6) * math.cos(a_)), int(p[1] + (wr - 6) * math.sin(a_))), (90, 90, 100), 2, LA)
        cv2.circle(img, p, 3, (215, 220, 230), -1, LA)

    aa_poly(img, [B(-62, -22), B(-48, -22), B(-48, -10), B(-62, -10)], (45, 45, 52), (20, 20, 25), 1)   # RTG
    for k in range(3):
        cv2.line(img, B(-60 + k * 5, -22), B(-60 + k * 5, -10), (90, 90, 100), 1, LA)
    aa_poly(img, [B(-54, -9), B(54, -9), B(50, -26), B(-50, -26)], (105, 110, 122), (40, 40, 50), 2)      # chassis
    aa_poly(img, [B(-48, -26), B(48, -26), B(42, -50), B(-32, -50), B(-48, -40)], (238, 240, 245), (80, 80, 92), 2)
    aa_poly(img, [B(-48, -29), B(46, -29), B(45, -35), B(-48, -35)], (30, 140, 255))                        # orange stripe
    aa_poly(img, [B(-48, -40), B(-32, -50), B(42, -50), B(40, -46), B(-30, -46), B(-46, -37)], (255, 255, 255))
    aa_poly(img, [B(16, -29), B(38, -29), B(35, -46), B(16, -46)], (75, 62, 50), (35, 30, 25), 1)          # sensor bay
    cv2.line(img, B(19, -43), B(33, -43), (160, 140, 110), 1, LA)
    for k in range(3):
        cv2.circle(img, B(-36 + k * 12, -19), 2, (60, 210, 255) if (fr // 20 + k) % 3 == 0 else (60, 70, 80), -1, LA)
    # ---- astronaut (seated on the roof, rotates with the body) ----
    SUIT, SUIT_EDGE, DARK = (240, 242, 248), (80, 80, 92), (45, 45, 55)
    bob = max(-3.0, min(3.0, st["vy"] * 0.6))                  # head/torso lag behind the bounce
    lean = 3 if (not st["air"] and st["vx"] > 3) else 0        # leans forward when going fast
    aa_poly(img, [B(-20, -50), B(8, -50), B(8, -55), B(-20, -55)], DARK, (20, 20, 25), 1)                    # seat
    aa_poly(img, [B(-26, -58), B(-16, -58), B(-16, -82 - bob), B(-26, -82 - bob)], (205, 208, 218), SUIT_EDGE, 1)  # backpack
    cv2.line(img, B(-21, -64), B(-21, -76 - bob), (60, 210, 255), 2, LA)                                     # pack light bar
    hip, knee, foot = B(-9, -57), B(9, -61), B(11, -50)                                                      # legs
    for a_, b_ in ((hip, knee), (knee, foot)):
        cv2.line(img, a_, b_, SUIT_EDGE, 9, LA)
        cv2.line(img, a_, b_, SUIT, 6, LA)
    cv2.line(img, foot, B(19, -50), DARK, 5, LA)                                                             # boot
    aa_poly(img, [B(-17, -56), B(-1, -56), B(1 + lean, -81 - bob), B(-17 + lean, -83 - bob)], SUIT, SUIT_EDGE, 1)  # torso
    aa_poly(img, [B(-12, -68 - bob), B(-3, -68 - bob), B(-3, -75 - bob), B(-12, -75 - bob)], (30, 140, 255), (20, 80, 160), 1)  # chest unit
    cv2.circle(img, B(-9, -71 - bob), 1, (255, 255, 255), -1, LA)
    head = B(-6 + lean, -93 - bob)
    cv2.circle(img, head, 13, SUIT_EDGE, -1, LA)                                                             # helmet
    cv2.circle(img, head, 12, SUIT, -1, LA)
    vis = B(-2 + lean, -93 - bob)
    cv2.ellipse(img, vis, (8, 9), math.degrees(ba), 0, 360, (45, 175, 235), -1, LA)                          # gold visor
    cv2.ellipse(img, vis, (8, 9), math.degrees(ba), 0, 360, (20, 90, 140), 1, LA)
    cv2.ellipse(img, B(-4 + lean, -97 - bob), (3, 2), math.degrees(ba), 0, 360, (200, 235, 255), -1, LA)    # visor glint
    sh = B(-5 + lean, -77 - bob)
    if st["air"]:                                                                                            # arm thrown up in the air
        el, hand = B(4, -86 - bob), B(10, -100 - bob)
    else:                                                                                                    # hand on the control stick
        el, hand = B(4, -70 - bob), B(16, -64)
        cv2.line(img, B(16, -50), hand, (40, 40, 48), 4, LA)
        cv2.circle(img, hand, 3, (60, 210, 255), -1, LA)
    for a_, b_ in ((sh, el), (el, hand)):
        cv2.line(img, a_, b_, SUIT_EDGE, 7, LA)
        cv2.line(img, a_, b_, SUIT, 4, LA)
    cv2.circle(img, hand, 4, (190, 195, 205), -1, LA)                                                        # glove
    cv2.line(img, B(26, -50), B(26, -76), (110, 110, 120), 4, LA)                                             # mast
    aa_poly(img, [B(16, -76), B(40, -76), B(40, -88), B(16, -88)], (235, 238, 245), (70, 70, 85), 2)
    cv2.circle(img, B(34, -82), 4, (255, 220, 60), -1, LA)
    cv2.circle(img, B(34, -82), 2, (255, 255, 255), -1, LA)
    cv2.circle(img, B(23, -82), 3, (255, 220, 60), -1, LA)
    cv2.line(img, B(-40, -50), B(-40, -64), (110, 110, 120), 3, LA)                                           # dish
    cv2.ellipse(img, B(-40, -66), (13, 5), math.degrees(ba) - 28, 0, 360, (240, 240, 245), -1, LA)
    cv2.ellipse(img, B(-40, -66), (13, 5), math.degrees(ba) - 28, 0, 360, (110, 110, 125), 1, LA)
    a1, a2, a3 = B(52, -16), B(66, -8), B(76, -18)                                                            # arm
    cv2.line(img, a1, a2, (60, 60, 70), 6, LA)
    cv2.line(img, a2, a3, (60, 60, 70), 5, LA)
    cv2.line(img, a1, a2, (170, 175, 190), 3, LA)
    cv2.line(img, a2, a3, (170, 175, 190), 2, LA)
    cv2.line(img, a3, B(80, -12), (60, 60, 70), 2, LA)
    cv2.line(img, a3, B(82, -22), (60, 60, 70), 2, LA)
    hl = B(54, -28)                                                                                           # headlight
    alpha_poly(img, [hl, B(190, -2), B(190, -60)], (190, 245, 255), 0.13)
    cv2.circle(img, hl, 5, (170, 245, 255), -1, LA)
    cv2.circle(img, hl, 2, (255, 255, 255), -1, LA)


def draw_parts(img, parts, cam, cy):
    for p in parts:
        k = p[4]
        col = (int(75 + 70 * k), int(120 + 80 * k), int(190 + 45 * k))
        cv2.circle(img, (int(p[0] - cam), int(p[1] - cy)), max(int(p[5] * (1 + (1 - k) * 2)), 1), col, -1, LA)


# ---------- game ----------
def new_game():
    st = dict(x=200.0, y=gy(200) - 15 - 22 - 120, vx=0.0, vy=0.0, ang=0.0, w=0.0, spin=0.0, fuel=100.0, thr=0.0,
              coin_count=0, best_x=0.0, air=True, dead=False, reason="", coins=[], cans=[], wy=None,
              nx_coin=900.0, nx_can=1400.0)
    populate(st, 4000)
    return st


COM_H, WR = 22, 15                          # centre of mass sits 22px above the axle line
WHEELS = (-46, 0, 46)                       # wheel x offsets (rocker-bogie, 3 wheels)
HEADS = ((-30, -30), (10, -30), (40, -30))  # roof points: touching ground = flipped = game over
KS, CS, INERTIA, SUB = 0.02, 0.09, 3000.0, 4


def step(st, thr_in, brake, tilt, boost):
    """Hill-Climb-Racing style physics: rigid body + spring/damper wheels.
    Gas pushes the wheels (the body pitches up = wheelies), brake slows/reverses,
    in the air gas/brake/tilt rotate the rover, landing on the roof ends the run."""
    ev = []
    if st["dead"]:
        return ev
    populate(st, st["x"] + 3500)                                      # endless world: keep building ahead
    st["coins"] = [c_ for c_ in st["coins"] if c_[0] > st["x"] - 1500]  # and forget what is far behind
    st["cans"] = [c_ for c_ in st["cans"] if c_[0] > st["x"] - 1500]
    if st["fuel"] <= M["rescue"] and not any(st["x"] + 200 < c_[0] < st["x"] + 1500 for c_ in st["cans"]):
        cx = st["x"] + 1100                                           # fuel at 20% -> place a tank just ahead
        st["cans"].append((float(cx), gy(cx) - 58))
    can = st["fuel"] > 0
    if thr_in and can:
        st["fuel"] -= 0.035 * M["burn"] * (1 + 1.5 * boost)
    thr = thr_in if can else 0.0
    if not st["air"]:                                                 # wheelie limiter: pitch relative to the slope under us
        rel = st["ang"] - math.atan(slope(st["x"]))                   # negative = nose higher than the ground
        if rel < -0.35:
            thr *= max(0.0, 1 - (-rel - 0.35) / 0.45)
    maxv = 10 * (1 + 0.3 * boost)
    dt = 1.0 / SUB
    was_air, touched, impact = st["air"], 0, 0.0

    for _ in range(SUB):
        c, s_ = math.cos(st["ang"]), math.sin(st["ang"])
        fx, fy, tq, n_touch = 0.0, M["grav"], 0.0, 0
        for wx in WHEELS:
            rx, ry = wx * c - COM_H * s_, wx * s_ + COM_H * c
            px, py = st["x"] + rx, st["y"] + ry
            if px < 0 or px > WORLD - 2:
                continue
            pen = py + WR - gy(px)
            if pen <= 0:
                continue
            pen = min(pen, 28)
            n_touch += 1
            sl = slope(px)
            inv = 1 / math.hypot(1, sl)
            tx, ty, nx, ny = inv, sl * inv, sl * inv, -inv            # ground tangent / normal (up)
            vpx, vpy = st["vx"] - st["w"] * ry, st["vy"] + st["w"] * rx
            vn, vt = vpx * nx + vpy * ny, vpx * tx + vpy * ty
            F = KS * pen - CS * vn + (0.2 * (pen - 12) if pen > 12 else 0.0)   # spring + damper + bump stop
            F = max(F, 0.0)
            D = -0.0015 * vt                                          # rolling resistance
            if thr > 0:
                D += 0.09 * thr * (1 + 0.9 * boost) * max(0.0, 1 - max(vt, 0) / maxv)
            if brake > 0.3:
                D -= 0.05 * brake if vt > 0.2 else (0.04 * brake if vt > -2.5 else 0.0)
            Fx, Fy = F * nx + D * tx, F * ny + D * ty
            fx += Fx
            fy += Fy
            tq += rx * Fy - ry * Fx
            if was_air and not touched:
                impact = max(impact, st["vy"])
            touched += 1
        if n_touch == 0:                                              # in the air: rotate the rover
            tq += tilt * 4.5 + (brake - thr) * 3.0                    # tilt keys/hand, gas = nose up, brake = nose down
        st["vx"] += fx * dt
        st["vy"] += fy * dt
        st["w"] += tq / INERTIA * dt
        st["x"] += st["vx"] * dt
        st["y"] += st["vy"] * dt
        st["ang"] += st["w"] * dt
        c, s_ = math.cos(st["ang"]), math.sin(st["ang"])
        for hx, hy in HEADS:                                          # roof hits the ground -> crash
            px, py = st["x"] + hx * c - hy * s_, st["y"] + hx * s_ + hy * c
            if 0 < px < WORLD - 2 and py > gy(px) - 2:
                st["dead"], st["reason"] = True, "CRASHED"
                ev.append(("crash",))
                break
        if st["dead"]:
            break

    st["air"] = touched == 0
    st["w"] *= 0.99 if st["air"] else 0.97
    if not st["air"] and st["w"] < 0:                                 # on the ground, resist rotating nose-up past the slope
        rel = st["ang"] - math.atan(slope(st["x"]))
        if rel < -0.3:
            st["w"] *= 0.82
    if st["x"] < 40:
        st["x"], st["vx"] = 40.0, max(st["vx"], 0.0)
    if was_air and not st["air"] and impact > 2.5 and not st["dead"]:
        ev.append(("land", impact))
    sv = st.setdefault("spv", 0.0)
    st["spv"] = st["vx"] / WR if not st["air"] else sv * 0.995 + thr * 0.01
    st["spin"] += st["spv"]

    ay = st["y"] + COM_H                                              # axle height for pickups
    for c_ in st["coins"][:]:
        if abs(c_[0] - st["x"]) < 36 and abs(c_[1] - ay) < 60:
            st["coins"].remove(c_)
            st["coin_count"] += 1
            ev.append(("coin", c_[0], c_[1]))
    for c_ in st["cans"][:]:
        if abs(c_[0] - st["x"]) < 36 and abs(c_[1] - ay) < 60:
            st["cans"].remove(c_)
            st["fuel"] = min(100, st["fuel"] + 40)
            ev.append(("fuel", c_[0], c_[1]))
    if not st["dead"] and st["fuel"] <= 0 and abs(st["vx"]) < 0.1 and not st["air"]:
        st["dead"], st["reason"] = True, "OUT OF FUEL"
        ev.append(("crash",))
    st["best_x"] = max(st["best_x"], st["x"])
    return ev


# ---------- leaderboard ----------
def score_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f"rover_highscores_{MAPI}.json")
WIN = "Mars Hill Rover"


def load_scores():
    """-> (top-5 list of {name, dist, coins}, last player name). Missing/corrupt file = empty board."""
    try:
        with open(score_path()) as f:
            d = json.load(f)
        sc = [dict(name=str(e["name"])[:12], dist=int(e["dist"]), coins=int(e.get("coins", 0))) for e in d.get("scores", [])]
        sc.sort(key=lambda e: (-e["dist"], -e["coins"]))
        return sc[:5], str(d.get("last", ""))[:12]
    except Exception:
        return [], ""


def save_scores(scores, last):
    try:
        with open(score_path(), "w") as f:
            json.dump({"last": last, "scores": scores}, f, indent=1)
    except Exception as e:
        print("could not save highscores:", e)


def submit_score(scores, name, dist, coins):
    """One row per player (their best run); only the 5 best players are kept. Returns True if this run is on the board."""
    key = name.lower()
    for e in scores:
        if e["name"].lower() == key:
            if dist > e["dist"]:
                e.update(name=name, dist=dist, coins=coins)
            break
    else:
        if dist > 0:
            scores.append(dict(name=name, dist=dist, coins=coins))
    scores.sort(key=lambda e: (-e["dist"], -e["coins"]))
    del scores[5:]
    return any(e["name"].lower() == key and e["dist"] == dist for e in scores)


def draw_board(img, scores, me="", live=None):
    """Top-5 panel in the top-right corner. `live` = current run distance (shown while playing)."""
    x1, y0 = W - 12, 12
    x0 = x1 - 300
    panel(img, x0, y0, x1, y0 + 176 + (30 if live is not None else 0))
    text(img, "LEADERBOARD", (x0 + 14, y0 + 30), 0.6, (60, 215, 255), 1)
    for i in range(5):
        y = y0 + 62 + i * 26
        if i < len(scores):
            e = scores[i]
            col = (60, 215, 255) if e["name"].lower() == me.lower() else (255, 255, 255)
            ds = f"{e['dist']} m"
            (tw, _), _ = cv2.getTextSize(ds, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            text(img, f"{i + 1}.", (x0 + 14, y), 0.5, col)
            text(img, e["name"], (x0 + 44, y), 0.5, col)
            text(img, ds, (x1 - 14 - tw, y), 0.5, col)
        else:
            text(img, f"{i + 1}.  ---", (x0 + 14, y), 0.5, (120, 120, 135))
    if live is not None:
        y = y0 + 62 + 5 * 26 + 4
        cv2.line(img, (x0 + 12, y - 18), (x1 - 12, y - 18), (110, 110, 125), 1, LA)
        ds = f"{live} m"
        (tw, _), _ = cv2.getTextSize(ds, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        text(img, f"NOW: {me}", (x0 + 14, y), 0.5, (130, 255, 170))
        text(img, ds, (x1 - 14 - tw, y), 0.5, (130, 255, 170))


def ask_name(default, scores):
    """Name-entry screen. Returns the typed name, or None if ESC was pressed."""
    name = default
    bg = np.empty((H, W, 3), np.uint8)
    draw_background(bg, 0)
    while True:
        img = bg.copy()
        draw_board(img, scores, name)
        panel(img, W // 2 - 300, H // 2 - 110, W // 2 + 300, H // 2 + 90)
        for s_, y_, sz, col, th in (("ROVER CLIMB RACING", H // 2 - 62, 1.0, (60, 215, 255), 2),
                                    ("ENTER YOUR NAME", H // 2 - 22, 0.6, (200, 200, 210), 1)):
            (tw, _), _ = cv2.getTextSize(s_, cv2.FONT_HERSHEY_DUPLEX, sz, th)
            text(img, s_, (W // 2 - tw // 2, y_), sz, col, th)
        (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
        text(img, name, (W // 2 - tw // 2, H // 2 + 30), 1.1, (255, 255, 255), 2)
        if int(time.time() * 2) % 2 == 0:
            cv2.line(img, (W // 2 + tw // 2 + 6, H // 2 + 6), (W // 2 + tw // 2 + 6, H // 2 + 36), (60, 215, 255), 3)
        cv2.line(img, (W // 2 - 200, H // 2 + 44), (W // 2 + 200, H // 2 + 44), (150, 150, 165), 1, LA)
        hint = "ENTER = start     ESC = quit"
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        text(img, hint, (W // 2 - tw // 2, H // 2 + 76), 0.5, (170, 170, 185))
        show(img)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            return None
        if k in (13, 10):
            if name.strip():
                return name.strip()
        elif k in (8, 127):
            name = name[:-1]
        elif 32 <= k < 127 and len(name) < 12 and not (k == 32 and not name):
            name += chr(k)


def main():
    g = None
    if "--keys" not in sys.argv:
        if "--cv" not in sys.argv:
            try:
                g = Gestures()
            except Exception as e:
                print(f"MediaPipe unavailable ({e}); trying OpenCV-only hand tracking.")
        if g is None:
            try:
                g = CVGestures()
            except Exception as e:
                print(f"Hand tracking unavailable ({e}); using keyboard.")
        if g:
            print(f"Gesture mode ON [{g.name}]: open palm=gas, fist=brake, two fingers=boost, tilt=rotate, C=calibrate")
    audio, keys = Audio(), Keys()
    throttle = brake = tilt = boost = 0.0
    label, fr, shake = "keyboard", 0, 0.0
    popups, parts = [], []
    prev, was_dead, rng2 = time.time(), False, np.random.default_rng(5)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    set_map(0)
    scores, last = load_scores()
    player = ask_name(last, scores)                      # ask who is playing before the first run
    mi = choose_map(0) if player is not None else None
    if mi is None:
        audio.stop()
        if g:
            g.stop()
        cv2.destroyAllWindows()
        return
    set_map(mi)
    scores, _ = load_scores()
    save_scores(scores, player)
    st = new_game()
    cam, cam_y = st["x"] - CAM_OFFSET, st["y"] - H * 0.6
    prev, run_msg = time.time(), ""

    while True:
        fr += 1
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord("n") and st["dead"]:                 # new player after a crash
            nm = ask_name(player, scores)
            if nm is None:
                break
            player = nm
            save_scores(scores, player)
            k = ord("r")
        if k == ord("b"):                                # change map
            mi = choose_map(MAPI)
            if mi is not None and mi != MAPI:
                set_map(mi)
                scores, _ = load_scores()
                k = ord("r")
        if k == ord("r"):
            run_msg = ""
            st, was_dead, popups, parts = new_game(), False, [], []
            cam, cam_y = st["x"] - CAM_OFFSET, st["y"] - H * 0.6
        if k == ord("m"):
            audio.toggle()
        if k == ord("c") and g:
            g.calibrate()
        preview = None
        if g:
            gas, br, tl, preview, label, bst = g.read()
        else:
            keys.feed(k)
            gas, br = keys.axis(GAS), keys.axis(BRAKE)
            tl = keys.axis(("e", "down")) - keys.axis(("q", "up"))
            bst = keys.axis(("space", " "))
            label = "gas" if gas else "brake" if br else "idle"
        throttle += (gas - throttle) * 0.18
        brake += (br - brake) * 0.25
        tilt += (tl - tilt) * 0.15
        boost += (bst - boost) * 0.1
        throttle = 0.0 if throttle < 0.02 else throttle
        brake = 0.0 if brake < 0.05 else brake

        was_air = st["air"]
        for e in step(st, throttle, brake, tilt, boost):
            if e[0] == "coin":
                audio.play("coin", 0.8)
                popups.append([e[1], e[2], 0, "+1", (60, 215, 255)])
            elif e[0] == "fuel":
                audio.play("fuel", 0.9)
                popups.append([e[1], e[2], 0, "+FUEL", (120, 230, 120)])
            elif e[0] == "land":
                audio.play("land", min(e[1] / 8, 1))
                shake = max(shake, min(e[1] * 1.3, 11))
                for _ in range(10):
                    parts.append([st["x"], gy(st["x"]) - 4, rng2.normal(0, 2), -rng2.random() * 2, 1.0, 5])
            elif e[0] == "crash":
                audio.play("crash")
                shake = 14
        if st["dead"] and not was_dead:                  # run just ended -> record it on the leaderboard
            was_dead = True
            dist = max(int(st["best_x"] - 200) // 10, 0)
            if submit_score(scores, player, dist, st["coin_count"]):
                rank = [e["name"].lower() for e in scores].index(player.lower()) + 1
                run_msg = f"LEADERBOARD #{rank}!"
            save_scores(scores, player)
        if st["fuel"] < 25 and not st["dead"] and fr % 45 == 0:
            audio.play("beep", 0.5)
        audio.engine(throttle, st["vx"], boost, st["air"], not st["dead"])
        if throttle > 0.2 and not st["air"] and abs(st["vx"]) > 1.2 and fr % 2 == 0 and not st["dead"]:
            parts.append([st["x"] - 42, gy(st["x"] - 42) - 6, -st["vx"] * 0.25 + rng2.normal(0, .4), -rng2.random() * 1.2, 1.0, 4])
        for p in parts:
            p[0] += p[2]; p[1] += p[3]; p[3] += 0.02; p[4] -= 0.025
        parts = [p for p in parts if p[4] > 0][-150:]
        for p in popups:
            p[2] += 1
        popups = [p for p in popups if p[2] < 40]

        # smooth camera with look-ahead and shake
        cam += ((st["x"] - CAM_OFFSET + st["vx"] * 14) - cam) * 0.1
        cam_y += ((st["y"] - H * 0.6) - cam_y) * 0.06
        shake *= 0.86
        sx_, sy_ = (rng2.normal(0, shake * 0.4), rng2.normal(0, shake * 0.4)) if shake > 0.3 else (0, 0)
        c_x, c_y = cam + sx_, cam_y + sy_

        img = np.empty((H, W, 3), np.uint8)
        draw_background(img, c_x)
        draw_ground(img, c_x, c_y)
        for i, (x_, y_) in enumerate(st["coins"]):
            sx = int(x_ - c_x)
            if -30 < sx < W + 30:
                draw_coin(img, sx, int(y_ - c_y), fr * 0.12 + x_ * 0.05)
        for x_, y_ in st["cans"]:
            sx = int(x_ - c_x)
            if -30 < sx < W + 30:
                bob = int(4 * math.sin(fr * 0.08 + x_))
                cv2.circle(img, (sx, int(y_ - c_y) + bob), 27 + int(2 * math.sin(fr * 0.1)), (130, 235, 130), 1, LA)
                draw_jerrycan(img, sx, int(y_ - c_y) + bob, 1.1)
        draw_parts(img, parts, c_x, c_y)
        draw_rover(img, c_x, c_y, st, boost, fr)
        for p in popups:
            text(img, p[3], (int(p[0] - c_x) - 14, int(p[1] - c_y) - 18 - p[2]), 0.7, p[4], 2)

        draw_hud(img, st, fr, st["vx"])
        mode = g.name if g else "KEYS"
        text(img, f"{mode}: {label.upper()}" + ("  BOOST" if boost > 0.5 else "") + ("   [MUTED]" if audio.muted else ""),
             (16, 160), 0.55, (130, 255, 170) if boost < 0.5 else (60, 180, 255))
        text(img, f"DANGER {int(float(difficulty(st['x'])) * 100)}%", (16, 184), 0.55, (90, 150, 255))
        text(img, f"MAP: {M['name']}", (16, 208), 0.55, (200, 200, 215))
        if preview is not None:
            ph_, pw_ = preview.shape[:2]
            img[H - ph_:, W - pw_:] = preview
            if getattr(g, "status", ""):
                (tw, _), _ = cv2.getTextSize(g.status, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
                panel(img, W // 2 - tw // 2 - 24, 24, W // 2 + tw // 2 + 24, 84)
                text(img, g.status, (W // 2 - tw // 2, 63), 1.0, (60, 215, 255), 2)
        if fr < 330:
            a = min(1, (330 - fr) / 60)
            hint = ("Open palm = GAS   Fist = BRAKE   2 fingers = BOOST   Tilt hand = rotate in air"
                    if g else "D = gas   A = brake   SPACE = boost   Q/E = tilt in air   M = mute")
            (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            text(img, hint, (W // 2 - tw // 2, H - 43), 0.7, tuple(int(255 * a) for _ in range(3)))
        if st["dead"]:
            for s_, y_, sz, th, col in ((st["reason"] or "GAME OVER", H // 2 - 50, 1.8, 3, (255, 255, 255)),
                                        (run_msg, H // 2 - 120, 1.0, 2, (60, 215, 255)),
                                        (f"{st['coin_count']} coins    {int(st['best_x'] - 200) // 10} m", H // 2 + 10, 1.0, 2, (60, 215, 255)),
                                        ("R = restart     N = new player     B = maps", H // 2 + 70, 0.8, 1, (200, 200, 210))):
                (tw, _), _ = cv2.getTextSize(s_, cv2.FONT_HERSHEY_DUPLEX, sz, th)
                text(img, s_, (W // 2 - tw // 2, y_), sz, col, th)
        draw_board(img, scores, player, None if st["dead"] else max(int(st["best_x"] - 200) // 10, 0))
        show(img)
        dt = time.time() - prev
        if dt < 1 / 60:
            time.sleep(1 / 60 - dt)
        prev = time.time()

    audio.stop()
    if g:
        g.stop()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()