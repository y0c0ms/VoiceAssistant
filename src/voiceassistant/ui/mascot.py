"""mascot.py: the octopus overlay, in one module.

Part 1 is the pixel-art renderer: every frame of the octopus (tentacles, mantle, eyes, the search magnifier, the
writing laptop, the play and gear buttons, the colour wave and the fedora), drawn with pygame into an RGBA canvas.
Part 2 is the Qt widget that shows those frames: the animation timer, the chat box, the settings panel, and the
click and hover handling for the buttons.
"""
import html
import math
import os
import random
import time
from collections import deque

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from voiceassistant.core import config

# ---------------------------------------------------------------------------------------------------------------
# Part 1: the renderer
# ---------------------------------------------------------------------------------------------------------------

W, H = 160, 240                    # canvas cells; H leaves room for tentacle tips under a full box
BOX_X0, BOX_X1 = 24, 136           # text container bounds in canvas cells
BOX_TOP = 60
BOX_MAX_H = 150.0                  # longer answers scroll inside the box
COLOUR_S = 1.6                     # seconds for the colour wave to cross the octopus
HAT_S = 0.6                        # seconds for the fedora to drop on
CHROMA = (255, 0, 255)             # transparent chroma key

HAT_COLORS = {
    "outline": (18, 12, 26),
    "dark": (46, 36, 62),
    "mid": (74, 58, 98),
    "light": (106, 86, 136),
    "band": (235, 165, 48),
    "band_hi": (255, 218, 115),
}
HAT = HAT_COLORS["dark"]           # kept for test compatibility
HAT_DROP = 24.0                    # canvas cells the fedora starts above its resting place

PALETTES = {
    "assistant": {
        "outline": (12, 16, 36),
        "dark": (68, 44, 150),
        "mid": (110, 80, 210),
        "light": (150, 125, 255),
        "cyan": (55, 195, 240),
        "cyan_hi": (135, 252, 255),
        "box_bg": (16, 20, 36),
        "mouth": (26, 22, 64),
        "tongue": (225, 110, 155),
    },
    "specialist": {
        "outline": (12, 16, 36),
        "dark": (170, 100, 30),
        "mid": (232, 160, 60),
        "light": (255, 226, 150),
        "cyan": (255, 200, 90),
        "cyan_hi": (255, 248, 200),
        "box_bg": (16, 20, 36),
        "mouth": (64, 30, 14),
        "tongue": (235, 115, 95),
    },
}
EYE_WHITE = (255, 255, 255)

PROP_COLORS = {
    "metal_dark": (48, 58, 82),
    "metal_mid": (92, 108, 142),
    "metal_hi": (175, 198, 230),
    "wood_dark": (92, 52, 28),
    "wood_hi": (164, 102, 56),
    "glass_tint": (165, 245, 255),
    "screen_bg": (10, 15, 28),
    "code_green": (95, 240, 165),
    "code_amber": (255, 195, 85),
    "code_cyan": (95, 235, 255),
}

PLAY_CENTRE = (62, 66)
GEAR_BUTTON_CENTRE = (98, 66)
BUTTON_R = 7.0


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _ease_out_back(x: float) -> float:
    x = max(0.0, min(1.0, x))
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * ((x - 1.0) ** 3) + c1 * ((x - 1.0) ** 2)


def _circle(surf, color, cx: float, cy: float, r: float) -> None:
    if r <= 0.2:
        return
    pygame.draw.circle(surf, color, (round(cx), round(cy)), max(1, round(r)))


def _fill_poly(surf, color, pts: list) -> None:
    if len(pts) >= 3:
        pygame.draw.polygon(surf, color, pts)


def _tube_band(surf, color, centre: list, normals: list, radii: list, offset: float, half: float) -> None:
    left, right = [], []
    for (cx, cy), (nx, ny), r in zip(centre, normals, radii):
        o = offset * r
        h = half * r
        left.append((cx + nx * (o + h), cy + ny * (o + h)))
        right.append((cx + nx * (o - h), cy + ny * (o - h)))
    pygame.draw.polygon(surf, color, left + right[::-1])


def _tentacle_geom(points: list):
    """The centre line, normals and radii of a tentacle spline: computed once per frame and shared by its layers."""
    if len(points) < 2:
        return None
    dense = []
    for (x0, y0, r0), (x1, y1, r1) in zip(points, points[1:]):
        steps = max(2, int(math.hypot(x1 - x0, y1 - y0) * 3))
        for s in range(steps):
            k = s / steps
            dense.append((x0 + (x1 - x0) * k, y0 + (y1 - y0) * k, r0 + (r1 - r0) * k))
    dense.append(points[-1])
    centre = [(x, y) for x, y, _ in dense]
    radii = [r for _, _, r in dense]
    normals = []
    for i in range(len(dense)):
        a = centre[max(0, i - 1)]
        b = centre[min(len(dense) - 1, i + 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        n = math.hypot(tx, ty) or 1.0
        normals.append((-ty / n, tx / n))
    return centre, normals, radii


def _tentacle(surf, pal: dict, geom, layer: str) -> None:
    """Draws one layer of a tentacle from its geometry (see _tentacle_geom)."""
    if geom is None:
        return
    centre, normals, radii = geom
    if layer == "outline":
        _tube_band(surf, pal["outline"], centre, normals, [r + 1.2 for r in radii], 0.0, 1.0)
    elif layer == "shadow":
        _tube_band(surf, pal["dark"], centre, normals, radii, 0.0, 1.0)
    elif layer == "mid":
        _tube_band(surf, pal["mid"], centre, normals, radii, -0.15, 0.72)
    elif layer == "highlight":
        _tube_band(surf, pal["cyan"], centre, normals, radii, -0.6, 0.14)
        for i in range(2, len(centre) - 2, 5):
            x, y = centre[i]
            nx, ny = normals[i]
            _circle(surf, pal["cyan_hi"], x + nx * 0.25 * radii[i], y + ny * 0.25 * radii[i], 0.9)


def _cubic(p0, p1, p2, p3, r0: float, r1: float, n: int = 16) -> list[tuple[float, float, float]]:
    pts = []
    for i in range(n):
        k = i / (n - 1)
        u = 1.0 - k
        x = u ** 3 * p0[0] + 3 * u * u * k * p1[0] + 3 * u * k * k * p2[0] + k ** 3 * p3[0]
        y = u ** 3 * p0[1] + 3 * u * u * k * p1[1] + 3 * u * k * k * p2[1] + k ** 3 * p3[1]
        pts.append((x, y, r0 + (r1 - r0) * k))
    return pts


def _blend_pts(a: list, b: list, k: float) -> list[tuple[float, float, float]]:
    if k <= 0.0:
        return a
    if k >= 1.0:
        return b
    ek = _smoothstep(k)
    return [
        (p[0] + (q[0] - p[0]) * ek, p[1] + (q[1] - p[1]) * ek, p[2] + (q[2] - p[2]) * ek)
        for p, q in zip(a, b)
    ]


def _right_ear_points(head_cx: float, head_cy: float) -> list[tuple[float, float, float]]:
    return _cubic(
        (head_cx + 12, head_cy + 14),
        (head_cx + 34, head_cy + 4),
        (head_cx + 28, head_cy - 14),
        (head_cx + 16, head_cy - 6),
        4.0, 1.3, n=16
    )


def _gear_points(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    pts = []
    for i in range(48):
        a = 2.0 * math.pi * i / 48.0
        rr = r if (i % 6) < 3 else r * 0.72
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _draw_gear(surf, pal: dict, cx: float, cy: float, r: float) -> None:
    _fill_poly(surf, pal["outline"], _gear_points(cx, cy, r + 1.2))
    _fill_poly(surf, pal["cyan"], _gear_points(cx, cy, r))
    _circle(surf, pal["dark"], cx, cy, r * 0.32)


def _draw_button(surf, pal: dict, centre: tuple[float, float], k: float, icon: str, hover: bool = False) -> None:
    cx, cy = centre
    pulse = 1.08 if hover else 1.0
    r = BUTTON_R * _smoothstep(k) * pulse
    if r <= 0.4:
        return
    # no disc: the icon alone, in the same style as the head gear
    fill_col = pal["cyan_hi"] if hover else pal["cyan"]
    if icon == "play":
        _fill_poly(surf, pal["outline"], [
            (cx - 0.45 * r - 1, cy - 0.65 * r - 1),
            (cx - 0.45 * r - 1, cy + 0.65 * r + 1),
            (cx + 0.75 * r + 1, cy),
        ])
        _fill_poly(surf, fill_col, [
            (cx - 0.45 * r, cy - 0.65 * r),
            (cx - 0.45 * r, cy + 0.65 * r),
            (cx + 0.75 * r, cy),
        ])
    else:
        _draw_gear(surf, pal, cx, cy, r * 0.88)


def _mantle_poly(cx: float, cy: float, scale: float, dx: float, dy: float, tilt: float = 0.0) -> list[tuple[float, float]]:
    pts = []
    ct, st = math.cos(tilt), math.sin(tilt)
    for i in range(72):
        t = 2.0 * math.pi * i / 72.0
        c, s = math.cos(t), math.sin(t)
        lx = 22.0 * c * (1.0 - 0.28 * max(0.0, s)) * scale
        ly = 22.0 * s * (1.0 if s < 0 else 0.85) * scale
        rx = lx * ct - ly * st
        ry = lx * st + ly * ct
        pts.append((cx + dx + rx, cy + dy + ry))
    return pts


def _mantle_folds(cx: float, cy: float) -> list[list[tuple[float, float]]]:
    return [
        [(cx - 15, cy + 1), (cx - 10, cy + 8), (cx - 4, cy + 11)],
        [(cx + 4, cy + 12), (cx + 10, cy + 8), (cx + 15, cy + 1)],
    ]


def _shutter_flash(t: float) -> float:
    """0..1 of the camera's flash: a short flash at the start of every 2.4 s cycle, like a shutter."""
    phase = t % 2.4
    return max(0.0, 1.0 - phase / 0.14) if phase < 0.14 else 0.0


def _draw_camera(surf, pal: dict, cx: float, cy: float, k: float, t: float) -> None:
    """A retro box camera held up in front of the face: viewfinder, lens, and a flash bulb that fires as it takes
    a picture. `k` 0..1 brings it in; `cx, cy` is its centre."""
    ek = _smoothstep(k)
    w, h = 20.0 * ek, 13.0 * ek
    x0, y0 = cx - w / 2, cy - h / 2
    flash = _shutter_flash(t)
    body = pygame.Rect(round(x0), round(y0), max(1, round(w)), max(1, round(h)))
    pygame.draw.rect(surf, pal["outline"], body.inflate(2, 2), border_radius=2)
    pygame.draw.rect(surf, PROP_COLORS["metal_mid"], body, border_radius=1)
    pygame.draw.rect(surf, PROP_COLORS["metal_hi"], pygame.Rect(body.left + 1, body.top + 1, max(1, body.width - 2), 2))
    view = pygame.Rect(round(x0 + w * 0.6), round(y0 - 3 * ek), max(1, round(5 * ek)), max(1, round(3 * ek)))
    pygame.draw.rect(surf, pal["outline"], view.inflate(1, 1))
    pygame.draw.rect(surf, PROP_COLORS["metal_dark"], view)
    bulb_col = tuple(round(a + (b - a) * flash) for a, b in zip((120, 120, 140), (255, 252, 230)))
    pygame.draw.rect(surf, pal["outline"], pygame.Rect(round(x0 + 2), round(y0 - 4 * ek), max(1, round(6 * ek)),
                                                       max(1, round(4 * ek))).inflate(1, 1))
    pygame.draw.rect(surf, bulb_col, pygame.Rect(round(x0 + 2), round(y0 - 4 * ek), max(1, round(6 * ek)),
                                                 max(1, round(4 * ek))))
    lens_x, lens_y = x0 + w * 0.38, cy + 0.5 * ek
    lens_r = 4.4 * ek
    _circle(surf, pal["outline"], lens_x, lens_y, lens_r + 1.2)
    _circle(surf, PROP_COLORS["metal_dark"], lens_x, lens_y, lens_r)
    _circle(surf, PROP_COLORS["glass_tint"], lens_x, lens_y, lens_r * 0.6)
    _circle(surf, EYE_WHITE, lens_x - 0.8, lens_y - 0.8, max(0.6, lens_r * 0.18))
    if flash > 0.02:
        _circle(surf, (255, 250, 225), lens_x, lens_y, 2.5 + 6.0 * flash)


def _draw_magnifying_glass(surf, pal: dict, lens_x: float, lens_y: float, hand_x: float, hand_y: float,
                         k: float, eye_center: tuple[float, float], pupil_offset: tuple[float, float]) -> None:
    """Draws a detailed pixel-art magnifying glass with wooden handle, metallic rim, magnified eye, and glass glint."""
    ek = _smoothstep(k)
    if ek <= 0.05:
        return
    r_outer = 9.2 * ek
    r_glass = 6.6 * ek

    # 1. Handle connecting hand_x, hand_y toward the bottom-right rim of the lens
    dx, dy = hand_x - lens_x, hand_y - lens_y
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    hx0, hy0 = lens_x + ux * (r_outer - 1.0), lens_y + uy * (r_outer - 1.0)
    hx1, hy1 = hand_x + ux * 3.5 * ek, hand_y + uy * 3.5 * ek
    pygame.draw.line(surf, pal["outline"], (round(hx0), round(hy0)), (round(hx1), round(hy1)), max(2, round(4.6 * ek)))
    pygame.draw.line(surf, PROP_COLORS["wood_dark"], (round(hx0), round(hy0)), (round(hx1), round(hy1)), max(1, round(2.6 * ek)))
    pygame.draw.line(surf, PROP_COLORS["wood_hi"], (round(hx0 - uy * 0.6), round(hy0 + ux * 0.6)),
                     (round(hx1 - uy * 0.6), round(hy1 + ux * 0.6)), 1)

    # 2. Metallic Bezel & Rim
    _circle(surf, pal["outline"], lens_x, lens_y, r_outer + 1.2)
    _circle(surf, PROP_COLORS["metal_dark"], lens_x, lens_y, r_outer)
    _circle(surf, PROP_COLORS["metal_mid"], lens_x - 0.5, lens_y - 0.5, r_outer - 1.1)
    _circle(surf, PROP_COLORS["metal_hi"], lens_x - 1.0, lens_y - 1.0, r_outer - 1.8)
    _circle(surf, pal["outline"], lens_x, lens_y, r_glass + 0.9)

    # 3. Glass Interior
    _circle(surf, PROP_COLORS["glass_tint"], lens_x, lens_y, r_glass)
    _circle(surf, EYE_WHITE, lens_x - 0.8, lens_y - 0.8, r_glass * 0.78)

    # 4. Magnified Eye Effect when lens passes over the right eye
    ex, ey = eye_center
    if math.hypot(lens_x - ex, lens_y - ey) < 12.0 and ek > 0.65:
        mag_px = lens_x + (ex - lens_x) * 0.45 + pupil_offset[0] * 1.4
        mag_py = lens_y + (ey - lens_y) * 0.45 + pupil_offset[1] * 1.4
        if math.hypot(mag_px - lens_x, mag_py - lens_y) < r_glass - 2.4:
            _circle(surf, pal["outline"], mag_px, mag_py, 3.2 * ek)
            if 0 <= int(round(mag_px - 1)) < W and 0 <= int(round(mag_py - 1)) < H:
                surf.set_at((int(round(mag_px - 1)), int(round(mag_py - 1))), EYE_WHITE)

    # 5. Crisp Diagonal Glass Glint
    if ek > 0.5:
        gx, gy = int(round(lens_x - 2.5 * ek)), int(round(lens_y - 2.5 * ek))
        pygame.draw.line(surf, EYE_WHITE, (gx - 1, gy + 2), (gx + 2, gy - 1), 1)
        if 0 <= gx + 2 < W and 0 <= gy + 2 < H:
            surf.set_at((gx + 2, gy + 2), EYE_WHITE)


def _draw_laptop(surf, pal: dict, k: float, t: float, bob_y: float) -> None:
    """Draws an angled 3/4 pixel-art laptop perched cleanly on the upper-left shoulder/box ledge."""
    ek = _smoothstep(k)
    if ek <= 0.05:
        return

    # Positioned at x=3..35, y=38..62 so it sits cleanly above the chat box and beside the head
    slide_x = (1.0 - ek) * -22.0
    bx = 3.0 + slide_x
    by = 38.0 + bob_y * 0.8

    # 1. Screen Lid (angled slightly toward the octopus's gaze)
    lid_outer = [
        (bx + 1, by),
        (bx + 26, by + 2),
        (bx + 25, by + 18),
        (bx, by + 16),
    ]
    lid_bezel = [
        (bx + 2, by + 1),
        (bx + 25, by + 3),
        (bx + 24, by + 17),
        (bx + 1, by + 15),
    ]
    screen_poly = [
        (bx + 3.5, by + 2.5),
        (bx + 23.5, by + 4.2),
        (bx + 22.8, by + 15.5),
        (bx + 2.8, by + 14.0),
    ]
    _fill_poly(surf, pal["outline"], lid_outer)
    _fill_poly(surf, PROP_COLORS["metal_mid"], lid_bezel)
    _fill_poly(surf, PROP_COLORS["screen_bg"], screen_poly)

    # 2. Animated Code / Text Lines Typing on Screen
    if ek > 0.55:
        scroll_phase = int(t * 7.0)
        line_specs = [
            (PROP_COLORS["code_cyan"], 13),
            (PROP_COLORS["code_green"], 16),
            (PROP_COLORS["code_amber"], 10),
            (PROP_COLORS["code_cyan"], 15),
        ]
        for row in range(3):
            col, max_len = line_specs[(scroll_phase + row) % len(line_specs)]
            if row == 2:
                prog = (t * 7.0) % 1.0
                length = max(2, int(max_len * prog))
            else:
                length = max_len
            y0 = by + 5.0 + row * 3.3
            x0 = bx + 4.5
            x1 = x0 + length
            y1 = y0 + (length * 0.08)
            pygame.draw.line(surf, col, (round(x0), round(y0)), (round(x1), round(y1)), 1)
            if row == 2 and int(t * 10) % 2 == 0:
                cx_cur, cy_cur = int(round(x1 + 1)), int(round(y1))
                if 0 <= cx_cur < W and 0 <= cy_cur < H:
                    surf.set_at((cx_cur, cy_cur), EYE_WHITE)

    # 3. Angled Keyboard Base
    deck_outer = [
        (bx, by + 16),
        (bx + 25, by + 18),
        (bx + 33, by + 23),
        (bx + 6, by + 21),
    ]
    deck_inner = [
        (bx + 2, by + 17),
        (bx + 24, by + 19),
        (bx + 30, by + 22),
        (bx + 7, by + 20),
    ]
    _fill_poly(surf, pal["outline"], deck_outer)
    _fill_poly(surf, PROP_COLORS["metal_dark"], deck_inner)
    pygame.draw.line(surf, pal["cyan"], (round(bx + 6), round(by + 18.5)), (round(bx + 24), round(by + 20)), 1)
    pygame.draw.line(surf, pal["cyan_hi"], (round(bx + 9), round(by + 20)), (round(bx + 26), round(by + 21.3)), 1)


def _parse_poses(pose) -> tuple[float, float, float]:
    """(search, write, camera) weights. Accepts the dict the widget sends, or a legacy (pose_name, weight) tuple."""
    if isinstance(pose, dict):
        return (float(pose.get("search", 0.0)), float(pose.get("write", 0.0)), float(pose.get("camera", 0.0)))
    if isinstance(pose, (tuple, list)) and len(pose) >= 2:
        name, pk = pose[0], float(pose[1])
        return (pk if name == "search" else 0.0), (pk if name == "write" else 0.0), (pk if name == "camera" else 0.0)
    return 0.0, 0.0, 0.0


def _draw_scene(surf, pal: dict, mood: str, t: float, box_h: float, hearing: float = 0.0,
                pose=("", 0.0), buttons: float = 0.0, hover_btn: str = "", listen_k: float | None = None) -> None:
    """Draws one complete mascot frame in a single agent palette (excluding the specialist hat)."""
    surf.fill(CHROMA)
    listening = mood == "listen"
    speaking = mood == "speak"
    thinking = mood == "think"
    idle = (mood == "idle") and (box_h < 1.0)
    search_k, write_k, camera_k = _parse_poses(pose)
    lk = (1.0 if listening else 0.0) if listen_k is None else max(0.0, min(1.0, float(listen_k)))

    head_cx = 80.0
    float_damp = 1.0 if box_h < 1.0 else 0.65
    bob_y = (math.sin(t * 2.4) * 1.3 + math.sin(t * 1.2) * 0.4) * float_damp
    bob_x = (0.85 * math.sin(t * 1.1) + 0.25 * math.cos(t * 2.2)) * float_damp
    head_tilt = (0.045 * math.sin(t * 1.1 - 0.4)) if idle else 0.0
    head_cy = 38.0 + bob_y

    # --- 1. LEFT UPPER TENTACLE: Blends smoothly between relaxed sway and listening ear-cup ---
    left_relaxed_pts = []
    for i in range(16):
        k = i / 15.0
        wave = math.sin(t * 3.3 + k * 2.7 + 1.1) * 2.3 * k
        px = head_cx - 12.0 - math.sin(k * math.pi * 0.85) * 21.0 + wave
        py = (head_cy + 14.0) - k * 19.0
        left_relaxed_pts.append((px, py, 4.0 * (1.0 - k * 0.68)))

    ear_wiggle = math.sin(t * 6.5) * 1.2 if listening else 0.35
    left_ear_pts = []
    for i in range(16):
        k = i / 15.0
        px = head_cx - 12.0 - math.sin(k * math.pi * 0.92) * 20.0 + ear_wiggle * k
        py = (head_cy + 14.0) - k * 24.0
        left_ear_pts.append((px, py, 4.0 * (1.0 - k * 0.68)))

    listen_pts = _blend_pts(left_relaxed_pts, left_ear_pts, lk)

    # --- 2. RIGHT UPPER TENTACLE: Ambient wave + idle stretch + wake-ear + magnifier ---
    idle_cycle = t % 11.0
    stretch_env = math.sin(math.pi * (idle_cycle - 7.5) / 2.6) if (idle and 7.5 <= idle_cycle <= 10.1) else 0.0

    right_pts = []
    for i in range(16):
        k = i / 15.0
        wave = math.sin(t * 3.6 + k * 2.5) * (2.0 + 3.2 * stretch_env) * k
        lift = -5.5 * stretch_env * (k ** 1.4)
        px = head_cx + 12.0 + math.sin(k * math.pi * 0.85) * 21.0 + wave
        py = (head_cy + 14.0) - k * 21.0 + lift
        right_pts.append((px, py, 4.0 * (1.0 - k * 0.68)))

    if hearing > 0.0:
        ear = _right_ear_points(head_cx, head_cy)
        sway = math.sin(t * 4.2) * 0.8
        ear_animated = [(x + sway * (i / 15.0), y, r) for i, (x, y, r) in enumerate(ear)]
        right_pts = _blend_pts(right_pts, ear_animated, hearing)

    # Magnifying glass pose (right arm holds magnifier up and scans smoothly)
    scan_dx = math.sin(t * 3.1) * 4.8 * _smoothstep(search_k)
    scan_dy = math.cos(t * 2.3) * 2.2 * _smoothstep(search_k)
    lens_x = head_cx + 12.0 + scan_dx
    lens_y = head_cy - 1.5 + scan_dy
    hand_x = lens_x + 7.2
    hand_y = lens_y + 7.2
    if search_k > 0.005:
        search_tgt = _cubic(
            (head_cx + 12.0, head_cy + 14.0),
            (head_cx + 35.0, head_cy + 12.0),
            (hand_x + 12.0, hand_y + 4.0),
            (hand_x, hand_y),
            4.0, 2.2, n=16
        )
        right_pts = _blend_pts(right_pts, search_tgt, search_k)

    # Camera pose (right arm holds a retro camera up in front of the face, then snaps a picture)
    if camera_k > 0.005:
        cam_tgt = _cubic(
            (head_cx + 12.0, head_cy + 14.0),
            (head_cx + 34.0, head_cy + 6.0),
            (head_cx + 30.0, head_cy - 12.0),
            (head_cx + 17.0, head_cy - 8.0),
            4.0, 2.2, n=16
        )
        right_pts = _blend_pts(right_pts, cam_tgt, camera_k)

    # Laptop typing pose (left arm reaches up-left onto the laptop keyboard)
    laptop_base_y = 38.0 + bob_y * 0.8
    if write_k > 0.005:
        tap_l = 1.6 * max(0.0, math.sin(t * 14.0))
        write_tgt = _cubic(
            (head_cx - 12.0, head_cy + 14.0),
            (head_cx - 26.0, head_cy + 20.0),
            (36.0, laptop_base_y + 24.0),
            (15.0, laptop_base_y + 18.5 + tap_l),
            4.0, 2.0, n=16
        )
        listen_pts = _blend_pts(listen_pts, write_tgt, write_k)

    # --- 3. SIDE FRAMING & BOTTOM WIGGLING TENTACLES ---
    box_top = BOX_TOP + bob_y
    box_bottom = box_top + box_h
    frame_left, frame_right = [], []
    steps = 20
    for i in range(steps):
        k = i / (steps - 1)
        arch = math.sin(k * math.pi)
        py = (box_top - 3.0) + k * (box_h + 8.0)
        r = 4.4 * (1.0 - k * 0.55)
        frame_left.append((head_cx - 12.0 - arch * 49.0, py, r))
        frame_right.append((head_cx + 12.0 + arch * 49.0, py, r))

    bottom = []
    for idx, (offset_x, curl) in enumerate([(-26, -1.0), (-9, 0.6), (9, -0.6), (26, 1.0)]):
        pts = []
        root_x, root_y = head_cx + offset_x * 0.5, head_cy + 9.0
        for i in range(14):
            k = i / 13.0
            base = math.sin(k * math.pi * 1.15) * (6.5 * curl)
            primary_wig = math.sin(t * 4.4 + idx * 1.6 + k * 3.0) * (2.6 * k)
            secondary_wig = math.cos(t * 2.1 + idx * 0.9 - k * 2.2) * (1.1 * k)
            ox = head_cx + offset_x + base + primary_wig + secondary_wig
            oy = box_bottom - 2.0 + k * 17.0
            w = _smoothstep(min(1.0, k / 0.3))
            pts.append((
                root_x + (ox - root_x) * w,
                root_y + (oy - root_y) * w,
                3.4 * (1.0 - k * 0.65) + 0.8 * (1.0 - w),
            ))
        bottom.append(pts)

    # Secondary helper tentacle tip tapping the right side of the laptop keyboard
    helper_write_pts = None
    if write_k > 0.25:
        tap_r = 1.6 * max(0.0, math.sin(t * 14.0 + math.pi))
        helper_write_pts = _cubic(
            (head_cx - 14.0, head_cy + 16.0),
            (head_cx - 24.0, head_cy + 24.0),
            (38.0, laptop_base_y + 26.0),
            (25.0, laptop_base_y + 19.5 + tap_r),
            3.2, 1.8, n=12
        )

    # Pass A: Bottom tentacles
    bottom_geom = [_tentacle_geom(tent) for tent in bottom]
    for layer in ["outline", "shadow", "mid", "highlight"]:
        for geom in bottom_geom:
            _tentacle(surf, pal, geom, layer)

    # Pass B: Tapered Mantle Dome & Shading
    breath = 1.0 + 0.013 * math.sin(t * 1.8)
    _fill_poly(surf, pal["outline"], _mantle_poly(head_cx + bob_x, head_cy, 1.06 * breath, 0.0, 0.0, head_tilt))
    _fill_poly(surf, pal["dark"], _mantle_poly(head_cx + bob_x, head_cy, 1.0 * breath, 0.0, 0.0, head_tilt))
    _fill_poly(surf, pal["mid"], _mantle_poly(head_cx + bob_x, head_cy, 0.9 * breath, -1.0, -1.0, head_tilt))
    _circle(surf, pal["light"], head_cx + bob_x - 4.0, head_cy - 7.0, 10.0)
    for fold in _mantle_folds(head_cx + bob_x, head_cy):
        pygame.draw.lines(surf, pal["dark"], False, fold, 1)

    # Cohesive Curved Forehead Specular Highlight
    arc = [
        (head_cx + bob_x - 12.0, head_cy - 6.0, 1.3),
        (head_cx + bob_x - 10.0, head_cy - 11.0, 2.0),
        (head_cx + bob_x - 5.0, head_cy - 14.0, 1.5),
    ]
    for hx, hy, hr in arc:
        _circle(surf, pal["cyan"], hx, hy, hr + 0.6)
    for hx, hy, hr in arc:
        _circle(surf, pal["cyan_hi"], hx, hy, hr * 0.75)

    # Pass C: Expressive Eyes, Brows & Mouth
    phase = t % 8.4
    if 4.10 <= phase <= 4.26:
        lid = math.sin(math.pi * (phase - 4.10) / 0.16)
    elif 7.40 <= phase <= 7.54:
        lid = math.sin(math.pi * (phase - 7.40) / 0.14)
    elif 7.64 <= phase <= 7.78:
        lid = math.sin(math.pi * (phase - 7.64) / 0.14)
    else:
        lid = 0.0

    idle_glance_x, idle_glance_y = 0, 0
    if idle:
        g_phase = t % 9.5
        if 2.2 <= g_phase <= 3.6:
            idle_glance_x = -1
        elif 5.8 <= g_phase <= 7.2:
            idle_glance_x = 1
            idle_glance_y = -1

    if search_k > 0.35:
        pupil_dx, pupil_dy = 1, 0
    elif write_k > 0.35:
        pupil_dx, pupil_dy = -1, 1
    elif listening:
        pupil_dx, pupil_dy = -1, 0
    elif thinking:
        pupil_dx, pupil_dy = 1, -1
    elif box_h >= 1.0:
        pupil_dx, pupil_dy = 0, 1
    else:
        pupil_dx, pupil_dy = idle_glance_x, idle_glance_y

    for side, eye_dx in [(-1, -8), (1, 8)]:
        ex, ey = int(round(head_cx + eye_dx + bob_x)), int(round(head_cy + 2))
        brow_lift = 1 if (listening or speaking or hearing > 0.3) else (-1 if write_k > 0.4 else 0)
        brow_y = ey - 8 - brow_lift
        pygame.draw.line(surf, pal["outline"], (ex - 3, brow_y), (ex + 3, brow_y - side), 1)
        _circle(surf, pal["outline"], ex, ey, 5.2)
        _circle(surf, EYE_WHITE, ex, ey, 4.2)
        _circle(surf, pal["outline"], ex + pupil_dx, ey + pupil_dy, 2.3)
        if 0 <= ex + pupil_dx - 1 < W and 0 <= ey + pupil_dy - 1 < H:
            surf.set_at((ex + pupil_dx - 1, ey + pupil_dy - 1), EYE_WHITE)
        if lid > 0.0:
            rows = int(round(10 * lid))
            for dy in range(-5, -5 + rows + 1):
                hw = int(math.sqrt(max(0.0, 5.2 ** 2 - dy ** 2)))
                pygame.draw.line(surf, pal["mid"], (ex - hw, ey + dy), (ex + hw, ey + dy))
            if rows > 0:
                edge = int(math.sqrt(max(0.0, 5.2 ** 2 - (-5 + rows) ** 2)))
                pygame.draw.line(surf, pal["outline"], (ex - edge, ey - 5 + rows), (ex + edge, ey - 5 + rows), 1)

    # Expressive pixel mouth
    mx, my = int(round(head_cx + bob_x)), int(round(head_cy + 9.5))
    if speaking:
        open_k = 0.5 + 0.5 * math.sin(t * 12.5)           # 0 shut .. 1 wide open, on the speech cadence
        if open_k > 0.3:
            hw = 2.4                                        # half width of the mouth, canvas cells
            hh = 1.0 + 1.9 * open_k                         # half height: opens with the cadence
            mouth_pts = [(mx - hw, my - 0.6), (mx - hw + 0.8, my + hh * 0.7), (mx, my + hh),
                         (mx + hw - 0.8, my + hh * 0.7), (mx + hw, my - 0.6), (mx, my - 0.9)]
            _fill_poly(surf, pal["mouth"], [(round(x), round(y)) for x, y in mouth_pts])
            if open_k > 0.7:                                # the tongue shows at the bottom of a wide mouth
                _circle(surf, pal["tongue"], mx, my + hh * 0.75, 1.1)
        else:
            pygame.draw.line(surf, pal["mouth"], (mx - 2, my + 1), (mx + 2, my + 1), 1)
    else:
        pygame.draw.line(surf, pal["mouth"], (mx - 2, my), (mx, my + 1), 1)
        pygame.draw.line(surf, pal["mouth"], (mx, my + 1), (mx + 2, my), 1)

    # Pass D: Laptop (drawn before typing tentacles so tentacle tips rest on top of the keyboard!)
    if write_k > 0.05:
        _draw_laptop(surf, pal, write_k, t, bob_y)

    # Pass E: Upper, Framing, and Helper Tentacles
    upper = [listen_pts, right_pts]
    if helper_write_pts is not None:
        upper.append(helper_write_pts)
    if box_h >= 1.0:
        upper += [frame_left, frame_right]
    upper_geom = [_tentacle_geom(tent) for tent in upper]
    for layer in ["outline", "shadow", "mid", "highlight"]:
        for geom in upper_geom:
            _tentacle(surf, pal, geom, layer)

    # Pass F: Magnifying Glass
    if search_k > 0.05:
        right_eye_pos = (head_cx + 8.0 + bob_x, head_cy + 2.0)
        _draw_magnifying_glass(
            surf, pal, lens_x, lens_y, hand_x, hand_y, search_k, right_eye_pos, (pupil_dx, pupil_dy)
        )
    if camera_k > 0.05:
        _draw_camera(surf, pal, head_cx + 6.0, head_cy - 11.0, camera_k, t)

    # Pass G: Hover Play & Gear Icons
    if buttons > 0.02:
        _draw_button(surf, pal, PLAY_CENTRE, buttons, "play", hover=(hover_btn == "play"))
        _draw_button(surf, pal, GEAR_BUTTON_CENTRE, buttons, "gear", hover=(hover_btn == "gear"))

    # Pass H: Expanding Chat Container Box
    if box_h >= 1.0:
        rect = pygame.Rect(BOX_X0, int(round(box_top)), BOX_X1 - BOX_X0, int(round(box_h)))
        pygame.draw.rect(surf, pal["outline"], rect.inflate(4, 4), border_radius=6)
        pygame.draw.rect(surf, pal["cyan"], rect.inflate(2, 2), border_radius=5)
        pygame.draw.rect(surf, pal["box_bg"], rect, border_radius=4)


def _solid(rgb: np.ndarray) -> np.ndarray:
    return ~((rgb[..., 0] == CHROMA[0]) & (rgb[..., 1] == CHROMA[1]) & (rgb[..., 2] == CHROMA[2]))


def _scene(agent: str, mood: str, t: float, box_h: float, hearing: float = 0.0,
           pose=("", 0.0), buttons: float = 0.0, hover_btn: str = "",
           listen_k: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    surf = pygame.Surface((W, H))
    _draw_scene(surf, PALETTES[agent], mood, t, box_h, hearing, pose, buttons, hover_btn, listen_k)
    rgb = pygame.surfarray.array3d(surf).astype(np.uint8)
    return rgb, _solid(rgb)


def _ellipse_polygon(cx: float, cy: float, rx: float, ry: float, tilt: float, n: int = 40) -> list:
    c, s = math.cos(tilt), math.sin(tilt)
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x, y = rx * math.cos(a), ry * math.sin(a)
        pts.append((cx + x * c - y * s, cy + x * s + y * c))
    return pts


def _fill_ellipse(surf, color, cx: float, cy: float, rx: float, ry: float, tilt: float) -> None:
    pygame.draw.polygon(surf, color, _ellipse_polygon(cx, cy, rx, ry, tilt))


def _hat(rgb: np.ndarray, solid: np.ndarray, hat: float, t: float) -> None:
    """Detective fedora perched on the dome with bounce-settled drop physics."""
    if hat <= 0.01:
        return
    head_cx = 80.0
    head_cy = 38.0 + math.sin(t * 2.4) * 1.3 + math.sin(t * 1.2) * 0.4
    eased_hat = _ease_out_back(hat) if hat < 0.999 else 1.0
    drop = (1.0 - eased_hat) * -HAT_DROP
    tilt = -0.14 + math.sin(t * 2.8) * 0.025
    hx, hy = head_cx + 1.5, head_cy - 19.5 + drop
    surf = pygame.Surface((W, H))
    surf.fill(CHROMA)
    o, d, m, lt = HAT_COLORS["outline"], HAT_COLORS["dark"], HAT_COLORS["mid"], HAT_COLORS["light"]
    b, bh = HAT_COLORS["band"], HAT_COLORS["band_hi"]
    _fill_ellipse(surf, o, hx, hy + 4.5, 25.5, 6.2, tilt)
    _fill_ellipse(surf, d, hx, hy + 4.5, 24.0, 4.8, tilt)
    _fill_ellipse(surf, o, hx - 0.5, hy - 3.5, 14.5, 10.5, tilt)
    _fill_ellipse(surf, d, hx - 0.5, hy - 3.5, 13.2, 9.2, tilt)
    _fill_ellipse(surf, m, hx - 2.0, hy - 4.5, 10.8, 7.5, tilt)
    _fill_ellipse(surf, lt, hx - 4.0, hy - 6.5, 6.2, 3.8, tilt)
    _fill_ellipse(surf, o, hx - 0.2, hy + 1.2, 13.6, 3.4, tilt)
    _fill_ellipse(surf, b, hx - 0.2, hy + 1.2, 12.6, 2.3, tilt)
    _fill_ellipse(surf, bh, hx - 3.0, hy + 0.8, 6.0, 1.3, tilt)
    _fill_ellipse(surf, o, hx + 0.5, hy + 4.8, 24.2, 5.0, tilt)
    _fill_ellipse(surf, m, hx + 0.5, hy + 4.8, 22.8, 3.6, tilt)
    _fill_ellipse(surf, lt, hx - 2.5, hy + 4.2, 16.5, 2.0, tilt)
    hat_rgb = pygame.surfarray.array3d(surf).astype(np.uint8)
    hat_solid = _solid(hat_rgb)
    rgb[hat_solid] = hat_rgb[hat_solid]
    solid |= hat_solid


def render_frame(agent_from: str, agent_to: str, progress: float, seed: tuple[float, float] | None,
                 hat: float, mood: str, t: float, box_h: float, hearing: float = 0.0,
                 pose=("", 0.0), buttons: float = 0.0, hover_btn: str = "",
                 listen_k: float | None = None) -> np.ndarray:
    """Renders one frame as RGBA (H, W, 4)."""
    box_h = max(0.0, min(BOX_MAX_H, box_h))
    if agent_from == agent_to or progress >= 1.0:
        rgb, solid = _scene(agent_to, mood, t, box_h, hearing, pose, buttons, hover_btn, listen_k)
    elif progress <= 0.0 or seed is None:
        rgb, solid = _scene(agent_from, mood, t, box_h, hearing, pose, buttons, hover_btn, listen_k)
    else:
        rgb_a, solid_a = _scene(agent_from, mood, t, box_h, hearing, pose, buttons, hover_btn, listen_k)
        rgb_b, solid_b = _scene(agent_to, mood, t, box_h, hearing, pose, buttons, hover_btn, listen_k)
        sx, sy = seed
        xs = np.arange(W)[:, None] + 0.5
        ys = np.arange(H)[None, :] + 0.5
        dist = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2)
        maxd = max(math.dist(seed, c) for c in [(10, 0), (150, 0), (10, 140), (150, 140)])
        eased_prog = _smoothstep(progress)
        reached = dist / maxd <= eased_prog
        rgb = np.where(reached[..., None], rgb_b, rgb_a)
        solid = np.where(reached, solid_b, solid_a)
    if hat > 0.0 and (agent_to == "specialist" or agent_from == "specialist"):
        _hat(rgb, solid, hat, t)
    out = np.dstack([rgb, np.where(solid, 255, 0).astype(np.uint8)])
    return np.ascontiguousarray(out.transpose(1, 0, 2))


def timeline(age_s: float, to_specialist: bool) -> dict:
    """Agent switch timeline: colour wave first then fedora on; fedora off first then colour wave back."""
    frm, to = ("assistant", "specialist") if to_specialist else ("specialist", "assistant")
    if to_specialist:
        if age_s < COLOUR_S:
            return {"colour_from": frm, "colour_to": to, "progress": max(0.0, age_s / COLOUR_S), "hat": 0.0}
        return {"colour_from": to, "colour_to": to, "progress": 1.0,
                "hat": min(1.0, (age_s - COLOUR_S) / HAT_S)}
    if age_s < HAT_S:
        return {"colour_from": frm, "colour_to": to, "progress": 0.0, "hat": 1.0 - age_s / HAT_S}
    done = min(1.0, (age_s - HAT_S) / COLOUR_S)
    return {"colour_from": frm, "colour_to": to, "progress": done, "hat": 0.0}


def random_seed_point(rng: random.Random) -> tuple[float, float]:
    a = rng.uniform(0, 2 * math.pi)
    r = math.sqrt(rng.random())
    return (80 + 20 * r * math.cos(a), 40 + 20 * r * math.sin(a))

# ---------------------------------------------------------------------------------------------------------------
# Part 2: the Qt widget
# ---------------------------------------------------------------------------------------------------------------

CELL = 2
COMPACT_SCALE = 0.5            # corner octopus is drawn at half size
FRAME_MS = 33                  # ~30 FPS for fluid spring & wave animations
IDLE_FRAME_MS = 50             # ~20 FPS while only the slow idle drift moves: half the work, same motion
PAD = 8                        # px inside the box
SPRING = 9.5                   # per second: responsive critically-damped box spring
BAR_H = 34                     # px of the recording bars strip at the bottom of the box
LEVELS = 48                    # waveform bars shown
SCROLLBAR_PX = 10              # width reserved for scrollbar inside content width calculation
HEAR_S = 2.6                   # seconds the right arm stays at the ear once the wake name is heard
HEAR_RAMP_S = 0.45             # eased ramp duration up and down
MIN_ACTIVE_H = 24.0            # canvas cells: smallest box while recording or showing turns
BAR_COLOUR = QColor("#5ef0e6")


def _ease(x: float) -> float:
    """Smoothstep: starts and ends slowly so limbs and props never snap."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


POSES = ("search", "write", "camera")     # the moods that take the arm to a prop: magnifier, laptop, camera

QUESTION_STYLE = (
    "background: #282d38; color: #f0f4f8; border: 1px solid #384152; "
    "border-radius: 11px; padding: 6px 10px; font-size: 13px;"
)
ANSWER_STYLE = "color: #e8eaed; background: transparent; font-size: 13px; line-height: 1.35;"


class Mascot(QWidget):
    setup_requested = Signal(str)     # panel requests name setup: "assistant" or "specialist"
    settings_closed = Signal()        # panel closed: compact transition can proceed
    play_requested = Signal()         # play button clicked: start conversation
    gear_requested = Signal()         # gear button clicked in compact corner mode

    def __init__(self, rng: random.Random | None = None):
        super().__init__()
        self.setFixedSize(W * CELL, H * CELL)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.mood = "idle"
        self.agent = "assistant"
        self.t = 0.0
        self.box_h = 0.0
        self.target_h = 0.0
        self._rng = rng or random.Random()
        self._switch = None
        self._hear_at: float | None = None
        self._recording = False

        # Independent smooth weights for every pose & listening posture so transitions cross-fade cleanly
        self._pose_weights: dict[str, float] = {"search": 0.0, "write": 0.0, "camera": 0.0}
        self._listen_k = 0.0
        self._pose_at = 0.0
        self._follow_until = 0.0             # the widget time until which the chat keeps the newest text in view
        self._last_tick: float | None = None  # real time of the last frame, so the animation keeps its speed

        self.compact = False
        self._buttons_k = 0.0
        self._hovering = False
        self._hovered_btn = ""
        self._levels: deque[float] = deque([0.0] * LEVELS, maxlen=LEVELS)
        self._turns = 0
        self._answer_label: QLabel | None = None
        self._answer_html = ""

        self._list = QWidget()
        self._list.setStyleSheet("background: transparent;")
        self._list_layout = QVBoxLayout(self._list)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(8)
        self._list_layout.addStretch(1)

        self._scroll = QScrollArea(self)
        self._scroll.setWidget(self._list)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea { background: transparent; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
            "QScrollBar:vertical { width: 5px; background: transparent; margin: 2px 0; }"
            "QScrollBar::handle:vertical { background: #3e4758; border-radius: 2px; min-height: 24px; }"
            "QScrollBar::handle:vertical:hover { background: #5ef0e6; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }"
        )
        self._scroll.hide()

        self._settings_open = False
        self._build_settings()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(FRAME_MS)

    def hideEvent(self, event) -> None:
        self._timer.stop()           # hidden: nothing is drawn, so no frames are rendered
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        self._timer.start(FRAME_MS)
        super().showEvent(event)

    # --- state from the overlay ---------------------------------------------------------------------
    def set_mood(self, mood: str) -> None:
        self.mood = mood

    def set_agent(self, agent: str) -> None:
        if agent == self.agent and self._switch is None:
            return
        self._switch = (self._displayed_colour(), agent, self.t, random_seed_point(self._rng))
        self.agent = agent

    def hear(self) -> None:
        """Wake name recognised: right arm smoothly cups to the ear."""
        self._hear_at = self.t

    def _hearing(self) -> float:
        if self._hear_at is None:
            return 0.0
        age = self.t - self._hear_at
        if age < 0.0 or age >= HEAR_S:
            return 0.0
        if age < HEAR_RAMP_S:
            return _ease(age / HEAR_RAMP_S)
        if age > HEAR_S - HEAR_RAMP_S:
            return _ease((HEAR_S - age) / HEAR_RAMP_S)
        return 1.0

    def set_compact(self, on: bool) -> None:
        self.compact = bool(on)
        scale = COMPACT_SCALE if self.compact else 1.0
        self.setFixedSize(int(W * CELL * scale), int(H * CELL * scale))
        self._relayout()
        self.update()

    def _build_settings(self) -> None:
        self._settings = QWidget(self)
        self._settings.setStyleSheet(
            "QWidget { color: #eaeef2; font-size: 12px; background: transparent; } "
            "QPushButton { background: #2c3340; border: 1px solid #3f495a; border-radius: 6px; padding: 4px 8px; } "
            "QPushButton:hover { background: #384152; border-color: #5ef0e6; } "
            "QComboBox { background: #2c3340; border: 1px solid #3f495a; border-radius: 4px; padding: 2px 6px; }"
        )
        lay = QVBoxLayout(self._settings)
        lay.setContentsMargins(PAD, PAD, PAD, PAD)
        lay.setSpacing(6)
        title = QLabel("Definições")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #5ef0e6;")
        lay.addWidget(title)
        self._voice = QComboBox()
        self._voice.addItems(["dii", "miro"])
        self._voice.currentTextChanged.connect(self._save_voice)
        self._tempo = QSlider(Qt.Orientation.Horizontal)
        self._tempo.setRange(80, 130)
        self._tempo.valueChanged.connect(self._save_tempo)
        for label, widget in (("Voz", self._voice), ("Velocidade", self._tempo)):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(widget)
            lay.addLayout(row)
        self._folders_lay = QVBoxLayout()
        lay.addLayout(self._folders_lay)
        for target, label in (("assistant", "Refazer o nome do assistente"),
                              ("specialist", "Refazer o nome do especialista")):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, t=target: self.setup_requested.emit(t))
            lay.addWidget(button)
        close = QPushButton("Fechar")
        close.clicked.connect(self.close_settings)
        lay.addWidget(close)
        self._settings.hide()

    def _refresh_settings(self) -> None:
        cfg = config.load()
        pt = cfg["tts"]["pt"]
        self._voice.blockSignals(True)
        self._voice.setCurrentText(pt["voice"])
        self._voice.blockSignals(False)
        self._tempo.blockSignals(True)
        self._tempo.setValue(round(pt["tempo"] * 100))
        self._tempo.blockSignals(False)
        while self._folders_lay.count():
            item = self._folders_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        folders = cfg["access"]["folders"]
        if not folders:
            self._folders_lay.addWidget(QLabel("Nenhuma pasta aprovada."))
        for folder in folders:
            row = QHBoxLayout()
            name = QLabel(folder)
            name.setWordWrap(True)
            remove = QPushButton("Remover")
            remove.clicked.connect(lambda _=False, f=folder: self._remove_folder(f))
            row.addWidget(name, 1)
            row.addWidget(remove)
            holder = QWidget()
            holder.setLayout(row)
            self._folders_lay.addWidget(holder)

    def _save_voice(self, voice: str) -> None:
        cfg = config.load()
        cfg["tts"]["pt"]["voice"] = voice
        config.save(cfg)

    def _save_tempo(self, value: int) -> None:
        cfg = config.load()
        cfg["tts"]["pt"]["tempo"] = round(value / 100, 2)
        config.save(cfg)

    def _remove_folder(self, folder: str) -> None:
        cfg = config.load()
        cfg["access"]["folders"] = [f for f in cfg["access"]["folders"] if f != folder]
        config.save(cfg)
        self._refresh_settings()
        self._relayout()

    def _settings_h(self) -> float:
        px = self._settings.sizeHint().height() + 2 * PAD
        return max(MIN_ACTIVE_H, min(float(BOX_MAX_H), px / CELL))

    def open_settings(self) -> None:
        self._settings_open = True
        self._refresh_settings()
        self._relayout()

    def close_settings(self) -> None:
        self._settings_open = False
        self._relayout()
        self.settings_closed.emit()

    def settings_open(self) -> bool:
        return self._settings_open

    def _in_hover_area(self, pos) -> bool:
        s = CELL * (COMPACT_SCALE if self.compact else 1.0)
        return 26 * s <= pos.x() <= 134 * s and 6 * s <= pos.y() <= 104 * s

    def _hit_button(self, pos) -> str:
        s = CELL * (COMPACT_SCALE if self.compact else 1.0)
        r = (BUTTON_R + 2.5) * s
        px, py = PLAY_CENTRE[0] * s, PLAY_CENTRE[1] * s
        if (pos.x() - px) ** 2 + (pos.y() - py) ** 2 <= r * r:
            return "play"
        gx, gy = GEAR_BUTTON_CENTRE[0] * s, GEAR_BUTTON_CENTRE[1] * s
        if (pos.x() - gx) ** 2 + (pos.y() - gy) ** 2 <= r * r:
            return "gear"
        return ""

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        self._hovering = self._in_hover_area(pos)
        btn = self._hit_button(pos) if (self._buttons_k > 0.35 and self._hovering) else ""
        if btn != self._hovered_btn:
            self._hovered_btn = btn
            self.setCursor(Qt.CursorShape.PointingHandCursor if btn else Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovering = False
        self._hovered_btn = ""
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._buttons_k > 0.45:
            btn = self._hit_button(event.position())
            if btn == "play":
                self.play_requested.emit()
                event.accept()
                return
            if btn == "gear":
                if self.compact:
                    self.gear_requested.emit()
                elif self._settings_open:
                    self.close_settings()
                else:
                    self.open_settings()
                event.accept()
                return
        super().mousePressEvent(event)

    def set_recording(self, on: bool) -> None:
        self._recording = bool(on)
        self._relayout()

    def push_level(self, level: float) -> None:
        self._levels.append(max(0.0, min(100.0, float(level))))

    def add_question(self, text: str) -> None:
        bubble = QLabel(html.escape(text))
        bubble.setWordWrap(True)
        bubble.setStyleSheet(QUESTION_STYLE)
        bubble.setMaximumWidth(int((BOX_X1 - BOX_X0) * CELL * 0.82))
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(bubble)
        self._insert(row_widget)
        self._answer_label = QLabel()
        self._answer_label.setWordWrap(True)
        self._answer_label.setTextFormat(Qt.TextFormat.RichText)
        self._answer_label.setStyleSheet(ANSWER_STYLE)
        self._insert(self._answer_label)
        self._answer_html = ""
        self._turns += 1
        self._relayout()

    def start_answer(self) -> None:
        if self._answer_label is not None:
            self._answer_label.setText("")
        self._answer_html = ""

    def set_answer(self, rich: str) -> None:
        if self._answer_label is None:
            return
        self._answer_label.setText(rich)
        self._answer_html = rich
        self._relayout()

    def set_status(self, text: str) -> None:
        if self._answer_label is None or self._answer_html:
            return
        self._answer_label.setText(f"<i style='color:#9aa6bd'>{html.escape(text)}</i>")
        self._relayout()

    # --- layout -------------------------------------------------------------------------------------
    def _insert(self, widget: QWidget) -> None:
        self._list_layout.insertWidget(self._list_layout.count() - 1, widget)
        QTimer.singleShot(0, self._scroll_to_end)

    def _scroll_to_end(self) -> None:
        self._follow_until = self.t + 0.8    # for the next moments the box grows, the newest text stays in view
        self._snap_to_end()
        QTimer.singleShot(60, self._snap_to_end)

    def _snap_to_end(self) -> None:
        self._list.layout().activate()
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _content_px(self) -> float:
        width = int((BOX_X1 - BOX_X0) * CELL - 2 * PAD - SCROLLBAR_PX)
        total = 0.0
        shown = 0
        for i in range(self._list_layout.count()):
            w = self._list_layout.itemAt(i).widget()
            if w is None:
                continue
            if isinstance(w, QLabel):
                total += w.heightForWidth(width) if w.hasHeightForWidth() else w.sizeHint().height()
            else:
                total += w.layout().totalHeightForWidth(width)
            shown += 1
        return total + max(0, shown - 1) * self._list_layout.spacing()

    def _relayout(self) -> None:
        if self._settings_open:
            self.target_h = self._settings_h()
            return
        if self.compact or (not self._recording and self._turns == 0):
            self.target_h = 0.0
        else:
            px = 2 * PAD
            if self._turns:
                px += self._content_px()
            if self._recording:
                px += BAR_H + PAD
            self.target_h = max(MIN_ACTIVE_H, min(float(BOX_MAX_H), px / CELL))
        self._scroll_to_end()

    def _idle_now(self) -> bool:
        """True when only the slow idle drift is moving: nothing to smooth, so the frames can come less often."""
        return (self.mood == "idle" and not self._recording and self.box_h < 0.5 and self.target_h == 0.0
                and not self._hovering and self._buttons_k < 0.01 and self._switch is None
                and self._listen_k < 0.01 and all(w < 0.01 for w in self._pose_weights.values())
                and (self._hear_at is None or self.t - self._hear_at > HEAR_S))

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(0.1, now - self._last_tick) if self._last_tick is not None else FRAME_MS / 1000.0
        self._last_tick = now
        self.t += dt
        listening_plain = (self._recording and not self._hovering
                           and (self._hear_at is None or self.t - self._hear_at > HEAR_S))
        want = IDLE_FRAME_MS if (self._idle_now() or listening_plain) else FRAME_MS
        if self._timer.interval() != want:
            self._timer.setInterval(want)
        self.box_h += (self.target_h - self.box_h) * min(1.0, dt * SPRING)
        if self._turns and self.t < self._follow_until and self.box_h >= 1.0:
            self._snap_to_end()         # the box is still growing: keep the newest text in view
        if self.box_h < 0.35 and self.target_h == 0.0:
            self.box_h = 0.0
        self._place()
        self.update()

    def _box_rect(self) -> QRectF:
        float_damp = 1.0 if self.box_h < 1.0 else 0.65
        bob = (math.sin(self.t * 2.4) * 1.3 + math.sin(self.t * 1.2) * 0.4) * float_damp
        return QRectF(
            BOX_X0 * CELL,
            (BOX_TOP + bob) * CELL,
            (BOX_X1 - BOX_X0) * CELL,
            self.box_h * CELL,
        )

    def _bars_rect(self) -> QRectF:
        box = self._box_rect()
        return QRectF(box.left() + PAD, box.bottom() - PAD - BAR_H, box.width() - 2 * PAD, BAR_H)

    def _place(self) -> None:
        if self._settings_open:
            box = self._box_rect()
            self._scroll.hide()
            self._settings.setGeometry(
                int(box.left() + PAD),
                int(box.top() + PAD),
                int(box.width() - 2 * PAD),
                int(max(0.0, box.height() - 2 * PAD)),
            )
            self._settings.show()
            return
        self._settings.hide()
        if self.box_h < 1.0 or (self._turns == 0):
            self._scroll.hide()
            return
        box = self._box_rect()
        bottom_reserve = (BAR_H + PAD) if self._recording else 0
        h = box.height() - 2 * PAD - bottom_reserve
        if h <= 0:
            self._scroll.hide()
            return
        self._scroll.setGeometry(
            int(box.left() + PAD),
            int(box.top() + PAD),
            int(box.width() - 2 * PAD),
            int(h),
        )
        self._scroll.show()

    # --- colour switch ------------------------------------------------------------------------------
    def _displayed_colour(self) -> str:
        if self._switch is None:
            return self.agent
        _, to, start, _ = self._switch
        state = timeline(self.t - start, to == "specialist")
        return state["colour_to"] if state["progress"] >= 0.5 else state["colour_from"]

    def _state(self) -> dict:
        if self._switch is None:
            return {
                "colour_from": self.agent,
                "colour_to": self.agent,
                "progress": 1.0,
                "hat": 1.0 if self.agent == "specialist" else 0.0,
                "seed": None,
            }
        _, to, start, seed = self._switch
        age = self.t - start
        if age >= COLOUR_S + HAT_S:
            self._switch = None
            return self._state()
        state = timeline(age, to == "specialist")
        state["seed"] = seed
        return state

    # --- paint --------------------------------------------------------------------------------------
    def paintEvent(self, _event) -> None:
        st = self._state()
        dt = max(0.0, self.t - self._pose_at)
        self._pose_at = self.t

        # Smoothly blend each pose independently so search <-> write <-> idle never snaps
        for name in POSES:
            goal = 1.0 if self.mood == name else 0.0
            self._pose_weights[name] += (goal - self._pose_weights[name]) * min(1.0, dt * 6.0)
            if self._pose_weights[name] < 0.002 and goal == 0.0:
                self._pose_weights[name] = 0.0

        listen_goal = 1.0 if self.mood == "listen" else 0.0
        self._listen_k += (listen_goal - self._listen_k) * min(1.0, dt * 6.5)

        buttons_goal = 1.0 if (self.box_h < 1.0 and self._hovering) else 0.0
        self._buttons_k += (buttons_goal - self._buttons_k) * min(1.0, dt * 9.0)

        frame = render_frame(
            st["colour_from"],
            st["colour_to"],
            st["progress"],
            st["seed"],
            st["hat"],
            self.mood,
            self.t,
            self.box_h,
            self._hearing(),
            self._pose_weights,
            self._buttons_k,
            self._hovered_btn,
            self._listen_k,
        )
        h, w = frame.shape[:2]
        img = QImage(frame.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        p.drawImage(QRectF(0, 0, self.width(), self.height()), img)
        if self._recording and self.box_h >= 1.0:
            self._paint_bars(p)
        p.end()

    def _paint_bars(self, p: QPainter) -> None:
        rect = self._bars_rect()
        p.setPen(Qt.PenStyle.NoPen)
        slot = rect.width() / LEVELS
        for i, level in enumerate(self._levels):
            edge_fade = math.sin(math.pi * (i + 0.5) / LEVELS) ** 0.35
            bar = max(2.0, (level / 100.0) * BAR_H * edge_fade)
            x = rect.left() + i * slot
            col = QColor(BAR_COLOUR)
            col.setAlpha(int(140 + 115 * min(1.0, level / 65.0)))
            p.setBrush(col)
            p.drawRoundedRect(QRectF(x, rect.top() + (BAR_H - bar) / 2, slot * 0.62, bar), 1.5, 1.5)
