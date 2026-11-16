"""overlay_window.py: the overlay window. It is the octopus (mascot.py) and nothing else: no card, no header,
no status line, no wave bar, no sources. The session's signals drive the octopus: its mood (listening shows the
recording bars in the box), its agent (colour and fedora), the question and the answer (the chat list in the
box). Esc closes the window; a drag on the top strip moves it and the position is kept across restarts.
"""
import json
import os

from PySide6.QtCore import QPropertyAnimation, Qt, QTimer
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from mascot import Mascot
from rich_text import render_html
from taskbar import hide_from_taskbar

MARGIN = 24
POS_PATH = os.path.expanduser("~/.config/voiceassistant/overlay.json")   # where the window was dragged to
DRAG_ZONE_PX = 44                                                       # the top strip that drags the window


class Overlay(QWidget):
    def __init__(self, bridge, on_escape):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.on_escape = on_escape
        self._answer = ""
        self._user_pos = self._load_pos()     # where the user dragged the window to, if they did
        self._placing = False

        self.mascot = Mascot()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.mascot)
        self.setFixedSize(self.mascot.size())

        bridge.mood.connect(self._on_mood)
        bridge.agent.connect(self.mascot.set_agent)
        bridge.heard.connect(self.mascot.hear)
        self.mascot.setup_requested.connect(bridge.setup_request.emit)
        self.mascot.settings_closed.connect(self._on_settings_closed)
        self.mascot.play_requested.connect(bridge.start_request.emit)
        self.mascot.gear_requested.connect(self._gear_from_corner)
        self._corner_pending = False
        self._opened_for_settings = False
        bridge.level.connect(self.mascot.push_level)
        bridge.question.connect(self._on_question)
        bridge.answer_reset.connect(self._on_answer_reset)
        bridge.answer.connect(self._on_answer)
        bridge.status.connect(self.mascot.set_status)

    def _on_mood(self, mood: str):
        self.mascot.set_mood(mood)
        self.mascot.set_recording(mood == "listen")

    def _on_question(self, text: str):
        self._answer = ""
        self.mascot.add_question(text)

    def _on_answer_reset(self):
        self._answer = ""
        self.mascot.start_answer()

    def _on_answer(self, delta: str):
        self._answer += delta
        self.mascot.set_answer(render_html(self._answer))

    def _load_pos(self) -> tuple[int, int] | None:
        """The saved position, if it is still on a screen (the monitor may be gone after a restart)."""
        try:
            with open(POS_PATH) as f:
                saved = json.load(f)
            x, y = int(saved["x"]), int(saved["y"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        on_screen = any(s.availableGeometry().contains(x + 20, y + 20) for s in QApplication.screens())
        return (x, y) if on_screen else None

    def _save_pos(self) -> None:
        os.makedirs(os.path.dirname(POS_PATH), exist_ok=True)
        with open(POS_PATH, "w") as f:
            json.dump({"x": self.x(), "y": self.y()}, f)

    def place(self):
        """Top-right of the primary screen, or where the user dragged it."""
        geo = QApplication.primaryScreen().availableGeometry()
        self._placing = True
        try:
            if self._user_pos is not None:
                x = max(geo.left(), min(self._user_pos[0], geo.right() - self.width()))
                y = max(geo.top(), min(self._user_pos[1], geo.bottom() - self.height()))
                self.move(x, y)
            else:
                self.move(geo.right() - self.width() - MARGIN, geo.bottom() - self.height() - MARGIN)
        finally:
            self._placing = False

    def _gear_from_corner(self) -> None:
        """The gear in the corner: the card opens with the settings, and shrinks again when they are closed."""
        self._opened_for_settings = True
        self.expand()
        self.mascot.open_settings()

    def _on_settings_closed(self) -> None:
        if self._corner_pending or self._opened_for_settings:
            self._opened_for_settings = False
            self.to_corner()

    def to_corner(self) -> None:
        """The card has closed: the octopus shrinks and stays in the bottom-right corner until the next call."""
        if self.mascot.settings_open():
            self._corner_pending = True       # the panel is open: the card shrinks once it is closed
            return
        self._corner_pending = False
        self.mascot.set_agent("assistant")   # idle is always the front desk's octopus, never the specialist's
        self.mascot.set_compact(True)
        self.setFixedSize(self.mascot.size())
        geo = QApplication.primaryScreen().availableGeometry()
        self._placing = True
        try:
            self.move(geo.right() - self.width() - MARGIN, geo.bottom() - self.height() - MARGIN)
        finally:
            self._placing = False

    def expand(self) -> None:
        """A new card: the octopus grows back to full size where it was dragged to, or the corner."""
        self.mascot.set_compact(False)
        self.setFixedSize(self.mascot.size())
        self.place()
        self.show()

    def open_settings_view(self) -> None:
        """From the tray menu: the card opens with the settings, and shrinks back when they are closed."""
        self._gear_from_corner()
        self.show()
        self.raise_()

    def show_window(self) -> None:
        """From the application menu: the octopus is shown and brought to the front."""
        self.show()
        self.raise_()
        self.activateWindow()

    def toggle_visible(self) -> None:
        """The tray's left click: the octopus is shown or hidden. The assistant keeps running either way."""
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # after the window is mapped, the window manager is asked to leave it out of the taskbar
        QTimer.singleShot(0, lambda: hide_from_taskbar(int(self.winId())))

    def mousePressEvent(self, e):
        """A press on the top strip hands the drag to the window manager."""
        if e.button() == Qt.MouseButton.LeftButton and e.position().y() <= DRAG_ZONE_PX and self.windowHandle():
            self.windowHandle().startSystemMove()
        super().mousePressEvent(e)

    def moveEvent(self, e):
        """A move the user made (not the window placing itself) is remembered across restarts."""
        super().moveEvent(e)
        if self._placing or not self.isVisible():
            return
        self._user_pos = (self.x(), self.y())
        QTimer.singleShot(400, self._save_pos)

    def fade_out(self, ms: int, then) -> None:
        """Fade the whole window out over `ms`, then call `then`."""
        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(ms)
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(then)
        self._fade.start()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.on_escape()
        else:
            super().keyPressEvent(e)
