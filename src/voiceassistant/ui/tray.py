"""tray.py: the tray icon, in the notification area with the other icons. Left click shows or hides the octopus; the
right-click menu has the settings, a restart and the close."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from voiceassistant.ui import mascot as go

ICON_BOX = (40, 10, 120, 90)     # canvas cells (x0, y0, x1, y1): the head and the upper arms, square
ICON_PX = 64


def octopus_icon() -> QIcon:
    """The octopus as the tray icon, drawn with the same renderer as the card."""
    frame = go.render_frame("assistant", "assistant", 1.0, None, 1.0, "idle", 1.3, 0.0, 0.0, ("", 0.0), 0.0)
    x0, y0, x1, y1 = ICON_BOX
    crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
    h, w = crop.shape[:2]
    image = QImage(crop.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
    pixmap = QPixmap.fromImage(image).scaled(ICON_PX, ICON_PX, Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation)
    return QIcon(pixmap)


def make_tray(win, on_restart, on_close) -> QSystemTrayIcon:
    """The tray icon for the overlay window `win`. Returns the icon: keep a reference while the app runs."""
    tray = QSystemTrayIcon(octopus_icon())
    tray.setToolTip("Assistente de voz")
    menu = QMenu()
    menu.addAction("Abrir definições", win.open_settings_view)
    menu.addAction("Reiniciar", on_restart)
    menu.addSeparator()
    menu.addAction("Fechar", on_close)
    tray.setContextMenu(menu)
    tray.activated.connect(lambda reason: win.toggle_visible()
                           if reason == QSystemTrayIcon.ActivationReason.Trigger else None)
    tray.menu = menu                                    # keeps the menu alive with the icon
    tray.show()
    return tray
