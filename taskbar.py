"""taskbar.py: keep the overlay out of the taskbar. Qt has no call for this, and its Tool flag still leaves a
normal-window type that the desktop lists, so the window manager is asked directly: _NET_WM_STATE_SKIP_TASKBAR is
added with a client message to the root window (X11, and XWayland under Plasma)."""
import ctypes

_CLIENT_MESSAGE = 33
_NET_WM_STATE_ADD = 1
_SOURCE_APPLICATION = 1
_SUBSTRUCTURE_NOTIFY_AND_REDIRECT = (1 << 19) | (1 << 20)


class _ClientMessage(ctypes.Structure):
    """The XClientMessageEvent layout of Xlib.h (64-bit)."""
    _fields_ = [("type", ctypes.c_int), ("serial", ctypes.c_ulong), ("send_event", ctypes.c_int),
                ("display", ctypes.c_void_p), ("window", ctypes.c_ulong), ("message_type", ctypes.c_ulong),
                ("format", ctypes.c_int), ("data", ctypes.c_long * 5)]


def hide_from_taskbar(window: int) -> bool:
    """Ask the window manager to skip `window` in the taskbar. False when there is no X11 display to ask."""
    try:
        x11 = ctypes.CDLL("libX11.so.6")
    except OSError:
        return False
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    x11.XSendEvent.restype = ctypes.c_int
    x11.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_long, ctypes.c_void_p]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    display = x11.XOpenDisplay(None)
    if not display:
        return False
    msg = _ClientMessage(type=_CLIENT_MESSAGE, send_event=1, display=display, window=window,
                         message_type=x11.XInternAtom(display, b"_NET_WM_STATE", 0), format=32)
    msg.data[0] = _NET_WM_STATE_ADD
    msg.data[1] = x11.XInternAtom(display, b"_NET_WM_STATE_SKIP_TASKBAR", 0)
    msg.data[2] = 0
    msg.data[3] = _SOURCE_APPLICATION
    event = (ctypes.c_char * 192)()                     # an XEvent is a union of 192 bytes
    ctypes.memmove(event, ctypes.byref(msg), ctypes.sizeof(msg))
    x11.XSendEvent(display, x11.XDefaultRootWindow(display), 0, _SUBSTRUCTURE_NOTIFY_AND_REDIRECT,
                   ctypes.addressof(event))
    x11.XFlush(display)
    return True
