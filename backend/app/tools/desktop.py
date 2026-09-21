"""
Desktop application GUI automation tools (Windows).

Gives the Local Agent eyes and hands INSIDE desktop apps (WhatsApp, Notepad,
Spotify, anything): list/focus windows, see a window's UI via a vision model
(screenshot -> Groq vision -> element descriptions with clickable positions),
click at coordinates, and send keys/shortcuts.

Pure PowerShell + user32 — no extra dependencies. Nothing here touches the
browser; these tools are registered on the Local Agent only.

=============================================================================
Why this file carries its own Win32 helper (`_PS_UI_HELPER`)
=============================================================================
Driving a modern Windows app from a background process is not as simple as
"find a process and send it keys". Three concrete failures were observed
live on this machine while automating WhatsApp, and the helper below exists
to fix each of them:

1. ONE APP, SEVERAL WINDOWS WITH THE SAME TITLE.
   WhatsApp Desktop is a WebView2 app. Both `WhatsApp.Root` AND the
   `msedgewebview2` child process expose a *visible top-level window titled
   "WhatsApp"*, and their rectangles differ (measured: -6,0 800x600 for the
   real app shell vs 1,1 786x592 for the WebView2 content host). The old code
   used `Get-Process | ... | Select-Object -First 1` independently in each
   tool, so `see_window` could screenshot one window while `click_window`
   computed its origin from the other — every click was then offset, and
   which window won varied between calls. `Find` below scores candidates so
   the real application window wins deterministically, and every tool here
   resolves the target through that same function.

2. A JUST-LAUNCHED APP HAS NO WINDOW YET.
   `open_app` returns the moment the launcher is spawned, but WhatsApp needs
   several seconds before its window exists (measured: `MainWindowHandle`
   stays 0 for ~6s). Anything that ran immediately afterwards found no
   window and gave up. `WaitFor` polls instead of failing instantly.

3. `SendKeys` DOES NOT REACH CHROMIUM/WEBVIEW2 CONTENT.
   .NET's `SendKeys.SendWait` drives input through legacy journal-playback
   hooks, which Chromium-based UI ignores. Typing therefore "succeeded"
   while nothing appeared in WhatsApp. `Type`/`Keys` below use `SendInput`,
   the modern low-level injection API, with `KEYEVENTF_UNICODE` for literal
   text so any character types correctly regardless of keyboard layout.

Focus is also verified rather than assumed: `SetForegroundWindow` is
routinely refused for a background process, so `Focus` applies the standard
`AttachThreadInput` + foreground-lock-timeout workaround and then CHECKS
`GetForegroundWindow`, retrying a few times, so callers never type into the
wrong window believing they succeeded.
"""

from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path
from typing import Type

import structlog
from app.tools.base import BaseTool
from pydantic import BaseModel, Field

from app.tools.runtime import on_main_loop, run_in_worker as _run_in_worker
from app.utils.vision import percent_to_pixels as _percent_to_pixels
from app.config import get_settings

logger = structlog.get_logger(__name__)

# Where screenshots are buffered between capture and vision call
_SHOT_DIR = Path.home() / ".self_improving_agent" / "screenshots"

_PS_COMMON = "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing;"
# Widest picture sent to the vision model. A two-monitor desktop is ~3840px
# wide; sent whole it is megabytes of base64 and the request is rejected.
_MAX_SHOT_WIDTH = 1600


# =============================================================================
# Shared Win32 helper compiled into every PowerShell call made by these tools
# (and by send_keys in tools/local.py, which imports it from here).
# =============================================================================

_PS_UI_HELPER = r"""
Add-Type @'
using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public class OaskUI {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int L, T, R, B; }

    public delegate bool EnumProc(IntPtr h, IntPtr l);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr l);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
    [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr h);
    [DllImport("user32.dll")] public static extern IntPtr SetActiveWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern IntPtr SetFocus(IntPtr h);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
    [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint a, uint b, bool f);
    [DllImport("user32.dll")] public static extern bool SystemParametersInfo(uint a, uint b, ref uint c, uint d);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(int f, int dx, int dy, int d, int e);
    [DllImport("user32.dll")] public static extern short VkKeyScan(char c);
    [DllImport("dwmapi.dll")] static extern int DwmGetWindowAttribute(IntPtr h, int attr, out int val, int size);
    [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
    [DllImport("user32.dll")] static extern IntPtr WindowFromPoint(POINT p);
    [DllImport("user32.dll")] static extern IntPtr GetAncestor(IntPtr h, uint flags);

    // Which process owns whatever is drawn at this screen position. The name
    // of the window is not enough to keep the agent out of the user's own
    // Chrome: a click with no window named is an ABSOLUTE screen click, and
    // Chrome may simply be what is in front of those coordinates.
    public static string ProcessAt(int x, int y) {
        try {
            POINT p; p.X = x; p.Y = y;
            IntPtr h = WindowFromPoint(p);
            if (h == IntPtr.Zero) return "";
            IntPtr root = GetAncestor(h, 2);   // GA_ROOT
            if (root != IntPtr.Zero) h = root;
            uint pid = 0; GetWindowThreadProcessId(h, out pid);
            if (pid == 0) return "";
            return System.Diagnostics.Process.GetProcessById((int)pid).ProcessName;
        } catch { return ""; }
    }

    // A Store app that Windows has SUSPENDED stays "visible" to user32 while
    // DWM reports it cloaked, and a suspended app has no live UI tree and is
    // not drawn on screen. Measured on Settings: both its windows reported
    // cloaked=2 (cloaked by the shell) with no accessibility children, and its
    // process reported Responding=False. Reading or photographing such a
    // window returns nothing useful - the screenshot would show whatever app
    // is actually in front of those coordinates.
    public static bool IsCloaked(IntPtr h) {
        int v = 0;
        try { return DwmGetWindowAttribute(h, 14, out v, 4) == 0 && v != 0; } catch { return false; }
    }

    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] static extern bool SetProcessDpiAwarenessContext(IntPtr ctx);

    // Windows lies about sizes to a program that has not said it understands
    // display scaling: at 150% scaling GetWindowRect and CopyFromScreen report
    // a virtual 1280x720 desktop while the real one is 1920x1080, so every
    // screenshot came back shrunk and every click landed short of its target.
    // -4 is DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 (Windows 10 1703+);
    // the older whole-process call is the fallback on anything earlier.
    public static void MakeDpiAware() {
        try { if (SetProcessDpiAwarenessContext(new IntPtr(-4))) return; } catch {}
        try { SetProcessDPIAware(); } catch {}
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct KEYBDINPUT { public ushort wVk; public ushort wScan; public uint dwFlags; public uint time; public IntPtr dwExtraInfo; }
    [StructLayout(LayoutKind.Explicit, Size = 40)]
    public struct INPUT { [FieldOffset(0)] public uint type; [FieldOffset(8)] public KEYBDINPUT ki; }
    [DllImport("user32.dll", SetLastError = true)]
    public static extern uint SendInput(uint n, INPUT[] p, int cb);

    const uint KEYEVENTF_KEYUP = 0x0002;
    const uint KEYEVENTF_UNICODE = 0x0004;

    // Shell/host processes that merely HOST another app's UI. A window owned
    // by one of these is never the real application window we want to drive.
    static string[] HostProcs = new string[] {
        "msedgewebview2", "applicationframehost", "wwahost", "textinputhost", "searchhost"
    };

    static bool IsHost(string proc) {
        foreach (string hp in HostProcs) { if (proc == hp) return true; }
        return false;
    }

    public class Win { public IntPtr H; public string Title; public string Proc; public RECT R; public int Score; }

    // Enumerate every visible, reasonably sized top-level window and score it
    // against `hint`. Highest score wins; ties broken by window area.
    public static Win Find(string hint) {
        string h = (hint == null ? "" : hint.ToLowerInvariant().Trim());
        List<Win> found = new List<Win>();
        EnumWindows(delegate(IntPtr hw, IntPtr lp) {
            if (!IsWindowVisible(hw)) return true;
            StringBuilder sb = new StringBuilder(512);
            GetWindowText(hw, sb, 512);
            string title = sb.ToString();
            if (title.Length == 0) return true;
            RECT r; if (!GetWindowRect(hw, out r)) return true;
            int w = r.R - r.L, ht = r.B - r.T;
            if (w < 120 || ht < 80) return true;          // tooltips, tray popups
            uint pid; GetWindowThreadProcessId(hw, out pid);
            string proc = "";
            try { proc = System.Diagnostics.Process.GetProcessById((int)pid).ProcessName.ToLowerInvariant(); } catch { }
            // Chrome windows are driven only by the browser tools. Found
            // 2026-09-17: send_keys('Chrome') typed into the user's own Chrome,
            // and a hint like 'Gemini' matched a Chrome tab title.
            if (proc == "chrome") return true;
            string lt = title.ToLowerInvariant();
            bool titleHit = h.Length > 0 && lt.Contains(h);
            bool procHit = h.Length > 0 && proc.Contains(h);
            if (h.Length > 0 && !titleHit && !procHit) return true;
            Win win = new Win();
            win.H = hw; win.Title = title; win.Proc = proc; win.R = r;
            win.Score = 0;
            if (procHit) win.Score += 100;                 // the app's own process
            if (titleHit) win.Score += 50;
            if (IsHost(proc)) win.Score -= 80;             // WebView2/UWP host shell
            win.Score += (w * ht) / 20000;                 // prefer the outer window
            found.Add(win);
            return true;
        }, IntPtr.Zero);
        Win best = null;
        foreach (Win w in found) {
            if (best == null || w.Score > best.Score) best = w;
        }
        return best;
    }

    // Poll for the window — a freshly launched app needs seconds to map one.
    public static Win WaitFor(string hint, int timeoutMs) {
        int waited = 0;
        while (true) {
            Win w = Find(hint);
            if (w != null) return w;
            if (waited >= timeoutMs) return null;
            System.Threading.Thread.Sleep(300);
            waited += 300;
        }
    }

    // Raise + focus, then VERIFY. A background process is normally refused the
    // foreground, so this uses the documented AttachThreadInput workaround and
    // drops the foreground lock timeout, retrying until GetForegroundWindow
    // actually reports our target.
    public static bool Focus(IntPtr h) {
        bool ok = RaiseToForeground(h);
        if (ok) FocusEmbeddedWebContent(h);
        return ok;
    }

    static bool RaiseToForeground(IntPtr h) {
        if (IsIconic(h)) ShowWindow(h, 9); else ShowWindow(h, 5);
        uint zero = 0;
        SystemParametersInfo(0x2001, 0, ref zero, 2);
        for (int i = 0; i < 6; i++) {
            IntPtr fg = GetForegroundWindow();
            if (fg == h) return true;
            uint fgT; GetWindowThreadProcessId(fg, out fgT);
            uint cur = GetCurrentThreadId();
            AttachThreadInput(cur, fgT, true);
            BringWindowToTop(h);
            SetForegroundWindow(h);
            SetActiveWindow(h);
            SetFocus(h);
            AttachThreadInput(cur, fgT, false);
            System.Threading.Thread.Sleep(250);
            if (GetForegroundWindow() == h) return true;
        }
        return GetForegroundWindow() == h;
    }

    [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr p, EnumProc cb, IntPtr l);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
    [StructLayout(LayoutKind.Sequential)]
    public struct GUITHREADINFO {
        public int cbSize; public int flags; public IntPtr hwndActive; public IntPtr hwndFocus;
        public IntPtr hwndCapture; public IntPtr hwndMenuOwner; public IntPtr hwndMoveSize;
        public IntPtr hwndCaret; public RECT rcCaret;
    }
    [DllImport("user32.dll")] public static extern bool GetGUIThreadInfo(uint tid, ref GUITHREADINFO info);

    // Apps that host a web page inside a native shell (WhatsApp Desktop is
    // WinUI + WebView2) keep keyboard focus on the shell's hidden input-site
    // window after being raised, so every keystroke — Ctrl+F included — is
    // swallowed and the page never sees it. Verified live after a cold
    // WhatsApp start: focus sat on InputSiteWindowClass and typing did nothing;
    // after SetFocus on the Chrome_WidgetWin_0 web host child, Ctrl+F opened the
    // search box and the typed text appeared. Apps without such a child are
    // left exactly as they were.
    static void FocusEmbeddedWebContent(IntPtr top) {
        IntPtr host = IntPtr.Zero;
        EnumChildWindows(top, delegate(IntPtr c, IntPtr l) {
            StringBuilder sb = new StringBuilder(128);
            GetClassName(c, sb, 128);
            if (sb.ToString() == "Chrome_WidgetWin_0" && IsWindowVisible(c)) { host = c; return false; }
            return true;
        }, IntPtr.Zero);
        if (host == IntPtr.Zero) return;
        uint pid;
        uint tid = GetWindowThreadProcessId(host, out pid);
        GUITHREADINFO gi = new GUITHREADINFO();
        gi.cbSize = Marshal.SizeOf(typeof(GUITHREADINFO));
        if (GetGUIThreadInfo(tid, ref gi) && gi.hwndFocus == host) return;   // already there
        uint cur = GetCurrentThreadId();
        AttachThreadInput(cur, tid, true);
        SetFocus(host);
        AttachThreadInput(cur, tid, false);
        System.Threading.Thread.Sleep(150);
    }

    // Literal text via SendInput + KEYEVENTF_UNICODE. Works in Chromium /
    // WebView2 content where .NET SendKeys silently does nothing.
    public static void Type(string text) {
        foreach (char ch in text) {
            INPUT[] inp = new INPUT[2];
            inp[0].type = 1; inp[0].ki.wScan = ch; inp[0].ki.dwFlags = KEYEVENTF_UNICODE;
            inp[1].type = 1; inp[1].ki.wScan = ch; inp[1].ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP;
            SendInput(2, inp, Marshal.SizeOf(typeof(INPUT)));
            System.Threading.Thread.Sleep(18);
        }
    }

    static void Tap(ushort vk, bool up) {
        INPUT[] inp = new INPUT[1];
        inp[0].type = 1; inp[0].ki.wVk = vk;
        inp[0].ki.dwFlags = up ? KEYEVENTF_KEYUP : 0;
        SendInput(1, inp, Marshal.SizeOf(typeof(INPUT)));
        System.Threading.Thread.Sleep(18);
    }

    static ushort NamedKey(string name) {
        switch (name.ToUpperInvariant()) {
            case "ENTER": case "RETURN": return 0x0D;
            case "TAB": return 0x09;
            case "ESC": case "ESCAPE": return 0x1B;
            case "BACKSPACE": case "BS": return 0x08;
            case "DELETE": case "DEL": return 0x2E;
            case "UP": return 0x26;
            case "DOWN": return 0x28;
            case "LEFT": return 0x25;
            case "RIGHT": return 0x27;
            case "HOME": return 0x24;
            case "END": return 0x23;
            case "PGUP": return 0x21;
            case "PGDN": return 0x22;
            case "SPACE": return 0x20;
            case "F1": return 0x70; case "F2": return 0x71; case "F3": return 0x72;
            case "F4": return 0x73; case "F5": return 0x74; case "F6": return 0x75;
            case "F7": return 0x76; case "F8": return 0x77; case "F9": return 0x78;
            case "F10": return 0x79; case "F11": return 0x7A; case "F12": return 0x7B;
        }
        return 0;
    }

    // Parse the familiar .NET SendKeys notation ('^f', '{ENTER}', '+{TAB}')
    // but deliver every keystroke through SendInput.
    public static void Keys(string spec) {
        bool ctrl = false, alt = false, shift = false;
        int i = 0;
        while (i < spec.Length) {
            char c = spec[i];
            if (c == '^') { ctrl = true; i++; continue; }
            if (c == '%') { alt = true; i++; continue; }
            if (c == '+') { shift = true; i++; continue; }
            ushort vk = 0;
            bool isUnicode = false;
            char literal = '\0';
            if (c == '{') {
                int close = spec.IndexOf('}', i);
                if (close < 0) { literal = c; isUnicode = true; i++; }
                else {
                    string name = spec.Substring(i + 1, close - i - 1);
                    vk = NamedKey(name);
                    if (vk == 0) {
                        if (name.Length == 1) { literal = name[0]; isUnicode = true; }
                        else { i = close + 1; ctrl = alt = shift = false; continue; }
                    }
                    i = close + 1;
                }
            } else { literal = c; isUnicode = true; i++; }

            if (ctrl) Tap(0x11, false);
            if (alt) Tap(0x12, false);
            if (shift) Tap(0x10, false);

            if (isUnicode && (ctrl || alt)) {
                // Shortcuts need a real virtual key, not a unicode packet.
                short sc = VkKeyScan(literal);
                Tap((ushort)(sc & 0xFF), false);
                Tap((ushort)(sc & 0xFF), true);
            } else if (isUnicode) {
                Type(literal.ToString());
            } else {
                Tap(vk, false);
                Tap(vk, true);
            }

            if (shift) Tap(0x10, true);
            if (alt) Tap(0x12, true);
            if (ctrl) Tap(0x11, true);
            ctrl = alt = shift = false;
        }
    }

    public static void Click(int x, int y, int times) {
        SetCursorPos(x, y);
        System.Threading.Thread.Sleep(140);
        for (int i = 0; i < times; i++) {
            mouse_event(2, 0, 0, 0, 0);
            System.Threading.Thread.Sleep(60);
            mouse_event(4, 0, 0, 0, 0);
            System.Threading.Thread.Sleep(90);
        }
    }
}
'@
# Must run before any window is measured or captured in this PowerShell process.
[OaskUI]::MakeDpiAware()
"""


# Windows PowerShell writes captured output in the console's OEM code page when
# the script comes in through -Command, so a window title such as "नमस्ते"
# arrived as "??????" (verified 2026-09-16). Every -Command script starts with this.
_PS_UTF8 = "try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}; "


def _ps(script: str, timeout: int = 60) -> str:
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", _PS_UTF8 + script],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )
    out = (proc.stdout or "").strip()
    if not out and proc.stderr:
        out = f"[stderr] {proc.stderr.strip()[:800]}"
    return out or "(no output)"


def _check_enabled() -> str | None:
    if not get_settings().local_tools_enabled:
        return "Local computer tools are disabled (local_tools_enabled=false)."
    return None


def _ps_quote(value: str) -> str:
    """Escape a value for embedding inside a PowerShell single-quoted string."""
    return (value or "").replace("'", "''")


_CHROME_HINT = re.compile(r"\bchrome\b|^\s*(?:the\s+|web\s+)?browser\s*$", re.IGNORECASE)
CHROME_IS_BROWSER_TOOLS = (
    "Chrome is controlled only with the browser tools (navigate_browser, perceive_page, see_page, "
    "type_into_element, click_element, click_at_position, press_key, download_file, add_to_cart), "
    "which work in the "
    "agent's own Chrome window; call use_browser first if they are not in your tool list. Desktop "
    "keys and clicks never go to a Chrome window, so the user's own Chrome profiles are not touched."
)


def chrome_window_refusal(hint: str) -> str | None:
    """A refusal when a desktop tool is pointed at Chrome, else None.

    Found 2026-09-17: open_app('Chrome') started the user's own Chrome on its
    last-used profile, and send_keys('Chrome') then typed into it.
    """
    if hint and _CHROME_HINT.search(hint):
        logger.info("desktop_tool_refused_chrome", hint=hint[:60])
        return f"Refused: '{hint}' is Chrome. {CHROME_IS_BROWSER_TOOLS}"
    return None


# =============================================================================
# Waiting for an app to finish STARTING, not merely for its window to exist
# =============================================================================
# Measured on this machine with a cold WhatsApp start (process not running):
# the window appears ~2s after launch but shows a static splash (logo, progress
# bar, "End-to-end encrypted") for ~8 more seconds before the chat list
# renders. Keystrokes sent during the splash go nowhere, and the vision model,
# shown the splash, described it as a "QR-code login screen" — so the agent
# told a logged-in user to log in. Neither signal that seemed obvious works:
# the WebView2 window appears DURING the splash, and vision (asked directly)
# even claimed the splash contained a list and a search box.
#
# What does separate the two states is how empty the splash is. Scored with
# _frame_detail on real PrintWindow captures, splash frames measured at most
# 2.9% non-background / 1.4% edge pixels, while every loaded, non-host app
# window measured (WhatsApp, Settings, the IDE, Command Palette, PowerToys)
# had at least 5.7% non-background. PrintWindow is used so the check works
# even when the new window opens behind whatever the user is looking at —
# which is exactly what happens when this background process launches an app.

_HOST_PROCESSES = {"msedgewebview2", "applicationframehost", "wwahost", "textinputhost", "searchhost"}
_SPLASH_MAX_NONBG = 4.5
_SPLASH_MAX_EDGE = 2.0

_PS_FRAME_CAPTURE = r"""
Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @'
using System;
using System.Drawing;
using System.Runtime.InteropServices;
public class OaskShot {
    public struct RECT { public int L, T, R, B; }
    [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] static extern bool IsIconic(IntPtr h);
    [DllImport("user32.dll")] static extern bool IsWindow(IntPtr h);
    [DllImport("user32.dll")] static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
    public static bool Capture(IntPtr h, string file) {
        if (!IsWindow(h) || IsIconic(h)) return false;
        RECT r; if (!GetWindowRect(h, out r)) return false;
        int w = r.R - r.L, ht = r.B - r.T;
        if (w < 50 || ht < 50) return false;
        using (Bitmap b = new Bitmap(w, ht)) {
            bool ok;
            using (Graphics g = Graphics.FromImage(b)) {
                IntPtr hdc = g.GetHdc();
                ok = PrintWindow(h, hdc, 2);   // PW_RENDERFULLCONTENT
                g.ReleaseHdc(hdc);
            }
            if (!ok) return false;
            b.Save(file);
        }
        return true;
    }
}
'@
"""


def _reduce_frame(image):
    """Greyscale copy 200px wide — the scale the splash thresholds were measured at."""
    grey = image.convert("L")
    return grey.resize((200, max(1, int(200 * grey.height / grey.width))))


def _frame_detail(small) -> tuple[float, float]:
    """(non-background %, horizontal-edge %) of a frame reduced by _reduce_frame."""
    w, h = small.size
    data = small.tobytes()
    hist = small.histogram()
    bg = max(range(256), key=hist.__getitem__)
    off = sum(1 for p in data if abs(p - bg) > 24)
    edges = sum(
        1 for y in range(h) for x in range(1, w)
        if abs(data[y * w + x] - data[y * w + x - 1]) > 40
    )
    return 100.0 * off / (w * h), 100.0 * edges / (w * h)


def _looks_like_splash(small) -> bool:
    nonbg, edge = _frame_detail(small)
    return nonbg < _SPLASH_MAX_NONBG and edge < _SPLASH_MAX_EDGE


def _frame_change(a, b) -> float:
    """Percentage of pixels that differ noticeably between two reduced frames."""
    from PIL import ImageChops

    if a.size != b.size:
        return 100.0
    diff = ImageChops.difference(a, b).tobytes()
    return 100.0 * sum(1 for p in diff if p > 20) / len(diff)


def wait_until_app_ready(
    hint: str, window_timeout_ms: int = 25000, ready_timeout_s: float = 45.0
) -> tuple[str | None, str]:
    """Wait for `hint`'s window to exist AND to finish loading.

    Returns (description, status) where status is:
      "ready"     — content is on screen (stable, or an app that keeps animating);
      "loading"   — the window exists but still looked like a splash when time ran out;
      "no_window" — no matching window appeared.
    """
    from PIL import Image

    _SHOT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [_SHOT_DIR / "ready0.png", _SHOT_DIR / "ready1.png"]
    max_frames = max(4, int(ready_timeout_s / 0.5))
    script = _PS_UTF8 + _PS_UI_HELPER + _PS_FRAME_CAPTURE + f"""
$w = [OaskUI]::WaitFor('{_ps_quote(hint)}', {int(window_timeout_ms)})
if (-not $w) {{ [Console]::Out.WriteLine('NONE'); [Console]::Out.Flush(); exit 0 }}
[Console]::Out.WriteLine('WIN|' + $w.Proc + '|' + $w.Title); [Console]::Out.Flush()
for ($i = 0; $i -lt {max_frames}; $i++) {{
    if ($i % 2 -eq 0) {{ $f = '{_ps_quote(str(frames[0]))}' }} else {{ $f = '{_ps_quote(str(frames[1]))}' }}
    $ok = [OaskShot]::Capture($w.H, $f)
    [Console]::Out.WriteLine('FRAME|' + $ok + '|' + $f); [Console]::Out.Flush()
    Start-Sleep -Milliseconds 500
}}
[Console]::Out.WriteLine('END'); [Console]::Out.Flush()
"""
    proc = subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", script],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace",
    )
    desc: str | None = None
    host = False
    prev = None
    stable = 0
    seen = 0
    last_splash = False
    try:
        for raw in proc.stdout:  # type: ignore[union-attr]
            line = raw.strip()
            if line == "NONE":
                return None, "no_window"
            if line == "END":
                break
            if line.startswith("WIN|"):
                _, pname, title = line.split("|", 2)
                desc = f"{pname} :: {title}"
                host = pname.lower() in _HOST_PROCESSES
                continue
            if not line.startswith("FRAME|"):
                continue
            _, ok, path = line.split("|", 2)
            if ok.strip().lower() != "true":
                prev, stable = None, 0       # minimised / not paintable yet
                continue
            try:
                with Image.open(path) as im:
                    im.load()
                    small = _reduce_frame(im)
            except Exception:
                continue
            seen += 1
            stable = stable + 1 if prev is not None and _frame_change(prev, small) < 0.5 else 0
            prev = small
            last_splash = (not host) and _looks_like_splash(small)
            if last_splash:
                continue                      # still on the splash — keep waiting
            if stable >= 2 or seen >= 6:      # settled, or an app that keeps animating
                logger.info("app_ready", window=desc, frames=seen)
                return desc, "ready"
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    if desc is None:
        return None, "no_window"
    logger.info("app_ready_timeout", window=desc, still_splash=last_splash)
    return desc, ("loading" if last_splash else "ready")


# =============================================================================
# 1. List open application windows
# =============================================================================

class ListWindowsInput(BaseModel):
    # Groq's tool-calling schema validation rejects a function whose
    # `parameters` object has an empty `properties` dict — somewhere in the
    # litellm/Groq request pipeline an all-empty `properties: {}` gets
    # dropped from the JSON body while `required: []` survives, and Groq then
    # errors with "'required' present but 'properties' is missing", failing the
    # whole run. This tool genuinely takes no arguments, so one harmless unused
    # optional field keeps `properties` non-empty end to end.
    unused: str = Field(default="", description="Not used — leave blank. This tool takes no real arguments.")


class ListWindowsTool(BaseTool):
    name: str = "list_windows"
    description: str = (
        "Lists the currently open application windows (title + application). "
        "Use first to find the exact window title for focus_window."
    )
    args_schema: Type[BaseModel] = ListWindowsInput

    def _run(self, unused: str = "") -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            # Enumerate real top-level windows rather than relying on
            # Process.MainWindowTitle, which is empty for many packaged apps
            # (WhatsApp reports no main window at all until it finishes
            # launching) and therefore hid them from this listing entirely.
            script = _PS_UI_HELPER + r"""
$seen = @{}
$sb = New-Object System.Text.StringBuilder
$cb = [OaskUI+EnumProc]{
    param($hw, $lp)
    if ([OaskUI]::IsWindowVisible($hw)) {
        $t = New-Object System.Text.StringBuilder 512
        [OaskUI]::GetWindowText($hw, $t, 512) | Out-Null
        $title = $t.ToString()
        if ($title) {
            $r = New-Object OaskUI+RECT
            if ([OaskUI]::GetWindowRect($hw, [ref]$r)) {
                $w = $r.R - $r.L; $h = $r.B - $r.T
                if ($w -ge 120 -and $h -ge 80) {
                    $procId = 0
                    [OaskUI]::GetWindowThreadProcessId($hw, [ref]$procId) | Out-Null
                    $pn = try { (Get-Process -Id $procId -ErrorAction Stop).ProcessName } catch { '?' }
                    $key = "$pn :: $title"
                    if (-not $seen.ContainsKey($key)) {
                        $seen[$key] = $true
                        # A suspended Store app or a tray app waiting to be
                        # summoned (PowerToys Quick Access, Command Palette)
                        # stays "visible" to user32 while DWM keeps it cloaked:
                        # it is not on screen, has no live interface, and must
                        # be woken with focus_window before it can be read,
                        # photographed or clicked.
                        $mark = ''
                        if ([OaskUI]::IsCloaked($hw)) { $mark = '   (hidden - focus_window to wake it)' }
                        [void]$sb.AppendLine("$pn :: $title   [${w}x${h}]$mark")
                    }
                }
            }
        }
    }
    return $true
}
[OaskUI]::EnumWindows($cb, [IntPtr]::Zero) | Out-Null
$sb.ToString().Trim()
"""
            out = _ps(script)
            return f"Open windows (app :: title [size]):\n{out}"
        return _run_in_worker(_async_run)


# =============================================================================
# 2. Focus (activate) a window by (partial) title or app name
# =============================================================================

class FocusWindowInput(BaseModel):
    window: str = Field(..., description="Part of the window title or the app name (e.g. 'WhatsApp', 'Notepad').")


class FocusWindowTool(BaseTool):
    name: str = "focus_window"
    description: str = (
        "Brings an application window to the front and gives it keyboard focus, "
        "e.g. focus_window('WhatsApp'). Waits for the window if the app is still "
        "starting up, and reports honestly whether focus was actually obtained."
    )
    args_schema: Type[BaseModel] = FocusWindowInput

    def _run(self, window: str) -> str:
        async def _async_run():
            blocked = _check_enabled() or chrome_window_refusal(window)
            if blocked:
                return blocked
            safe = _ps_quote(window)
            script = _PS_UI_HELPER + f"""
$w = [OaskUI]::WaitFor('{safe}', 15000)
if (-not $w) {{ Write-Output 'NOTFOUND'; exit 0 }}
$ok = [OaskUI]::Focus($w.H)
Write-Output ("RESULT|" + $ok + "|" + $w.Proc + "|" + $w.Title)
"""
            out = _ps(script)
            if out.startswith("NOTFOUND"):
                return (
                    f"No visible window matching '{window}' (waited 15s). "
                    "If you just launched the app, it may still be starting; "
                    "call list_windows to see what is actually open."
                )
            if out.startswith("RESULT|"):
                _, ok, proc, title = out.split("|", 3)
                logger.info("desktop_window_focused", window=window, ok=ok)
                if ok.strip().lower() == "true":
                    return f"Focused '{title}' ({proc}) — it now has keyboard focus."
                return (
                    f"Found '{title}' ({proc}) but Windows refused to bring it to the "
                    "foreground. Do not send keys yet — retry focus_window."
                )
            return f"focus_window unexpected output: {out[:300]}"
        return _run_in_worker(_async_run)


# =============================================================================
# 3. See a window (screenshot -> Groq vision -> UI description with positions)
# =============================================================================

class SeeWindowInput(BaseModel):
    window: str = Field(default="", description="Part of window title/app to capture (empty = the whole screen, every monitor).")
    question: str = Field(default="Describe every clickable element with its approximate pixel position",
                          description="What to look for, e.g. 'where is the search box and message input?'")


class SeeWindowTool(BaseTool):
    name: str = "see_window"
    description: str = (
        "Takes a screenshot of a window (or, with window='', the whole screen across "
        "every monitor) and returns a visual "
        "description: every button/field/input with its approximate pixel coordinates "
        "RELATIVE TO THAT WINDOW. Use it to learn where to click with click_window, "
        "or to verify app state (e.g. is the chat open?). Always call this again "
        "after any click or typing, because the layout changes."
    )
    args_schema: Type[BaseModel] = SeeWindowInput

    def _run(self, window: str = "", question: str = "Describe every clickable element with its approximate pixel position") -> str:
        async def _async_run():
            blocked = _check_enabled() or chrome_window_refusal(window)
            if blocked:
                return blocked

            _SHOT_DIR.mkdir(parents=True, exist_ok=True)
            shot = _SHOT_DIR / "see.png"
            safe = _ps_quote(window)

            # Resolve through the SAME scorer click_window uses, so the origin
            # these coordinates are relative to is the origin clicks are
            # applied to. Focus first so the window is actually on top and not
            # occluded by another app in the screenshot.
            script = _PS_UI_HELPER + _PS_COMMON + f"""
$hint = '{safe}'
# NOTE: PowerShell variable names are CASE-INSENSITIVE, so the capture size
# must not be held in $W/$H — those would silently overwrite $w, the window
# object, and its .Proc/.Title would come back empty.
$capL = 0; $capT = 0; $capW = 0; $capH = 0; $mode = 'SCREEN'; $desc = 'entire screen'
if ($hint -ne '') {{
    $win = [OaskUI]::WaitFor($hint, 8000)
    if ($win) {{
        [OaskUI]::Focus($win.H) | Out-Null
        Start-Sleep -Milliseconds 450
        $r = New-Object OaskUI+RECT
        if ([OaskUI]::GetWindowRect($win.H, [ref]$r)) {{
            $capL = $r.L; $capT = $r.T; $capW = $r.R - $r.L; $capH = $r.B - $r.T
            if ($capW -gt 0 -and $capH -gt 0) {{ $mode = 'WINDOW'; $desc = $win.Proc + ' :: ' + $win.Title }}
        }}
    }}
    # A named window that does not exist must not silently become a screenshot
    # of the whole screen: the agent then clicked blindly on the user's screen.
    if ($mode -eq 'SCREEN') {{
        Write-Output 'NOWINDOW|'
        exit
    }}
    if ([OaskUI]::IsCloaked($win.H)) {{
        Write-Output 'CLOAKED|'
        exit
    }}
}}
if ($mode -eq 'SCREEN') {{
    # The VIRTUAL screen is every monitor together. PrimaryScreen.Bounds was
    # used before, so a window on a second monitor was simply invisible.
    # Its origin can be negative (a monitor placed left of the main one), which
    # is why $capL/$capT travel back and are added to the coordinates below.
    $b = [System.Windows.Forms.SystemInformation]::VirtualScreen
    $capL = $b.X; $capT = $b.Y; $capW = $b.Width; $capH = $b.Height
    $desc = 'entire screen (' + [System.Windows.Forms.Screen]::AllScreens.Count + ' monitor(s))'
}}
$bmp = New-Object System.Drawing.Bitmap($capW, $capH)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($capL, $capT, 0, 0, $bmp.Size)
# Two monitors side by side make a picture far larger than the vision model
# accepts. Positions are asked for as percentages, so a smaller copy describes
# the same layout; the real capture size below still turns them into pixels.
$save = $bmp
if ($capW -gt {_MAX_SHOT_WIDTH}) {{
    $sw = {_MAX_SHOT_WIDTH}; $sh = [int][math]::Round($capH * ({_MAX_SHOT_WIDTH} / $capW))
    if ($sh -lt 1) {{ $sh = 1 }}
    $small = New-Object System.Drawing.Bitmap($sw, $sh)
    $sg = [System.Drawing.Graphics]::FromImage($small)
    $sg.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $sg.DrawImage($bmp, 0, 0, $sw, $sh)
    $sg.Dispose()
    $save = $small
}}
$save.Save('{_ps_quote(str(shot))}')
Write-Output ("CAP|" + $mode + "|" + $capL + "|" + $capT + "|" + $capW + "|" + $capH + "|" + $desc)
"""
            cap = _ps(script)
            if "CLOAKED|" in cap:
                return (
                    f"'{window}' is open but SUSPENDED by Windows (its window is cloaked), so it is not "
                    "drawn on screen — nothing was captured, because the screenshot would have shown "
                    "whatever app is actually in front of those coordinates. Call focus_window to wake it, "
                    "wait a moment, then look again."
                )
            if "NOWINDOW|" in cap:
                # Found 2026-09-15: see_window('Gemini') found no such window,
                # captured the entire screen instead, and the agent clicked at
                # screen coordinates on whatever happened to be there.
                return (
                    f"No visible window matching '{window}' — nothing was captured and nothing should be clicked. "
                    f"If '{window}' is a website or web app, it is in the browser: use navigate_browser, "
                    "see_page and perceive_page (see_window never captures Chrome windows). "
                    "To look at the whole screen on purpose, call see_window with window=''."
                )
            line = next((l for l in cap.splitlines() if l.startswith("CAP|")), "")
            if not line:
                return f"Screenshot failed: {cap[:300]}"
            _, mode, left, top, width, height, desc = line.split("|", 6)
            img_w, img_h = int(width), int(height)

            try:
                image_b64 = base64.b64encode(shot.read_bytes()).decode()
            except Exception as e:
                return f"Could not read screenshot: {e}"

            # Ask for PERCENTAGES, not raw pixels. Vision models are good at
            # relative placement but poor at absolute pixel counts — asked for
            # pixels against this very window, the model returned x values of
            # ~900 for a 786px-wide image. Percentages are scale-free, and we
            # convert them here using the real capture size.
            raw = await _describe_image(
                image_b64,
                f"You are looking at a screenshot of a {'desktop application window' if mode == 'WINDOW' else 'computer screen'}.\n"
                f"Question: {question}\n\n"
                "List every visible interactive element (buttons, text inputs, search boxes, "
                "list rows, chat rows, icons) that a user could click.\n"
                "Format EVERY element on its own line exactly as:\n"
                "`label` -> (X%, Y%)\n"
                "where X% and Y% are the CENTRE of that element expressed as a PERCENTAGE "
                "(0-100) of the image width and height. Example: `Search box` -> (31%, 26%).\n"
                "Do not give pixel values. Be concise and list the most useful elements first.\n"
                "If the window shows only an app logo, its name, a progress bar or a short notice such as "
                "'End-to-end encrypted', start your answer with `LOADING SCREEN - the app is still starting` "
                "instead of listing elements. Only mention a QR code if a square QR code is actually drawn "
                "in the image.",
            )
            if raw is None:
                return (
                    "Vision is unavailable right now, so I cannot see this window. "
                    "Do NOT guess coordinates for click_window — a blind click can hit the "
                    "wrong control. Retry see_window, or drive the app with send_keys "
                    "keyboard navigation instead."
                )

            # WINDOW coordinates stay relative to the window; SCREEN ones are used
            # as absolute positions, so they need the captured origin added.
            off_x, off_y = (0, 0) if mode == "WINDOW" else (int(left), int(top))
            converted = _percent_to_pixels(raw, img_w, img_h, off_x, off_y)
            origin = (
                "window's top-left corner (pass these straight to click_window with the same `window`)"
                if mode == "WINDOW"
                else "SCREEN's top-left corner — pass them to click_window with window='' so they are used as absolute screen coordinates"
            )
            return (
                f"Captured {desc} ({mode}, {img_w}x{img_h} px).\n"
                f"Coordinates below are pixels from the {origin}.\n\n{converted}"
            )
        return _run_in_worker(_async_run)


async def _describe_image(image_b64: str, prompt: str) -> str | None:
    """Describe a window screenshot through the shared vision provider chain.

    Gemini goes first when GEMINI_API_KEY is set and the Groq vision model is
    the fallback (see app.utils.vision). The Groq-only path this replaced was
    capped at 1000 output tokens per minute on this account, so a couple of
    see_window calls in a row regularly came back empty.
    """
    from app.utils.vision import describe_image

    # see_window's body runs on a worker thread's own loop (run_in_worker); the
    # vision request goes to the main loop, which owns LiteLLM's connections.
    return await on_main_loop(describe_image(image_b64, prompt, mime="image/png", max_tokens=900))


# =============================================================================
# 4. Click at window-relative coordinates
# =============================================================================

class ClickWindowInput(BaseModel):
    x: int = Field(..., description="X position in pixels, relative to the window's top-left (as reported by see_window).")
    y: int = Field(..., description="Y position in pixels, relative to the window's top-left.")
    window: str = Field(default="", description="Target window title/app (focuses it first). Empty = click on screen at absolute position.")
    double_click: bool = Field(default=False, description="Perform a double click.")


class ClickWindowTool(BaseTool):
    name: str = "click_window"
    description: str = (
        "Clicks inside an application window at the coordinates see_window reported "
        "(x/y are relative to that window's top-left; the tool focuses the window and "
        "translates to absolute screen positions itself). Always take a fresh "
        "see_window reading before clicking — never reuse coordinates from an older "
        "screenshot, and never guess them."
    )
    args_schema: Type[BaseModel] = ClickWindowInput

    def _run(self, x: int, y: int, window: str = "", double_click: bool = False) -> str:
        async def _async_run():
            blocked = _check_enabled() or chrome_window_refusal(window)
            if blocked:
                return blocked
            safe = _ps_quote(window)
            clicks = 2 if double_click else 1
            script = _PS_UI_HELPER + f"""
$hint = '{safe}'
$originL = 0; $originT = 0; $label = 'screen'
if ($hint -eq '') {{
    # An absolute screen click, so nothing named Chrome was refused earlier:
    # check what is actually drawn at that point instead.
    $owner = [OaskUI]::ProcessAt({int(x)}, {int(y)})
    if ($owner -match 'chrome') {{ Write-Output ('CHROMEPOINT|' + $owner); exit 0 }}
}}
if ($hint -ne '') {{
    $w = [OaskUI]::WaitFor($hint, 15000)
    if (-not $w) {{ Write-Output 'NOTFOUND'; exit 0 }}
    if (-not [OaskUI]::Focus($w.H)) {{ Write-Output 'FOCUSFAIL'; exit 0 }}
    Start-Sleep -Milliseconds 350
    $r = New-Object OaskUI+RECT
    if ([OaskUI]::GetWindowRect($w.H, [ref]$r)) {{ $originL = $r.L; $originT = $r.T }}
    $label = $w.Proc + ' :: ' + $w.Title
}}
$ax = $originL + {int(x)}; $ay = $originT + {int(y)}
[OaskUI]::Click($ax, $ay, {clicks})
Write-Output ("CLICKED|" + $ax + "|" + $ay + "|" + $label)
"""
            out = _ps(script)
            if out.startswith("CHROMEPOINT|"):
                owner = out.split("|", 1)[1].strip()
                logger.info("desktop_click_refused_chrome_at_point", x=x, y=y, process=owner[:40])
                return (
                    f"Refused: ({x}, {y}) is inside a Chrome window ({owner}), which may be one of the "
                    f"user's own. {CHROME_IS_BROWSER_TOOLS}"
                )
            if out.startswith("NOTFOUND"):
                return (
                    f"No visible window matching '{window}' (waited 15s) — nothing was clicked. "
                    "Call list_windows to see what is open."
                )
            if out.startswith("FOCUSFAIL"):
                return (
                    f"Found '{window}' but could not bring it to the foreground — nothing was "
                    "clicked (a click would have landed on whatever is actually in front). Retry."
                )
            if out.startswith("CLICKED|"):
                _, ax, ay, label = out.split("|", 3)
                logger.info("desktop_click", x=x, y=y, window=window)
                return (
                    f"Clicked ({x}, {y}) inside {label} at screen ({ax}, {ay}). "
                    "Call see_window again to confirm what changed before the next step."
                )
            return f"click_window unexpected output: {out[:300]}"
        return _run_in_worker(_async_run)


# =============================================================================
# 5. Read a window as Markdown (Windows UI Automation, no vision model)
# =============================================================================
# Added 2026-09-20, the local half of the same idea as read_page_as_markdown.
# see_window photographs a window and asks a vision model what it sees, which
# costs a free-tier call, takes seconds and returns ESTIMATED positions - the
# model judges "about 31% across" and the click lands near, not on, the target.
#
# Windows already knows the answer exactly. UI Automation is the service screen
# readers use: every control in a window, with the name it announces, its type,
# its state and its true rectangle. It is built into Windows, needs no install,
# no API key and no internet, and it is exact rather than estimated.
#
# Not every app answers: one that paints its own interface (some games, some
# Electron apps before their accessibility tree is built) exposes little or
# nothing. The tool says so plainly and points back at see_window, so the
# screenshot path remains the fallback rather than the default.

_PS_UIA = r"""
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
"""

_UIA_MAX_NODES = 300
_UIA_MAX_DEPTH = 14
_UIA_SECONDS = 20

# The buttons every window has. A window that exposes ONLY these has not built
# its accessibility tree - measured on WhatsApp, whose WebView2 content stays
# hidden however often it is asked - so the agent must be told to use its eyes
# instead of believing the window is nearly empty.
_UIA_WINDOW_FURNITURE = {"minimize", "restore", "maximize", "close", "system menu",
                         "non client input sink window", "appwindow custom title bar"}
# Control types worth reporting even with no name of their own.
_UIA_ALWAYS = {"Edit", "Document", "CheckBox", "RadioButton", "ComboBox", "List", "Tree", "Slider", "Tab"}
# Types that are pure layout: they carry nothing the agent can act on.
_UIA_SKIP = {"Pane", "Custom", "Group", "Thumb", "Separator", "TitleBar", "ScrollBar"}


class ReadWindowInput(BaseModel):
    window: str = Field(default="", description="Part of the window title or app name to read (e.g. 'Notepad', 'WhatsApp').")
    max_items: int = Field(default=120, description="Most controls to list.")


class ReadWindowTool(BaseTool):
    name: str = "read_window_as_markdown"
    description: str = (
        "Reads an application window as a Markdown outline using Windows' own accessibility "
        "service: every button, field, menu item and list row with the name Windows announces, "
        "its state (checked, disabled, selected) and its EXACT position for click_window. "
        "Try this BEFORE see_window on any desktop app: it is instant, free, needs no vision "
        "model, and its positions are measured rather than estimated. If it reports that the "
        "app exposes nothing, fall back to see_window."
    )
    args_schema: Type[BaseModel] = ReadWindowInput

    def _run(self, window: str = "", max_items: int = 120) -> str:
        async def _async_run():
            blocked = _check_enabled() or chrome_window_refusal(window)
            if blocked:
                return blocked
            if not window.strip():
                return (
                    "read_window_as_markdown needs a window: pass part of its title or app name "
                    "(call list_windows to see what is open). To look at the whole screen, use see_window."
                )

            cap = max(10, min(int(max_items), _UIA_MAX_NODES))
            safe = _ps_quote(window)
            script = _PS_UI_HELPER + _PS_UIA + rf"""
$win = [OaskUI]::WaitFor('{safe}', 8000)
if (-not $win) {{ Write-Output 'NOWINDOW'; exit }}
$r = New-Object OaskUI+RECT
[OaskUI]::GetWindowRect($win.H, [ref]$r) | Out-Null
Write-Output ("WIN|" + $win.Proc + "|" + $win.Title + "|" + $r.L + "|" + $r.T + "|" + ($r.R - $r.L) + "|" + ($r.B - $r.T))
Write-Output ("CLOAKED|" + [int][OaskUI]::IsCloaked($win.H))

$root = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$win.H)
if (-not $root) {{ Write-Output 'NOTREE'; exit }}
$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker

# A packaged app (Settings, Photos, the Store) draws inside a frame window
# owned by ApplicationFrameHost, so the element behind its OWN handle is an
# empty pane - measured on Settings, which returned a childless Pane and looked
# like an app with no controls at all. When that happens, the top-level window
# with the same title that DOES have content is read instead. The coordinates
# below still come from the original window's rectangle, so they stay in the
# same frame of reference click_window uses.

$clock = [System.Diagnostics.Stopwatch]::StartNew()
$script:count = 0
# Pure layout containers are walked THROUGH but never reported: a XAML app
# (Notepad 11, Settings) wraps its real controls in dozens of unnamed Panes,
# and reporting those spent the whole budget before anything the agent could
# act on was reached.
$layout = @('Pane', 'Custom', 'Group', 'Thumb', 'Separator', 'TitleBar', 'ScrollBar', 'Image')

function Walk($node, $depth) {{
    if ($script:count -ge {cap} -or $depth -gt {_UIA_MAX_DEPTH}) {{ return }}
    if ($clock.Elapsed.TotalSeconds -ge {_UIA_SECONDS}) {{ return }}
    try {{ $child = $walker.GetFirstChild($node) }} catch {{ return }}
    $siblings = 0
    while ($child -ne $null -and $siblings -lt 120) {{
        try {{ $cur = $child.Current }} catch {{ break }}
        $type = $cur.ControlType.ProgrammaticName -replace 'ControlType\.', ''
        $name = ($cur.Name -replace '\|', '/') -replace '\s+', ' '

        if (-not ($name -eq '' -and $layout -contains $type)) {{
            $state = @()
            if (-not $cur.IsEnabled) {{ $state += 'disabled' }}
            if ($cur.IsOffscreen) {{ $state += 'off screen' }}
            $value = ''
            try {{
                $vp = $child.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
                if ($vp) {{ $value = ($vp.Current.Value -replace '\|', '/') -replace '\s+', ' ' }}
            }} catch {{}}
            try {{
                $tp = $child.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern)
                if ($tp) {{ $state += $tp.Current.ToggleState.ToString().ToLower() }}
            }} catch {{}}
            try {{
                $sp = $child.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern)
                if ($sp -and $sp.Current.IsSelected) {{ $state += 'selected' }}
            }} catch {{}}
            $box = $cur.BoundingRectangle
            $x = -1; $y = -1; $w = 0; $h = 0
            if (-not [double]::IsInfinity($box.X) -and $box.Width -gt 0) {{
                $x = [int]$box.X; $y = [int]$box.Y; $w = [int]$box.Width; $h = [int]$box.Height
            }}
            if ($value.Length -gt 60) {{ $value = $value.Substring(0, 60) }}
            Write-Output ("N|" + $depth + "|" + $type + "|" + $name + "|" + $x + "|" + $y + "|" + $w + "|" + $h + "|" + ($state -join ',') + "|" + $value)
            $script:count++
        }}

        Walk $child ($depth + 1)
        if ($script:count -ge {cap}) {{ return }}
        try {{ $child = $walker.GetNextSibling($child) }} catch {{ break }}
        $siblings++
    }}
}}
Walk $root 1
Write-Output ("END|" + $script:count + "|" + [int]$clock.Elapsed.TotalSeconds)
"""
            out = _ps(script, timeout=_UIA_SECONDS + 25)
            if "NOWINDOW" in out:
                return (
                    f"No visible window matching '{window}' (waited 8s). Call list_windows to see "
                    "what is open, or open_app to start it."
                )
            if "NOTREE" in out:
                return (
                    f"'{window}' does not answer Windows' accessibility service, so it cannot be read "
                    "as text. Use see_window for this app instead."
                )

            header, nodes, ended, cloaked = "", [], "", False
            for line in out.splitlines():
                if line.startswith("WIN|"):
                    header = line
                elif line.startswith("N|"):
                    nodes.append(line)
                elif line.startswith("END|"):
                    ended = line
                elif line.startswith("CLOAKED|"):
                    cloaked = line.strip().endswith("1")
            if not header:
                return f"read_window_as_markdown could not read '{window}': {out[:300]}"

            _, proc, title, left, top, width, height = header.split("|", 6)
            origin_x, origin_y = int(left), int(top)
            lines = [
                f"# {title} ({proc})",
                f"Window is {width}x{height} px. Every position below is measured, not estimated — "
                f"pass it to click_window with window='{window}'.",
                "",
            ]

            listed = 0
            for line in nodes:
                try:
                    _, depth, ctype, name, x, y, w, h, state, value = line.split("|", 9)
                except ValueError:
                    continue
                name = name.strip()
                if ctype in _UIA_SKIP and not name:
                    continue
                if not name and ctype not in _UIA_ALWAYS:
                    continue
                indent = "  " * max(0, min(int(depth) - 1, 8))
                label = f'`{name}`' if name else f"(unnamed {ctype})"
                parts = [f"{indent}- {label} — {ctype}"]
                if state:
                    parts.append(f"[{state}]")
                if value:
                    parts.append(f'holds "{value.strip()}"')
                if int(x) >= 0 and int(w) > 0:
                    cx = int(x) + int(w) // 2 - origin_x
                    cy = int(y) + int(h) // 2 - origin_y
                    parts.append(f"at ({cx}, {cy})")
                lines.append(" ".join(parts))
                listed += 1

            real = sum(
                1 for line in lines[3:]
                if line.strip().startswith("- ") and not any(
                    f"`{word}`" in line.lower() for word in _UIA_WINDOW_FURNITURE
                )
            )
            if listed and real < 3:
                lines.append(
                    "\n(Only the window's own title-bar buttons are exposed - this app does not "
                    "publish its contents to Windows' accessibility service. Use see_window to look "
                    "at it instead.)"
                )
            if cloaked and not listed:
                return (
                    f"'{window}' is open but SUSPENDED by Windows (its window is cloaked), so it has no "
                    "live interface to read and is not being drawn on screen - see_window would photograph "
                    "whatever app is actually in front of it. Call focus_window to wake it, wait a moment, "
                    "then read it again."
                )
            if not listed:
                return (
                    f"'{title}' exposes no named controls through Windows' accessibility service "
                    "(apps that paint their own interface often do not). Use see_window for this app."
                )

            if ended:
                _, count, seconds = ended.split("|", 2)
                if int(count) >= cap:
                    lines.append(f"\n(stopped at {cap} controls — raise max_items for more)")
            logger.info("window_read_as_markdown", window=window[:40], controls=listed)
            return "\n".join(lines)

        return _run_in_worker(_async_run)


DESKTOP_TOOLS = [
    ListWindowsTool,
    FocusWindowTool,
    SeeWindowTool,
    ReadWindowTool,
    ClickWindowTool,
]
