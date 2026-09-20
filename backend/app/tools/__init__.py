"""
Tools the agent can call.

  browser.py - drive the user's Chrome through Playwright (navigate, perceive,
               see_page vision, type, click, keys, scroll, extract, login gate)
  local.py   - files, folders, apps, PowerShell, send_keys on this computer
  desktop.py - desktop app GUI: list/focus windows, see_window vision, click_window

Modules are imported lazily by app.agent.toolkit so importing one tool family
never drags in the others.
"""
