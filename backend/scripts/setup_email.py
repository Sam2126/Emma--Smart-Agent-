"""
Set up the agent's email tools: asks for the mailbox address and its app
password, checks that both reading (IMAP) and sending (SMTP) sign in, and
writes them to the project's .env file. Nothing is sent.

Run it with setup_email.bat in the project folder.

Gmail: the normal Google password is refused. Create an App Password at
Google Account > Security > 2-Step Verification > App passwords (2-Step
Verification must be on), and paste the 16 letters here.
"""

from __future__ import annotations

import getpass
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.tools.mail import MailError, check_login, mail_config, provider_for  # noqa: E402

ENV_FILE = BACKEND.parent / ".env"


def env_value(value: str) -> str:
    """A .env value that python-dotenv reads back exactly."""
    if value and re.fullmatch(r"[A-Za-z0-9@._+-]+", value):
        return value
    if "'" not in value:
        return f"'{value}'"
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_env(values: dict[str, str]) -> None:
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    remaining = dict(values)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if key in remaining and not line.lstrip().startswith("#"):
            lines[i] = f"{key}={env_value(remaining.pop(key))}"
    if remaining:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# --- Email (setup_email.bat) ---")
        lines += [f"{key}={env_value(value)}" for key, value in remaining.items()]
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _Settings:
    def __init__(self, address: str, password: str, name: str) -> None:
        self.email_address = address
        self.email_app_password = password
        self.email_display_name = name
        self.email_imap_host = ""
        self.email_imap_port = 993
        self.email_smtp_host = ""
        self.email_smtp_port = 0


def main() -> int:
    print("Email setup for the Self-Improving Agent")
    print("-----------------------------------------")
    address = input("Your email address: ").strip()
    if provider_for(address) == "gmail":
        print("\nGmail needs an App Password, not your Google password:")
        print("  https://myaccount.google.com/apppasswords  (2-Step Verification must be on)")
        print("  Create one named 'Self-Improving Agent' and paste the 16 letters below.\n")
    password = getpass.getpass("App password (hidden while you type): ").strip()
    name = input("Your name as recipients should see it (optional): ").strip()

    try:
        cfg = mail_config(_Settings(address, password, name))
    except MailError as e:
        print(f"\n{e}")
        return 1
    print(f"\nChecking the sign-in at {cfg.imap_host} and {cfg.smtp_host} (nothing is sent)...")
    problems = check_login(cfg)
    if problems:
        print("\nThe sign-in did not work:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nNothing was saved. Fix the above and run setup_email.bat again.")
        return 1

    values = {"EMAIL_ADDRESS": address, "EMAIL_APP_PASSWORD": cfg.password}
    if name:
        values["EMAIL_DISPLAY_NAME"] = name
    write_env(values)
    print(f"\nDone: reading and sending both work for {address}. Saved to {ENV_FILE}.")
    print("The agent can now read, draft and send your email (no restart needed).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled; nothing was saved.")
        sys.exit(1)
