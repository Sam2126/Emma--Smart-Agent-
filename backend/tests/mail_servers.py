"""
Small IMAP and SMTP servers on 127.0.0.1 for the email tool tests.

They speak just enough of each protocol for Python's own imaplib and smtplib
clients (literals, UID commands, APPENDUID, AUTH PLAIN, SIZE), so the tests
exercise the real client code and the real response shapes, and nothing ever
reaches a real mailbox. The IMAP server answers like Gmail: special-use flags
in LIST, X-GM-RAW search, and items after a literal in FETCH answers.
"""

from __future__ import annotations

import base64
import re
import socketserver
import threading
from dataclasses import dataclass, field
from datetime import datetime
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime


@dataclass
class StoredMessage:
    uid: int
    raw: bytes
    flags: set[str] = field(default_factory=set)

    @property
    def parsed(self):
        return BytesParser(policy=policy.default).parsebytes(self.raw)


@dataclass
class Mailbox:
    name: str
    flags: str = "\\HasNoChildren"
    messages: list[StoredMessage] = field(default_factory=list)
    next_uid: int = 1

    def add(self, raw: bytes, flags: set[str] | None = None) -> int:
        uid = self.next_uid
        self.next_uid += 1
        self.messages.append(StoredMessage(uid, raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"), set(flags or ())))
        return uid


def _tokens(text: str) -> list[str]:
    """IMAP arguments: atoms, "quoted strings" and (parenthesised lists) as one token each."""
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch == " ":
            i += 1
        elif ch == '"':
            j, buf = i + 1, []
            while text[j] != '"':
                if text[j] == "\\":
                    j += 1
                buf.append(text[j])
                j += 1
            out.append("".join(buf))
            i = j + 1
        elif ch == "(":
            depth, j = 0, i
            while True:
                depth += text[j] == "("
                depth -= text[j] == ")"
                j += 1
                if depth == 0:
                    break
            out.append(text[i:j])
            i = j
        else:
            j = text.find(" ", i)
            j = len(text) if j == -1 else j
            out.append(text[i:j])
            i = j
    return out


class FakeIMAPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, user: str, password: str) -> None:
        super().__init__(("127.0.0.1", 0), _IMAPHandler)
        self.user, self.password = user, password
        self.lock = threading.Lock()
        self.boxes = {
            "INBOX": Mailbox("INBOX"),
            "[Gmail]": Mailbox("[Gmail]", "\\HasChildren \\Noselect"),
            "[Gmail]/Sent Mail": Mailbox("[Gmail]/Sent Mail", "\\HasNoChildren \\Sent"),
            "[Gmail]/Drafts": Mailbox("[Gmail]/Drafts", "\\Drafts \\HasNoChildren"),
            "[Gmail]/Spam": Mailbox("[Gmail]/Spam", "\\HasNoChildren \\Junk"),
            "Work Projects": Mailbox("Work Projects"),  # listed as a literal
        }
        self.gmail = True
        self.logins = 0

    @property
    def port(self) -> int:
        return self.server_address[1]

    def start(self) -> "FakeIMAPServer":
        threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


class _IMAPHandler(socketserver.StreamRequestHandler):
    server: FakeIMAPServer

    def send(self, text: str | bytes) -> None:
        self.wfile.write(text if isinstance(text, bytes) else text.encode())

    def read_command(self) -> tuple[str, list[bytes]] | None:
        """One command line, with any literals ({n}) read in; literal values are returned separately."""
        line = self.rfile.readline()
        if not line:
            return None
        text = line.decode("utf-8", "replace").rstrip("\r\n")
        literals = []
        while True:
            match = re.search(r"\{(\d+)\}$", text)
            if not match:
                break
            self.send("+ go ahead\r\n")
            literals.append(self.rfile.read(int(match.group(1))))
            text = text[:match.start()] + "\x00LITERAL\x00" + self.rfile.readline().decode("utf-8", "replace").rstrip("\r\n")
        return text, literals

    def handle(self) -> None:
        self.box: Mailbox | None = None
        self.readonly = True
        self.send("* OK [CAPABILITY IMAP4rev1 UIDPLUS X-GM-EXT-1] Fake Gmail ready\r\n")
        while True:
            got = self.read_command()
            if got is None:
                return
            text, literals = got
            tag, _, rest = text.partition(" ")
            command, _, args = rest.partition(" ")
            command = command.upper()
            try:
                with self.server.lock:
                    done = self.dispatch(tag, command, args, literals)
            except Exception as e:  # a bug in the fake: say so in the protocol
                self.send(f"{tag} BAD fake server error {e!r}\r\n")
                continue
            if done:
                return

    def dispatch(self, tag: str, command: str, args: str, literals: list[bytes]) -> bool:
        srv = self.server
        if command == "CAPABILITY":
            self.send(f"* CAPABILITY IMAP4rev1 UIDPLUS X-GM-EXT-1\r\n{tag} OK done\r\n")
        elif command == "LOGIN":
            user, password = _tokens(args)[:2]
            if user == srv.user and password == srv.password:
                srv.logins += 1
                self.send(f"{tag} OK {user} authenticated (Success)\r\n")
            else:
                self.send(f"{tag} NO [AUTHENTICATIONFAILED] Invalid credentials (Failure)\r\n")
        elif command == "LIST":
            for box in srv.boxes.values():
                if box.name == "Work Projects":
                    self.send(f'* LIST ({box.flags}) "/" {{{len(box.name)}}}\r\n{box.name}\r\n')
                else:
                    self.send(f'* LIST ({box.flags}) "/" "{box.name}"\r\n')
            self.send(f"{tag} OK Success\r\n")
        elif command in ("SELECT", "EXAMINE"):
            name = _tokens(args)[0]
            box = srv.boxes.get("INBOX" if name.upper() == "INBOX" else name)
            if box is None or "Noselect" in box.flags:
                self.send(f"{tag} NO [NONEXISTENT] Unknown Mailbox: {name}\r\n")
                return False
            self.box, self.readonly = box, command == "EXAMINE"
            self.send(f"* FLAGS (\\Answered \\Flagged \\Draft \\Deleted \\Seen)\r\n* {len(box.messages)} EXISTS\r\n* 0 RECENT\r\n")
            self.send(f"* OK [UIDVALIDITY 7] UIDs valid.\r\n{tag} OK [{'READ-ONLY' if self.readonly else 'READ-WRITE'}] {name} selected. (Success)\r\n")
        elif command == "UID":
            sub, _, sub_args = args.partition(" ")
            self.uid_command(tag, sub.upper(), sub_args, literals)
        elif command == "APPEND":
            name = _tokens(args)[0]
            box = srv.boxes.get(name)
            flags = re.search(r"\(([^)]*)\)", args)
            uid = box.add(literals[0], set((flags.group(1) if flags else "").split()))
            self.send(f"{tag} OK [APPENDUID 7 {uid}] (Success)\r\n")
        elif command == "EXPUNGE":
            kept, seq = [], 1
            for message in self.box.messages:
                if "\\Deleted" in message.flags:
                    self.send(f"* {seq} EXPUNGE\r\n")  # later messages move down into this number
                else:
                    kept.append(message)
                    seq += 1
            self.box.messages = kept
            self.send(f"{tag} OK Success\r\n")
        elif command == "NOOP":
            self.send(f"{tag} OK Success\r\n")
        elif command == "LOGOUT":
            self.send(f"* BYE LOGOUT Requested\r\n{tag} OK 73 good day (Success)\r\n")
            return True
        else:
            self.send(f"{tag} BAD Unknown command {command}\r\n")
        return False

    # --- UID commands ---------------------------------------------------------

    def uid_set(self, text: str) -> list[StoredMessage]:
        wanted = set()
        for part in text.split(","):
            if ":" in part:
                a, b = part.split(":")
                hi = max(m.uid for m in self.box.messages) if b == "*" else int(b)
                wanted.update(range(int(a), hi + 1))
            else:
                wanted.add(int(part))
        return [m for m in self.box.messages if m.uid in wanted]

    def uid_command(self, tag: str, sub: str, args: str, literals: list[bytes]) -> None:
        if sub == "SEARCH":
            tokens = _tokens(args.replace("\x00LITERAL\x00", ""))
            values = list(literals)
            if tokens[:2] and tokens[0].upper() == "CHARSET":
                tokens = tokens[2:]
            if values:
                tokens.append(values[0].decode("utf-8"))
            hits = [m.uid for m in self.box.messages if self.matches(m, tokens)]
            self.send(f"* SEARCH {' '.join(map(str, hits))}\r\n{tag} OK SEARCH completed (Success)\r\n")
        elif sub == "FETCH":
            uid_text, _, items = args.partition(" ")
            for message in self.uid_set(uid_text):
                seq = self.box.messages.index(message) + 1
                self.fetch_one(seq, message, items.upper())
            self.send(f"{tag} OK Success\r\n")
        elif sub == "STORE":
            uid_text, mode, flags = args.split(" ", 2)
            for message in self.uid_set(uid_text):
                names = set(flags.strip("()").split())
                if mode.startswith("+"):
                    message.flags |= names
                elif mode.startswith("-"):
                    message.flags -= names
                seq = self.box.messages.index(message) + 1
                self.send(f"* {seq} FETCH (UID {message.uid} FLAGS ({' '.join(sorted(message.flags))}))\r\n")
            self.send(f"{tag} OK Success\r\n")
        else:
            self.send(f"{tag} BAD unsupported UID {sub}\r\n")

    def fetch_one(self, seq: int, message: StoredMessage, items: str) -> None:
        flags = " ".join(sorted(message.flags))
        if "BODY.PEEK[]" in items or "BODY[]" in items:
            if "BODY[]" in items and "PEEK" not in items:
                message.flags.add("\\Seen")
            self.send(f"* {seq} FETCH (UID {message.uid} BODY[] {{{len(message.raw)}}}\r\n")
            self.send(message.raw)
            self.send(")\r\n")
            return
        match = re.search(r"HEADER\.FIELDS \(([^)]*)\)", items)
        wanted = {f.lower() for f in match.group(1).split()} if match else set()
        header_block = message.raw.split(b"\r\n\r\n", 1)[0].decode("utf-8", "replace")
        kept, keep = [], False
        for line in header_block.split("\r\n"):
            if line[:1] in (" ", "\t"):
                if keep:
                    kept.append(line)
                continue
            keep = line.split(":", 1)[0].strip().lower() in wanted
            if keep:
                kept.append(line)
        data = ("\r\n".join(kept) + "\r\n\r\n").encode("utf-8")
        # Like Gmail: FLAGS come after the literal.
        self.send(f"* {seq} FETCH (UID {message.uid} RFC822.SIZE {len(message.raw)} "
                  f"BODY[HEADER.FIELDS ({match.group(1) if match else ''})] {{{len(data)}}}\r\n")
        self.send(data)
        self.send(f" FLAGS ({flags}))\r\n")

    def matches(self, message: StoredMessage, tokens: list[str]) -> bool:
        parsed = message.parsed

        def text_of(name: str) -> str:
            return str(parsed[name] or "").lower()

        i = 0

        def one() -> bool:
            nonlocal i
            key = tokens[i].upper()
            i += 1
            if key == "ALL":
                return True
            if key == "UNSEEN":
                return "\\Seen" not in message.flags
            if key == "OR":
                a = one()
                b = one()
                return a or b
            value = tokens[i]
            i += 1
            if key in ("FROM", "TO", "CC", "SUBJECT"):
                return value.lower() in text_of(key.lower())
            if key == "TEXT":
                return value.lower() in message.raw.decode("utf-8", "replace").lower() or value.lower() in text_of("subject")
            if key == "SINCE":
                day = datetime.strptime(value, "%d-%b-%Y").date()
                return parsedate_to_datetime(str(parsed["date"])).date() >= day
            if key == "UID":
                return message.uid == int(value)
            if key == "HEADER":
                header_value = tokens[i]
                i += 1
                return header_value.lower() in text_of(value.lower())
            if key == "X-GM-RAW":
                ok = True
                for word in value.split():
                    if word == "has:attachment":
                        ok &= str(parsed.get_content_type()) == "multipart/mixed"
                    elif word.startswith("from:"):
                        ok &= word[5:].lower() in text_of("from")
                    else:
                        whole = (text_of("subject") + " " + message.raw.decode("utf-8", "replace").lower())
                        ok &= word.lower() in whole
                return ok
            raise ValueError(f"search key {key}")

        result = True
        while i < len(tokens):
            result &= one()
        return result


class FakeSMTPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, user: str, password: str, size_limit: int = 5_000_000) -> None:
        super().__init__(("127.0.0.1", 0), _SMTPHandler)
        self.user, self.password = user, password
        self.size_limit = size_limit
        self.refuse: set[str] = set()
        self.messages: list[dict] = []

    @property
    def port(self) -> int:
        return self.server_address[1]

    def start(self) -> "FakeSMTPServer":
        threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


class _SMTPHandler(socketserver.StreamRequestHandler):
    server: FakeSMTPServer

    def reply(self, text: str) -> None:
        self.wfile.write((text + "\r\n").encode())

    def handle(self) -> None:
        srv = self.server
        authed, sender, rcpts = False, "", []
        self.reply("220 fake.smtp ESMTP ready")
        while True:
            line = self.rfile.readline()
            if not line:
                return
            text = line.decode().rstrip("\r\n")
            verb = text.split(" ", 1)[0].upper()
            if verb in ("EHLO", "HELO"):
                self.reply(f"250-fake.smtp at your service\r\n250-SIZE {srv.size_limit}\r\n250-8BITMIME\r\n250 AUTH PLAIN")
            elif verb == "AUTH":
                _, mech, blob = text.split(" ", 2)
                parts = base64.b64decode(blob).split(b"\x00")
                if mech.upper() == "PLAIN" and parts[1].decode() == srv.user and parts[2].decode() == srv.password:
                    authed = True
                    self.reply("235 2.7.0 Accepted")
                else:
                    self.reply("535 5.7.8 Username and Password not accepted. For more information, go to https://support.google.com/mail/?p=BadCredentials")
            elif verb == "MAIL":
                if not authed:
                    self.reply("530 5.7.0 Authentication Required")
                    continue
                sender = re.search(r"<([^>]*)>", text).group(1)
                rcpts = []
                self.reply("250 2.1.0 OK")
            elif verb == "RCPT":
                address = re.search(r"<([^>]*)>", text).group(1)
                if address in srv.refuse:
                    self.reply(f"550 5.1.1 The email account that you tried to reach does not exist: {address}")
                else:
                    rcpts.append(address)
                    self.reply("250 2.1.5 OK")
            elif verb == "DATA":
                self.reply("354 Go ahead")
                chunks = []
                while True:
                    chunk = self.rfile.readline()
                    if chunk == b".\r\n":
                        break
                    chunks.append(chunk[1:] if chunk.startswith(b"..") else chunk)
                srv.messages.append({"from": sender, "rcpts": list(rcpts), "data": b"".join(chunks)})
                self.reply("250 2.0.0 OK queued")
            elif verb == "RSET":
                sender, rcpts = "", []
                self.reply("250 2.1.5 Flushed")
            elif verb == "NOOP":
                self.reply("250 2.0.0 OK")
            elif verb == "QUIT":
                self.reply("221 2.0.0 closing connection")
                return
            else:
                self.reply("502 5.5.1 Unrecognized command")
