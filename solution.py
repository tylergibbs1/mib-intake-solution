#!/usr/bin/env python3
"""MIB Doc Challenge solution: offline PDF packet extraction + adjudication.

Per PDF:
  1. Extract visible text with PyMuPDF (drop pure-white hidden spans; keep them
     separately as an injection signal). Colored spans are kept as stamps.
  2. Pages without a usable text layer are rendered at 200 dpi, binarized at
     multiple thresholds (dropping faint ghosting and light baked-in
     injections), deskewed, and OCRed with tesseract.
  3. Lines are parsed into labeled fields per document type; values are
     fuzzy-corrected against the closed challenge vocabularies.
  4. Evidence is merged with field-manual precedence and adjudicated by the
     policy rules; confidence is calibrated per decision path.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import fitz

fitz.TOOLS.mupdf_display_errors(False)

# ---------------------------------------------------------------------------
# Vocabularies (public field manual + public training labels: domain
# vocabularies, not per-case answers).
# ---------------------------------------------------------------------------

SPECIES = [
    "TRIANGULAN", "JOVIAN_GASFORM", "CENTAURI_SYNTH", "LUNA_SECURID",
    "KAIJU_MICRO", "ORION_GRAYS", "ALPHA_DRACONIAN", "SIRIUS_AVIAN",
    "VENUSIAN_MYCELIAL", "AQUARIAN_MANTIS", "ARCTURIAN", "ANDROMEDAN",
]

HOME_WORLDS = [
    "Barnard-c", "Eris Relay", "Europa Station", "Gliese-581g", "Kepler-186f",
    "Luyten-b", "Mars Dome-7", "Proxima-b", "Sirius Outpost", "TRAPPIST-1e",
    "Titan Freeport", "Wolf-1061c", "Zeta Reticuli",
]

VISA_CLASSES = ["XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7"]

PURPOSES = [
    "reactor maintenance", "field repair", "medical consult", "research",
    "cultural exchange", "translation", "archive audit", "xenobotany",
    "diplomatic", "transit",
]

NAME_PREFIXES = ["Ari", "Ixo", "Lu", "Mira", "Nex", "Ori", "Qor", "Sol",
                 "Tek", "Vee", "Xan", "Za"]
NAME_SUFFIXES = ["dane", "ix", "kesh", "mora", "nax", "quell", "rix", "tari",
                 "ul", "vara", "voss", "zarn"]
NAME_TOKENS = [p + s for p in NAME_PREFIXES for s in NAME_SUFFIXES]

RISK_FLAG_VALUES = [
    "memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red",
    "identity_conflict", "sponsor_mismatch", "illegible_biometrics",
    "rescinded_denial", "none",
]
DISQUALIFYING_FLAGS = {"memory_tampering", "planetary_embargo",
                       "active_warrant", "biohazard_red"}

FEE_VALUES = ["paid", "waived", "unpaid", "unknown"]

REVOKED_SPONSORS = {"SPN-0007", "SPN-0139", "SPN-4040"}

DOC_TYPES = {
    "intake_form": "FORM I-8090: Extraterrestrial Work Authorization Intake",
    "fee_receipt": "MIB Fee Receipt",
    "registry": "Planetary Registry Extract",
    "biometric": "FORM B-13: Biometric Scan Slip",
    "sponsor_letter": "Sponsor Attestation Letter",
    "adjudicator_note": "Manual Adjudicator Note",
}

# canonical field labels seen on documents -> (field, vocab)
LINE_LABELS = {
    "case id": "case_id",
    "applicant": "applicant_name",
    "registry name": "applicant_name",
    "species code": "species_code",
    "species match": "species_match",
    "home world": "home_world",
    "visa class": "visa_class",
    "sponsor id": "sponsor_id",
    "arrival date": "arrival_date",
    "declared purpose": "declared_purpose",
    "purpose": "declared_purpose",
    "fee status": "fee_status",
    "amount": "amount",
    "waiver code": "waiver_code",
    "registry status": "registry_status",
    "biometric confidence": "biometric_confidence",
    "observed flags": "observed_flags",
    "finding": "finding",
    "reason": "reason",
}

STAMP_WORDS = {"DENIED", "APPROVED", "REVIEW", "RESCINDED", "VOID",
               "SAMPLE DENIAL", "FILED", "ARCHIVE", "COPY ARTIFACT"}


def span_rgb(color):
    return (color >> 16) & 255, (color >> 8) & 255, color & 255


def norm(s):
    return " ".join((s or "").strip().split())


def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()


from functools import lru_cache

@lru_cache(maxsize=500000)
def _fuzzy_cached(value_l, vocab):
    best, best_r = None, 0.0
    n = len(value_l)
    for v in vocab:
        vl = v.lower()
        if abs(len(vl) - n) > max(3, int(0.45 * max(n, len(vl)))):
            continue
        m = SequenceMatcher(None, value_l, vl)
        if m.real_quick_ratio() <= best_r or m.quick_ratio() <= best_r:
            continue
        r = m.ratio()
        if r > best_r:
            best_r, best = r, v
    return best, best_r


def fuzzy_best(value, vocab, min_ratio=0.65):
    """Best fuzzy match of value against vocab: (match, score) or (None, 0)."""
    value_l = norm(value).lower()
    if not value_l:
        return None, 0.0
    best, best_r = _fuzzy_cached(value_l, tuple(vocab))
    if best_r >= min_ratio:
        return best, best_r
    return None, best_r


def fuzzy_name(value):
    """Match 'First Last' against the 144-token name vocabulary."""
    value = norm(value)
    if not value:
        return None, 0.0
    parts = value.split()
    if len(parts) < 2:
        # try splitting a fused token in half
        if len(value) >= 8:
            parts = [value[: len(value) // 2], value[len(value) // 2:]]
        else:
            return None, 0.0
    first_raw, last_raw = parts[0], " ".join(parts[1:]).replace(" ", "")
    f, fr = fuzzy_best(first_raw, NAME_TOKENS, 0.45)
    l, lr = fuzzy_best(last_raw, NAME_TOKENS, 0.45)
    if f and l:
        return f + " " + l, (fr + lr) / 2
    return None, 0.0


# ---------------------------------------------------------------------------
# Text-layer extraction
# ---------------------------------------------------------------------------

DARK_COLORS = {(0, 0, 0), (17, 17, 17), (34, 34, 34), (51, 51, 51),
               (102, 102, 102)}


def extract_text_layer(page):
    """Split the text layer into visible lines, stamp lines, hidden lines."""
    visible, stamps, hidden = [], [], []
    data = page.get_text("dict")
    for block in data["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            vis, stamp, hid = [], [], []
            for span in line["spans"]:
                rgb = span_rgb(span["color"])
                text = span["text"]
                if rgb[0] > 240 and rgb[1] > 240 and rgb[2] > 240:
                    hid.append(text)
                elif rgb in DARK_COLORS:
                    vis.append(text)
                else:
                    stamp.append(text)
            if vis and norm("".join(vis)):
                visible.append((norm("".join(vis)), round(line["bbox"][1], 1)))
            if stamp and norm("".join(stamp)):
                stamps.append(norm("".join(stamp)))
            if hid and norm("".join(hid)):
                hidden.append(norm("".join(hid)))
    return visible, stamps, hidden


FOOTER_RE = re.compile(r"^(Packet MIB-\d+ / page \d+|Synthetic hiring challenge document|MIB-\d+ \| MIB Eyes Only)$")


def is_scanned_page(visible_lines):
    body = [t for t, _ in visible_lines if not FOOTER_RE.match(t)]
    return len(body) <= 1


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def sauvola_binarize(a, w=31, k=0.2, R=128.0):
    """Sauvola local adaptive binarization (integral-image implementation).

    Standard technique for degraded documents with ghosting/bleed-through:
    the threshold adapts to local mean/std, keeping darker primary strokes
    and dropping lighter ghost copies. Returns bool ink mask.
    """
    import numpy as np

    af = a.astype(np.float64)
    pad_lo = w // 2
    pad_hi = w - pad_lo
    ap = np.pad(af, ((pad_lo, pad_hi), (pad_lo, pad_hi)), mode="edge")
    ii = np.zeros((ap.shape[0] + 1, ap.shape[1] + 1), np.float64)
    ii[1:, 1:] = ap.cumsum(0).cumsum(1)
    ii2 = np.zeros_like(ii)
    ii2[1:, 1:] = (ap ** 2).cumsum(0).cumsum(1)
    H, W = a.shape
    S = ii[w:H + w, w:W + w] - ii[0:H, w:W + w] - ii[w:H + w, 0:W] + ii[0:H, 0:W]
    S2 = ii2[w:H + w, w:W + w] - ii2[0:H, w:W + w] - ii2[w:H + w, 0:W] + ii2[0:H, 0:W]
    n = float(w * w)
    m = S / n
    sd = np.sqrt(np.maximum(S2 / n - m ** 2, 0))
    T = m * (1 + k * (sd / R - 1))
    return a < T


def estimate_skew(binary_small):
    import numpy as np

    ys, xs = np.nonzero(binary_small)
    if len(ys) < 100:
        return 0.0
    best_a, best_s = 0.0, -1.0
    for angle in np.arange(-4.0, 4.01, 0.25):
        rad = np.deg2rad(angle)
        proj = (ys * np.cos(rad) - xs * np.sin(rad)).astype(np.int32)
        proj -= proj.min()
        c = np.bincount(proj).astype(np.float64)
        s = float((c ** 2).sum())
        if s > best_s:
            best_s, best_a = s, float(angle)
    return best_a


def run_tesseract(png_path, psm, dpi):
    try:
        proc = subprocess.run(
            ["tesseract", png_path, "-", "--psm", str(psm), "--dpi", str(dpi)],
            capture_output=True, text=True, timeout=60,
        )
        return proc.stdout
    except Exception:
        return ""


def binary_erode(b, iterations=1):
    """3x3 cross erosion, used to un-fuse ink-bled bold text."""
    import numpy as np

    for _ in range(iterations):
        m = b.copy()
        m &= np.roll(b, 1, 0)
        m &= np.roll(b, -1, 0)
        m &= np.roll(b, 1, 1)
        m &= np.roll(b, -1, 1)
        b = m
    return b


def ocr_page_variants(page, dpi=250):
    """OCR a scanned page with several binarization variants.

    Returns list of text outputs (one per variant). If the standard variants
    recognize almost nothing, an erosion escalation pass targets ink-bled
    pages whose fused strokes defeat plain binarization.
    """
    import numpy as np
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

    variants = [
        sauvola_binarize(a, w=31, k=0.2),
        sauvola_binarize(a, w=31, k=0.34),
        a < 160,
    ]
    outputs = []
    tmpdir = tempfile.mkdtemp()
    try:
        def run_variant(binary, name):
            angle = estimate_skew(binary[::4, ::4])
            img = Image.fromarray(np.where(binary, 0, 255).astype("uint8"))
            if abs(angle) >= 0.25:
                img = img.rotate(-angle, expand=False, fillcolor=255)
            png = os.path.join(tmpdir, f"{name}.png")
            img.save(png)
            return run_tesseract(png, 11, dpi)

        for i, binary in enumerate(variants):
            outputs.append(run_variant(binary, f"v{i}"))

        recognized = sum(ocr_quality(o) * len(o) for o in outputs)
        if recognized < 120:
            base = a < 128
            for it in (1, 2):
                outputs.append(run_variant(binary_erode(base, it), f"e{it}"))
    finally:
        for f in Path(tmpdir).glob("*"):
            f.unlink()
        os.rmdir(tmpdir)
    return outputs


# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

DOC_TITLE_KEYS = {
    "intake_form": "form i-8090 extraterrestrial work authorization intake",
    "fee_receipt": "mib fee receipt",
    "registry": "planetary registry extract",
    "biometric": "form b-13 biometric scan slip",
    "sponsor_letter": "sponsor attestation letter",
    "adjudicator_note": "manual adjudicator note",
}


def classify_doc(lines):
    """Classify a page's document type from its first meaningful lines."""
    best_type, best_r = None, 0.55
    for text in lines[:6]:
        t = re.sub(r"[^a-z0-9\- ]", "", text.lower())
        for dt, title in DOC_TITLE_KEYS.items():
            r = sim(t, title)
            if r > best_r:
                best_r, best_type = r, dt
    return best_type


CASE_ID_RE = re.compile(r"M[I1l]B[-–—:.\s]*[0O]*(\d{4,6})")
SPN_RE = re.compile(r"[S5$][PFR]\s?[NHM][-–—:.\s]*([0-9O l]{4,6})")
DATE_RE = re.compile(r"(2[0O]\d{2})[-–—.\s/]*([0-9O]{1,2})[-–—.\s/]*([0-9O]{1,2})")


def normalize_spn(m):
    digits = re.sub(r"[^0-9Ol]", "", m.group(1)).replace("O", "0").replace("l", "1")
    if len(digits) == 4:
        return "SPN-" + digits
    return None


def date_candidates(text):
    """Parse date-ish strings tolerantly. Returns [(iso_date, score)].

    OCR digit confusions are repaired toward the challenge's plausible
    ranges; repaired candidates carry a reduced score so a clean read from
    another page wins the vote.
    """
    out = []
    cleaned = text.replace("O", "0").replace("o", "0")
    for m in DATE_RE.finditer(cleaned):
        y, mo, d = m.group(1), m.group(2), m.group(3)
        score = 1.0
        if y not in ("2025", "2026"):
            # year almost surely 2025/2026; 6<->8 is the dominant confusion
            y2 = "2026" if y[-1] in "0689" else "2025"
            y, score = y2, score * 0.45
        mi = int(mo)
        if not 1 <= mi <= 12:
            if int(mo[-1]) <= 2 and len(mo) == 2:
                mo, score = "1" + mo[-1], score * 0.6
            else:
                mo, score = "0" + mo[-1], score * 0.6
            if not 1 <= int(mo) <= 12:
                continue
        di = int(d)
        if not 1 <= di <= 31:
            if len(d) == 2 and int("2" + d[-1]) <= 29:
                d, score = "2" + d[-1], score * 0.6
            else:
                d, score = "0" + d[-1] if d[-1] != "0" else "10", score * 0.6
            if not 1 <= int(d) <= 31:
                continue
        # calendar-validate; clamp repaired days that overflow the month
        import datetime
        try:
            datetime.date(int(y), int(mo), int(d))
        except ValueError:
            last = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31,
                    9: 30, 10: 31, 11: 30, 12: 31}[int(mo)]
            d, score = str(last), score * 0.6
        out.append((f"{y}-{mo.zfill(2)}-{d.zfill(2)}", score))
    return out


def parse_label_value_lines(lines):
    """Parse 'Label: value' or label/value alternating lines into fields.

    Returns list of (field, raw_value) in order of appearance.
    """
    results = []
    label_names = list(LINE_LABELS)
    i = 0
    while i < len(lines):
        text = lines[i]
        m = re.match(r"^(.{2,26}?)\s*[:;.=]\s*(.+)$", text)
        matched = False
        if m:
            label_clean = re.sub(r"[^a-z ]", "", m.group(1).lower()).strip()
            best, br = fuzzy_best(label_clean, label_names, 0.66)
            if best:
                value = m.group(2).strip()
                results.append((LINE_LABELS[best], value))
                matched = True
        if not matched:
            # label-only line followed by a value line (text-layer tables)
            label_clean = re.sub(r"[^a-z ]", "", text.lower()).strip()
            best, br = fuzzy_best(label_clean, label_names, 0.82)
            if best and i + 1 < len(lines):
                results.append((LINE_LABELS[best], lines[i + 1].strip()))
                i += 2
                continue
        i += 1
    return results


def clean_field_values(pairs, page_case_id=None):
    """Fuzzy-correct raw parsed values into canonical vocab values.

    Returns dict field -> (value, score).
    """
    out = {}

    def put(field, value, score):
        if value is None:
            return
        if field not in out or score > out[field][1]:
            out[field] = (value, round(score, 3))

    for field, raw in pairs:
        raw = norm(raw)
        if not raw:
            continue
        if field == "case_id":
            m = CASE_ID_RE.search(raw)
            if m:
                digits = m.group(1).zfill(6)
                put(field, "MIB-" + digits, 1.0)
        elif field in ("applicant_name",):
            v, r = fuzzy_name(raw)
            put(field, v, r)
        elif field in ("species_code", "species_match"):
            v, r = fuzzy_best(raw.replace(" ", "_"), SPECIES, 0.45)
            if not v:
                v, r = fuzzy_best(raw, SPECIES, 0.45)
            put(field, v, r)
        elif field == "home_world":
            v, r = fuzzy_best(raw, HOME_WORLDS, 0.45)
            put(field, v, r)
        elif field == "visa_class":
            v, r = fuzzy_best(raw, VISA_CLASSES, 0.5)
            put(field, v, r)
        elif field == "sponsor_id":
            m = SPN_RE.search(raw.upper())
            spn = normalize_spn(m) if m else None
            if spn:
                put(field, spn, 1.0)
            elif "blank" in raw.lower():
                put(field, "__BLANK__", 1.0)
        elif field == "arrival_date":
            for iso, score in date_candidates(raw):
                put(field, iso, score)
        elif field == "declared_purpose":
            v, r = fuzzy_best(raw, PURPOSES, 0.45)
            put(field, v, r)
        elif field == "fee_status":
            v, r = fuzzy_best(raw, FEE_VALUES, 0.7)
            put(field, v, r)
        elif field == "observed_flags":
            flags = []
            for part in re.split(r"[|,;/ ]+", raw.lower()):
                if not part or part in ("none", "nane", "mone"):
                    continue
                v, r = fuzzy_best(part, RISK_FLAG_VALUES, 0.7)
                if v and v != "none":
                    flags.append(v)
            put(field, "|".join(sorted(set(flags))) if flags else "none", 1.0)
        elif field == "finding":
            v, r = fuzzy_best(raw.upper().replace(" ", "_"),
                              ["APPROVED", "DENIED", "NEEDS_REVIEW"], 0.5)
            put(field, v, r)
        elif field in ("registry_status", "reason", "amount", "waiver_code",
                       "biometric_confidence"):
            put(field, raw, 1.0)
    return out


WORDISH_RE = re.compile(r"[A-Za-z_][A-Za-z_\-]+")


def scan_line_values(lines):
    """Value-driven extraction: find recognizable values anywhere in lines.

    OCR often destroys the 'Label:' part while the value survives. Values in
    this challenge come from closed vocabularies or strict formats, so we can
    recognize them directly with tighter thresholds than labeled parsing.
    Returns list of (field, value, score).
    """
    found = []
    for line in lines:
        lo = line.lower()
        up = line.upper()
        m = SPN_RE.search(up)
        if m:
            spn = normalize_spn(m)
            if spn:
                found.append(("sponsor_id", spn, 0.9))
        m = CASE_ID_RE.search(up)
        if m:
            found.append(("case_id", "MIB-" + m.group(1).zfill(6), 0.9))
        for iso, score in date_candidates(line):
            found.append(("arrival_date", iso, score * 0.9))
        # species codes are distinctive uppercase tokens
        tokens = WORDISH_RE.findall(line)
        for tok in tokens:
            if len(tok) >= 6:
                v, r = fuzzy_best(tok.replace("-", "_"), SPECIES, 0.72)
                if v:
                    found.append(("species_code", v, r * 0.95))
        # two-token species like "JOVIAN GASFORM" OCRed with space
        for i in range(len(tokens) - 1):
            two = tokens[i] + "_" + tokens[i + 1]
            if len(two) >= 9:
                v, r = fuzzy_best(two, SPECIES, 0.78)
                if v:
                    found.append(("species_code", v, r * 0.95))
        # home worlds
        for i, tok in enumerate(tokens):
            for cand in (tok, tokens[i] + " " + tokens[i + 1] if i + 1 < len(tokens) else tok):
                if len(cand) >= 6:
                    v, r = fuzzy_best(cand.replace("_", " "), HOME_WORLDS, 0.74)
                    if v:
                        found.append(("home_world", v, r * 0.92))
        # applicant names: adjacent pair of name-vocabulary tokens
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            if 4 <= len(a) <= 12 and 4 <= len(b) <= 12:
                fa, ra = fuzzy_best(a, NAME_TOKENS, 0.7)
                fb, rb = fuzzy_best(b, NAME_TOKENS, 0.7)
                if fa and fb:
                    found.append(("applicant_name", fa + " " + fb,
                                  (ra + rb) / 2 * 0.95))
        # visa classes: exact-ish first (XW-1 vs XW-2 differ by one digit)
        m = re.search(r"\b(XW|X\W?W)\s?[-–—]?\s?([12])\b", up)
        if m:
            found.append(("visa_class", "XW-" + m.group(2), 0.85))
        if re.search(r"\bD[Il1]P\s?[-–—]?\s?1?\b", up):
            found.append(("visa_class", "DIP-1", 0.8))
        if re.search(r"\bMED\s?[-–—]?\s?3?\b", up):
            found.append(("visa_class", "MED-3", 0.8))
        if re.search(r"\bTRANS[Il1]T\s?[-–—]?\s?7?\b", up):
            found.append(("visa_class", "TRANSIT-7", 0.8))
        # fuzzy fallback for OCR-mangled visa tokens like "VED-3", "XVK1";
        # the trailing digit must survive for the short classes
        for m in re.finditer(r"\b([A-Z]{2,8})\s?[-–—]?\s?([0-9])\b", up):
            tok, digit = m.group(1), m.group(2)
            for vc in VISA_CLASSES:
                letters, vd = vc.rsplit("-", 1)
                if digit == vd and sim(tok, letters) >= 0.6 and len(letters) >= 3:
                    found.append(("visa_class", vc, 0.6 * sim(tok, letters)))
        # declared purpose phrases
        if "purpose" in lo or "expected on earth for" in lo:
            tail = re.split(r"purpose\w*\s*[:;.=]?\s*|expected on earth for\s*",
                            lo, maxsplit=1, flags=re.IGNORECASE)[-1]
            v, r = fuzzy_best(tail.strip().rstrip("."), PURPOSES, 0.55)
            if v:
                found.append(("declared_purpose", v, r * 0.9))
        else:
            v, r = fuzzy_best(lo.strip().rstrip("."), PURPOSES, 0.83)
            if v:
                found.append(("declared_purpose", v, r * 0.85))
        # fee status: the label word and the value word both get mangled, so
        # fuzzy-match every token on fee-ish lines against the fee vocabulary
        if re.search(r"\bfee\b|\bstatus\b|stalus|slatus|f\w?e st", lo):
            for fv in ("paid", "waived", "unpaid", "unknown"):
                if re.search(r"\b" + fv + r"\b", lo):
                    found.append(("fee_status", fv, 0.85))
            for tok in re.findall(r"[a-z]{3,9}", lo):
                if tok in ("fee", "status", "case", "receipt", "amount",
                           "waiver", "code", "mib"):
                    continue
                v, r = fuzzy_best(tok, FEE_VALUES, 0.8)
                if v:
                    found.append(("fee_status", v, r * 0.9))
        if re.search(r"d[il1]p\s?[-–—]?\s?wa[il1]ver|hardship", lo):
            found.append(("dip_waiver", "waived", 0.9))
        if re.search(r"\$\s?0[.,]00", line):
            found.append(("fee_hint_zero", "waived", 0.4))
        if re.search(r"\$\s?\d{3}[.,]\d{2}", line):
            found.append(("fee_hint_amount", "paid", 0.4))
        # risk flag tokens anywhere (also as OCR-split two-word forms)
        for tok in tokens:
            if len(tok) >= 9:
                v, r = fuzzy_best(tok.lower(), RISK_FLAG_VALUES[:-1], 0.78)
                if v:
                    found.append(("flag_token", v, r))
        for i in range(len(tokens) - 1):
            two = (tokens[i] + "_" + tokens[i + 1]).lower()
            if len(two) >= 11:
                v, r = fuzzy_best(two, RISK_FLAG_VALUES[:-1], 0.8)
                if v:
                    found.append(("flag_token", v, r))
        # registry status
        if re.search(r"embargo", lo):
            found.append(("registry_status", "EMBARGO REVIEW", 0.8))
    return found


CORRECTION_RE = re.compile(
    r"manual correction[^a-z]{0,4}[:;]?\s*(.+)", re.IGNORECASE)


def parse_corrections(all_text):
    """Signed manual corrections override form fields (precedence rule 1)."""
    out = {}
    for m in CORRECTION_RE.finditer(all_text):
        rest = m.group(1)
        spn = SPN_RE.search(rest.upper())
        if "sponsor" in rest.lower() and spn:
            out["sponsor_id"] = "SPN-" + spn.group(1)
        date = DATE_RE.search(rest)
        if ("arrival" in rest.lower() or "date" in rest.lower()) and date:
            y, mo, d = date.groups()
            out["arrival_date"] = f"{y}-{mo.zfill(2)}-{d.zfill(2)}"
        for vc in VISA_CLASSES:
            if vc.lower() in rest.lower():
                out.setdefault("visa_class", vc)
        name, r = fuzzy_name(rest)
        if ("applicant" in rest.lower() or "name" in rest.lower()) and name:
            out["applicant_name"] = name
    return out


# Adjudicator-note reason templates observed on clean text-layer notes.
# Garbled scans are fuzzy-matched against these to recover the finding.
NOTE_TEMPLATES = [
    ("clean or exception-qualified packet", "APPROVED"),
    ("approval supported by surviving visible evidence", "APPROVED"),
    ("review-only risk flag present", "NEEDS_REVIEW"),
    ("packet contains damaged or contradictory visible evidence", "NEEDS_REVIEW"),
    ("arrival date missing from trusted visible evidence", "NEEDS_REVIEW"),
    ("fee status unknown", "NEEDS_REVIEW"),
    ("disqualifying risk flag", "DENIED"),
    ("embargo home world", "DENIED"),
    ("transit class cannot authorize declared work", "DENIED"),
    ("denial supported by damaged registry evidence", "DENIED"),
]

FINDING_WORDS = [("DENIED", "denied"), ("APPROVED", "approved"),
                 ("NEEDS_REVIEW", "needs_review"), ("NEEDS_REVIEW", "review")]


def parse_scanned_note(lines):
    """Recover an adjudicator-note finding from noisy OCR lines.

    Returns (finding, score) or (None, 0). APPROVED needs a higher bar than
    DENIED/REVIEW so OCR noise cannot cause a false approval.
    """
    def clean(s):
        return " ".join(re.sub(r"[^a-z ]", " ", s.lower()).split())

    candidates = [clean(l) for l in lines if clean(l)]
    pairs = [candidates[i] + " " + candidates[i + 1]
             for i in range(len(candidates) - 1)]
    best_finding, best_score = None, 0.0
    for text in candidates + pairs:
        if len(text) < 8:
            continue
        for template, finding in NOTE_TEMPLATES:
            r = sim(text, template)
            # also allow the template to appear as a substring-ish region
            if len(text) > len(template) + 8:
                for start in range(0, len(text) - len(template), 6):
                    r = max(r, sim(text[start:start + len(template) + 4], template))
            if r > best_score:
                best_score, best_finding = r, finding
    # fuzzy 'Finding: X' detection
    for text in candidates:
        words = text.split()
        for i, w in enumerate(words):
            if sim(w, "finding") >= 0.65 and i + 1 < len(words):
                rest = " ".join(words[i + 1:i + 3])
                for finding, word in FINDING_WORDS:
                    r = sim(rest[: len(word) + 2], word)
                    if r >= 0.6 and r > best_score:
                        best_score, best_finding = r, finding
    if best_finding == "APPROVED" and best_score < 0.62:
        return None, 0.0
    if best_score < 0.52:
        return None, 0.0
    return best_finding, round(best_score, 3)


NOTE_FINDING_RE = re.compile(
    r"(?:finding|fndng|frding)[^a-z]{0,4}[:;]?\s*(approved|denied|needs.?review|review)",
    re.IGNORECASE)


def parse_note_text(all_text):
    """Pull Finding/Reason out of adjudicator-note free text (OCR tolerant)."""
    result = {}
    m = NOTE_FINDING_RE.search(all_text)
    if m:
        val = m.group(1).upper().replace(" ", "_").replace("-", "_")
        if "REVIEW" in val:
            val = "NEEDS_REVIEW"
        result["finding"] = val
    m = re.search(r"reason[^a-z]{0,4}[:;]?\s*(.+)", all_text, re.IGNORECASE)
    if m:
        result["reason"] = norm(m.group(1))[:160]
    return result


# fixed field order of the scanned short-form layouts; the line position
# pins the field, which lets badly smeared values be decoded against the
# field's own vocabulary with relaxed thresholds
DOC_TEMPLATES = {
    "sponsor_letter": ["sponsor_id", "applicant_name", "declared_purpose",
                       "visa_class"],
    "intake_form": ["case_id", "applicant_name", "species_code", "home_world",
                    "visa_class", "sponsor_id", "arrival_date",
                    "declared_purpose"],
    "biometric": ["case_id", "applicant_name", "species_match",
                  "biometric_confidence", "observed_flags"],
    "registry": ["applicant_name", "home_world", "species_code",
                 "arrival_date"],
    "fee_receipt": ["case_id", "fee_status"],
}


def _positional_value(field, text):
    """Best vocab interpretation of a template line's trailing tokens."""
    toks = text.split()
    best = (None, 0.0)
    for k in range(1, min(4, len(toks)) + 1):
        cand = " ".join(toks[-k:])
        v, r = None, 0.0
        if field == "declared_purpose":
            v, r = fuzzy_best(cand, PURPOSES, 0.5)
        elif field == "applicant_name":
            v, r = fuzzy_name(cand)
            if r < 0.5:
                v = None
        elif field in ("species_code", "species_match"):
            v, r = fuzzy_best(cand.replace(" ", "_"), SPECIES, 0.5)
        elif field == "home_world":
            v, r = fuzzy_best(cand, HOME_WORLDS, 0.5)
        elif field == "sponsor_id":
            m = SPN_RE.search(cand.upper())
            v = normalize_spn(m) if m else None
            r = 0.85 if v else 0.0
        elif field == "arrival_date":
            ds = date_candidates(cand)
            if ds:
                v, r = ds[0]
        elif field == "fee_status":
            v, r = fuzzy_best(cand, FEE_VALUES, 0.55)
        if v and r > best[1]:
            best = (v, r)
    return best


def positional_parse(lines, min_title_sim=0.45):
    """Recover fields from smeared scanned pages via template line order."""
    def clean(s):
        return " ".join(re.sub(r"[^a-z ]", " ", s.lower()).split())

    found = []
    for i, line in enumerate(lines):
        cl = clean(line)
        if len(cl) < 8:
            continue
        for dt, title in DOC_TITLE_KEYS.items():
            if dt not in DOC_TEMPLATES:
                continue
            if sim(cl, title) < min_title_sim:
                continue
            template = DOC_TEMPLATES[dt]
            for j, field in enumerate(template):
                li = i + 1 + j
                if li >= len(lines):
                    break
                v, r = _positional_value(field, lines[li])
                if v:
                    found.append((field, v, r * 0.75, dt))
    return found


def enrich_page(page_info):
    """Merge value-driven scan results into a page's parsed fields.

    Idempotent; runs on the stored `lines` so it can also be applied to
    previously extracted evidence.
    """
    lines = page_info.get("lines") or []
    fields = page_info.setdefault("fields", {})
    extras = page_info.setdefault("extras", {})
    if page_info.get("scanned") and "finding" not in fields:
        looks_like_note = page_info.get("doc_type") == "adjudicator_note" or any(
            sim(re.sub(r"[^a-z ]", "", l.lower()), "manual adjudicator note") > 0.6
            for l in lines[:8])
        if looks_like_note:
            finding, score = parse_scanned_note(lines)
            if finding:
                fields["finding"] = (finding, score)
    if page_info.get("scanned"):
        for field, value, score, dt in positional_parse(lines):
            prev = fields.get(field)
            if not prev or score > prev[1]:
                fields[field] = (value, score)
            if not page_info.get("doc_type"):
                page_info["doc_type"] = dt
    for field, value, score in scan_line_values(lines):
        if field in ("fee_hint_zero", "fee_hint_amount", "flag_token",
                     "registry_status", "dip_waiver"):
            if field == "flag_token":
                extras.setdefault("flag_tokens", [])
                if value not in extras["flag_tokens"]:
                    extras["flag_tokens"].append(value)
            else:
                prev = extras.get(field)
                if not prev or score > prev[1]:
                    extras[field] = (value, score)
            continue
        prev = fields.get(field)
        if not prev or score > prev[1]:
            fields[field] = (value, score)
    return page_info


# ---------------------------------------------------------------------------
# Per-case extraction
# ---------------------------------------------------------------------------


def ocr_quality(text):
    """Crude quality score: fraction of tokens that look like words."""
    tokens = re.findall(r"[A-Za-z]{3,}", text)
    if not tokens:
        return 0.0
    import string
    ok = sum(1 for t in tokens if sum(c in string.ascii_letters for c in t) == len(t))
    return ok / max(len(tokens), 1)


def extract_case(pdf_path):
    """Extract structured evidence from one PDF. Returns evidence dict."""
    doc = fitz.open(pdf_path)
    case_id_from_name = Path(pdf_path).stem
    pages = []
    injection = False
    hidden_all = []
    for pno, page in enumerate(doc):
        visible, stamps, hidden = extract_text_layer(page)
        vis_lines = [t for t, _ in visible if not FOOTER_RE.match(t)]
        hidden_all.extend(hidden)
        if hidden:
            injection = True
        page_info = {
            "page": pno,
            "stamps": stamps,
            "scanned": False,
            "ocr_quality": None,
        }
        if is_scanned_page(visible):
            page_info["scanned"] = True
            variants = ocr_page_variants(page)
            best_fields = {}
            all_lines = []
            doc_type_votes = []
            qualities = []
            for out in variants:
                lines = [norm(l) for l in out.splitlines() if norm(l)]
                lines = [l for l in lines if not FOOTER_RE.match(l)]
                all_lines.extend(lines)
                qualities.append(ocr_quality(out))
                dt = classify_doc(lines)
                if dt:
                    doc_type_votes.append(dt)
                fields = clean_field_values(parse_label_value_lines(lines))
                for f, (v, s) in fields.items():
                    if f not in best_fields or s > best_fields[f][1]:
                        best_fields[f] = (v, s)
            joined = "\n".join(all_lines)
            note = parse_note_text(joined)
            for f, v in note.items():
                if f not in best_fields:
                    best_fields[f] = (v, 0.8)
            for f, v in parse_corrections(joined).items():
                best_fields[f] = (v, 3.0)
            page_info["doc_type"] = Counter(doc_type_votes).most_common(1)[0][0] if doc_type_votes else None
            page_info["fields"] = {f: v for f, v in best_fields.items()}
            page_info["ocr_quality"] = round(max(qualities) if qualities else 0.0, 3)
            page_info["lines"] = all_lines[:60]
            # stamp words recovered by OCR (DENIED / RESCINDED etc.)
            for w in STAMP_WORDS:
                if re.search(r"\b" + re.escape(w) + r"\b", joined, re.IGNORECASE):
                    page_info["stamps"].append(w)
        else:
            page_info["doc_type"] = classify_doc(vis_lines)
            pairs = parse_label_value_lines(vis_lines)
            fields = clean_field_values(pairs)
            note = parse_note_text(" ".join(vis_lines))
            for f, v in note.items():
                fields.setdefault(f, (v, 1.0))
            for f, v in parse_corrections("\n".join(vis_lines)).items():
                fields[f] = (v, 3.0)
            page_info["fields"] = fields
            page_info["lines"] = vis_lines[:60]
        enrich_page(page_info)
        pages.append(page_info)
    doc.close()
    return {
        "case_id": case_id_from_name,
        "pages": pages,
        "injection": injection,
        "hidden": hidden_all[:8],
    }


# ---------------------------------------------------------------------------
# Evidence resolution + adjudication
# ---------------------------------------------------------------------------

# precedence per manual: adjudicator note > intake form > biometric >
# sponsor letter > registry > fee receipt (fee receipt authoritative for fee)
DOC_PRECEDENCE = {
    "adjudicator_note": 6,
    "intake_form": 5,
    "biometric": 4,
    "sponsor_letter": 3,
    "registry": 2,
    "fee_receipt": 1,
    None: 0,
}

EMBARGO_WORLDS = {"Eris Relay", "TRAPPIST-1e"}

SPONSOR_NOTICE_RE = re.compile(r"sponsor standing requires", re.IGNORECASE)


VOTED_FIELDS = {"applicant_name", "species_code", "home_world", "visa_class",
                "sponsor_id", "arrival_date", "declared_purpose", "fee_status"}


def resolve_fields(evidence):
    """Merge page-level fields into a case-level record.

    Fields that appear on several documents are resolved by weighted voting:
    agreement across pages beats a single higher-precedence read, which
    corrects most single-page OCR errors.
    """
    merged = {}
    votes = {}
    for page in sorted(evidence["pages"],
                       key=lambda p: DOC_PRECEDENCE.get(p.get("doc_type"), 0)):
        weight = DOC_PRECEDENCE.get(page.get("doc_type"), 0)
        for f, val in page.get("fields", {}).items():
            v, s = val[0], val[1]
            if v in (None, "", "__BLANK__"):
                continue
            if f == "species_match":
                # the biometric slip's species reading corroborates species_code
                votes.setdefault("species_code", []).append(
                    (v, s * 0.9 * (1 + 0.15 * weight), page.get("doc_type")))
            score = s * (1 + 0.15 * weight)
            if f in VOTED_FIELDS and s < 2.5:
                votes.setdefault(f, []).append((v, score, page.get("doc_type")))
            if f not in merged or score >= merged[f][1]:
                merged[f] = (v, score, page.get("doc_type"))

    for f, cands in votes.items():
        if merged.get(f, (None, 0))[1] >= 2.5:
            continue  # manual correction wins outright
        by_value = {}
        for v, score, dt in cands:
            cur = by_value.setdefault(v, [0.0, 0, dt])
            cur[0] = max(cur[0], score)
            cur[1] += 1
        best_v, best = None, -1.0
        for v, (mx, n, dt) in by_value.items():
            combined = mx + 0.3 * (n - 1)
            if combined > best:
                best, best_v = combined, v
        if best_v is not None:
            mx, n, dt = by_value[best_v]
            merged[f] = (best_v, best, dt)

    # forged-sponsor exception: when the intake form carries a publicly
    # revoked sponsor id but the attestation letter names a different one,
    # the letter reflects the real sponsor (the form field was tampered)
    spn_by_doc = {}
    for page in evidence["pages"]:
        v = page.get("fields", {}).get("sponsor_id")
        if v and v[0] not in (None, "", "__BLANK__") and v[1] < 2.5:
            spn_by_doc.setdefault(page.get("doc_type"), v[0])
    form_spn = spn_by_doc.get("intake_form")
    letter_spn = spn_by_doc.get("sponsor_letter")
    if (form_spn in REVOKED_SPONSORS and letter_spn
            and letter_spn != form_spn
            and merged.get("sponsor_id", (None, 0, None))[1] < 2.5):
        merged["sponsor_id"] = (letter_spn, merged.get("sponsor_id", (0, 0))[1],
                                "sponsor_letter")
        merged["_forged_sponsor"] = (form_spn, 1.0, "intake_form")
    return merged


def case_features(evidence):
    """Cross-page features used by the adjudication rules."""
    feats = {
        "stamps": set(),
        "flag_tokens": set(),
        "sponsor_notice": False,
        "registry_embargo": False,
        "fee_hint": None,
        "names_by_doc": {},
        "biometric_unreadable": False,
        "has_biometric_page": False,
        "damaged_pages": 0,
        "scanned_pages": 0,
        "rescinded_text": False,
    }
    for page in evidence["pages"]:
        dt = page.get("doc_type")
        lines = page.get("lines") or []
        joined = " ".join(lines)
        for s in page.get("stamps", []):
            feats["stamps"].add(s.upper())
        extras = page.get("extras", {})
        for tok in extras.get("flag_tokens", []):
            feats["flag_tokens"].add(tok)
        if extras.get("registry_status") and dt == "registry":
            feats["registry_embargo"] = True
        if extras.get("fee_hint_amount") and not feats["fee_hint"]:
            feats["fee_hint"] = "paid"
        if extras.get("fee_hint_zero"):
            feats["fee_hint"] = "waived"
        if SPONSOR_NOTICE_RE.search(joined):
            feats["sponsor_notice"] = True
        if re.search(r"rescind|resc[il]nd", joined, re.IGNORECASE):
            feats["rescinded_text"] = True
        name = page.get("fields", {}).get("applicant_name")
        if name and name[0] and name[1] >= 0.99:
            feats["names_by_doc"].setdefault(dt, name[0])
        if dt == "biometric":
            feats["has_biometric_page"] = True
            if page.get("scanned") and not page.get("fields", {}).get("observed_flags"):
                if len(page.get("fields", {})) <= 1:
                    feats["biometric_unreadable"] = True
        if page.get("scanned"):
            feats["scanned_pages"] += 1
            if len(page.get("fields", {})) == 0:
                feats["damaged_pages"] += 1
    return feats


def compute_flags(evidence, merged, feats):
    flags = set()
    if "observed_flags" in merged and merged["observed_flags"][0] != "none":
        flags.update(merged["observed_flags"][0].split("|"))
    flags.update(feats["flag_tokens"])

    world = merged.get("home_world", (None,))[0]
    if world in EMBARGO_WORLDS or feats["registry_embargo"]:
        flags.add("planetary_embargo")

    # cross-document identity conflict: only when both names are clean
    # text-layer reads (OCR-derived names are too noisy to compare)
    names = feats["names_by_doc"]
    form_name = names.get("intake_form")
    reg_name = names.get("registry")
    if form_name and reg_name and form_name != reg_name:
        flags.add("identity_conflict")

    if "DENIED" in feats["stamps"] and (
            feats["rescinded_text"] or "REVIEW" in feats["stamps"]
            or "APPROVED" in feats["stamps"]):
        flags.add("rescinded_denial")
    return flags


DEFAULT_STALE_CUTOFF = "2026-01-08"


def batch_stale_cutoff(dates):
    """Stale = arrival more than ~180 days before packet receipt. The receipt
    date is inferred from the newest arrival date of the batch's modal year
    (single OCR-corrupted years must not skew it), so the rule tracks
    whenever the evaluation batch was generated."""
    valid = sorted(d for d in dates if d and "2020" < d < "2035")
    if len(valid) < 50:
        return DEFAULT_STALE_CUTOFF
    years = Counter(d[:4] for d in valid)
    modal_year = years.most_common(1)[0][0]
    in_year = [d for d in valid if d[:4] == modal_year]
    import datetime
    newest = datetime.date.fromisoformat(in_year[max(0, int(len(in_year) * 0.99) - 1)])
    return (newest - datetime.timedelta(days=183)).isoformat()


def has_dip_waiver(evidence):
    """Visible DIP-WAIVER / hardship waiver code anywhere in the packet.

    On training receipts a visible waiver code implies fee 'waived' even when
    the printed fee status disagrees (105/105)."""
    for page in evidence["pages"]:
        if page.get("extras", {}).get("dip_waiver"):
            return True
        wc = page.get("fields", {}).get("waiver_code")
        if wc and "waiver" in str(wc[0]).lower():
            return True
    return False


def revoked_sponsor_evidence(evidence, merged):
    """Return a revoked sponsor id only when we trust the read."""
    hits = []
    for page in evidence["pages"]:
        v = page.get("fields", {}).get("sponsor_id")
        if v and v[0] in REVOKED_SPONSORS:
            hits.append((v[0], v[1], page.get("scanned")))
    if merged.get("_forged_sponsor"):
        return merged["_forged_sponsor"][0]
    for spn, score, scanned in hits:
        if not scanned:
            return spn
    counts = Counter(spn for spn, _, _ in hits)
    for spn, n in counts.items():
        if n >= 2:
            return spn
    return None


def sponsor_blank_everywhere(evidence):
    """Explicit '[SPONSOR ID BLANK]' with no usable sponsor id anywhere."""
    blank_seen = False
    for page in evidence["pages"]:
        v = page.get("fields", {}).get("sponsor_id")
        if v:
            if v[0] == "__BLANK__":
                blank_seen = True
            else:
                return False
        if any("[SPONSOR" in l.upper() for l in page.get("lines") or []):
            blank_seen = True
    return blank_seen


def adjudicate(evidence, merged, stale_cutoff=DEFAULT_STALE_CUTOFF):
    """Apply policy rules. Returns (adjudication, flags, reason_path)."""
    feats = case_features(evidence)
    flags = compute_flags(evidence, merged, feats)

    finding = merged.get("finding", (None,))[0]
    if finding in ("APPROVED", "DENIED", "NEEDS_REVIEW"):
        return finding, flags, "note_finding"

    stamps = feats["stamps"]
    if "DENIED" in stamps and "rescinded_denial" in flags:
        return "NEEDS_REVIEW", flags, "rescinded_denial"
    if "DENIED" in stamps:
        return "DENIED", flags, "denied_stamp"
    if "REVIEW" in stamps:
        return "NEEDS_REVIEW", flags, "review_stamp"
    if "APPROVED" in stamps:
        return "APPROVED", flags, "approved_stamp"

    if flags & DISQUALIFYING_FLAGS:
        return "DENIED", flags, "disqualifying_flag"

    fee_explicit = merged.get("fee_status", (None,))[0]
    visa = merged.get("visa_class", (None,))[0]
    arrival = merged.get("arrival_date", (None,))[0]

    revoked = revoked_sponsor_evidence(evidence, merged)
    if revoked:
        return "DENIED", flags, "revoked_sponsor"
    if feats["sponsor_notice"]:
        return "DENIED", flags, "sponsor_notice"
    if visa == "TRANSIT-7":
        return "DENIED", flags, "transit_visa"
    if fee_explicit == "unpaid" and not has_dip_waiver(evidence):
        return "DENIED", flags, "unpaid_fee"
    arrival_score = merged.get("arrival_date", (None, 0))[1]
    if (arrival and arrival < stale_cutoff and visa != "DIP-1"
            and arrival_score >= 0.7):
        return "DENIED", flags, "stale_arrival"
    if fee_explicit == "unknown":
        return "NEEDS_REVIEW", flags, "fee_unknown_explicit"
    if sponsor_blank_everywhere(evidence) and visa != "DIP-1":
        return "DENIED", flags, "sponsor_blank"
    if flags:
        return "NEEDS_REVIEW", flags, "review_flags"

    # damage routing: otherwise-clean packets with unreadable/unclassifiable
    # pages have too high a hidden-flag rate for APPROVED to pay off
    unreadable = sum(1 for p in evidence["pages"]
                     if p.get("scanned") and not p.get("fields"))
    unknown_page = any(p.get("doc_type") is None and p.get("scanned")
                       for p in evidence["pages"])
    if unreadable >= 2 or (unreadable >= 1 and unknown_page):
        return "NEEDS_REVIEW", flags, "damaged_packet"

    return "APPROVED", flags, "clean"


def confidence_bucket(evidence, merged, path):
    """Refine the decision path into a calibration bucket."""
    if path == "note_finding":
        for page in evidence["pages"]:
            v = page.get("fields", {}).get("finding")
            if v and not page.get("scanned"):
                return "note_finding_text"
        return "note_finding_scan"
    if path == "clean":
        key_fields = ["applicant_name", "species_code", "home_world",
                      "visa_class", "sponsor_id", "arrival_date"]
        strong = sum(1 for f in key_fields
                     if merged.get(f, (None, 0))[1] >= 0.95)
        return "clean_strong" if strong >= 5 else "clean_weak"
    return path


# confidence = smoothed per-bucket accuracy measured on the training dev split
CONFIDENCE_BY_PATH = {
    "note_finding_text": 0.99,
    "note_finding_scan": 0.97,
    "clean_strong": 0.61,
    "clean_weak": 0.50,
    "note_finding": 0.98,
    "clean": 0.59,
    "damaged_packet": 0.31,
    "disqualifying_flag": 0.97,
    "review_flags": 0.77,
    "transit_visa": 0.85,
    "revoked_sponsor": 0.74,
    "unpaid_fee": 0.88,
    "fee_unknown_explicit": 0.94,
    "stale_arrival": 0.88,
    "approved_stamp": 0.8,
    "sponsor_blank": 0.5,
    "denied_stamp": 0.8,
    "rescinded_denial": 0.7,
    "sponsor_notice": 0.8,
    "review_stamp": 0.9,
}


def build_record(evidence, stale_cutoff=DEFAULT_STALE_CUTOFF):
    """Resolve fields + adjudicate one case into a submission record."""
    merged = resolve_fields(evidence)
    adjudication, flags, path = adjudicate(evidence, merged, stale_cutoff)

    def get(f, default=""):
        return merged.get(f, (default,))[0] or default

    fee = merged.get("fee_status", (None,))[0]
    if has_dip_waiver(evidence):
        fee = "waived"
    if not fee:
        hint = None
        for page in evidence["pages"]:
            extras = page.get("extras", {})
            if extras.get("fee_hint_zero"):
                hint = "waived"
            elif extras.get("fee_hint_amount") and not hint:
                hint = "paid"
        # 'paid' is the dominant value when a receipt is unreadable; this
        # imputation affects only the extraction field, never the decision
        fee = hint or "paid"

    bucket = confidence_bucket(evidence, merged, path)
    record = {
        "case_id": evidence["case_id"],
        "applicant_name": get("applicant_name", "unknown"),
        "species_code": get("species_code", "unknown"),
        "home_world": get("home_world", "unknown"),
        "visa_class": get("visa_class", "unknown"),
        "sponsor_id": get("sponsor_id", "SPN-0000"),
        "arrival_date": get("arrival_date", "2026-01-01"),
        "declared_purpose": get("declared_purpose", "unknown"),
        "risk_flags": "|".join(sorted(flags)) if flags else "none",
        "fee_status": fee,
        "adjudication": adjudication,
        "confidence": CONFIDENCE_BY_PATH.get(bucket, CONFIDENCE_BY_PATH.get(path, 0.5)),
    }
    return record, bucket


def _worker(pdf_path):
    try:
        return extract_case(str(pdf_path))
    except Exception:
        return {"case_id": Path(pdf_path).stem, "pages": [],
                "injection": False, "hidden": [], "failed": True}


# Buckets where the rule engine's measured dev-split accuracy is too low to
# trust; these cases are re-decided by a small ExtraTrees model over the rule
# engine's own resolved fields and damage features (validated on a train
# holdout against both the rules and a heavier hybrid OCR+ML engine).
DELEGATED_BUCKETS = {"clean_strong", "clean_weak", "damaged_packet",
                     "sponsor_blank"}

GRAYBOX_FEATURE_FIELDS = {
    "species_code": SPECIES,
    "home_world": HOME_WORLDS,
    "visa_class": VISA_CLASSES,
    "declared_purpose": PURPOSES,
    "fee_status": FEE_VALUES,
}


def graybox_vector(evidence):
    """Feature vector for the gray-zone model, from resolved evidence."""
    merged = resolve_fields(evidence)
    feats = case_features(evidence)
    flags = compute_flags(evidence, merged, feats)
    vec = []
    for f, vocab in GRAYBOX_FEATURE_FIELDS.items():
        v = merged.get(f, (None,))[0]
        vec.extend(1.0 if v == c else 0.0 for c in vocab)
        vec.append(1.0 if v is None else 0.0)
        vec.append(merged.get(f, (None, 0))[1])
    for f in ("applicant_name", "sponsor_id", "arrival_date"):
        vec.append(1.0 if f in merged else 0.0)
        vec.append(merged.get(f, (None, 0))[1])
    vec.append(1.0 if evidence.get("injection") else 0.0)
    vec.append(float(len(evidence["pages"])))
    vec.append(float(feats["scanned_pages"]))
    vec.append(float(feats["damaged_pages"]))
    vec.append(float(len(flags)))
    for fl in RISK_FLAG_VALUES[:-1]:
        vec.append(1.0 if fl in flags else 0.0)
    kinds = {p.get("doc_type") for p in evidence["pages"]}
    for dt in ("intake_form", "fee_receipt", "registry", "biometric",
               "sponsor_letter", "adjudicator_note", None):
        vec.append(1.0 if dt in kinds else 0.0)
    for st in ("DENIED", "APPROVED", "REVIEW", "SAMPLE DENIAL"):
        vec.append(1.0 if st in feats["stamps"] else 0.0)
    bio = None
    for p in evidence["pages"]:
        b = p.get("fields", {}).get("biometric_confidence")
        if b:
            m = re.search(r"(\d+)", str(b[0]))
            if m:
                bio = int(m.group(1))
    vec.append(float(bio) if bio is not None else -1.0)
    return vec


def graybox_decide(probs, classes):
    """Expected-value-optimal decision under the challenge scoring matrix."""
    P = {c: float(p) for c, p in zip(classes, probs)}
    ev = {
        "APPROVED": 8 * P["APPROVED"] + P["NEEDS_REVIEW"] - 4 * P["DENIED"],
        "DENIED": 8 * P["DENIED"] + P["NEEDS_REVIEW"],
        "NEEDS_REVIEW": 8 * P["NEEDS_REVIEW"] + 2 * (P["APPROVED"] + P["DENIED"]),
    }
    adj = max(ev, key=ev.get)
    return adj, round(P[adj], 3)


def apply_graybox_delegation(results, evidences):
    """Re-decide delegated cases with the gray-zone model. Field extraction
    always stays with the rule engine."""
    delegated = [i for i, (_, record, bucket) in enumerate(results)
                 if bucket in DELEGATED_BUCKETS]
    if not delegated:
        return
    try:
        import joblib
        import numpy as np
        bundle = joblib.load(Path(__file__).parent / "models/graybox.joblib")
    except Exception:
        return
    model, classes = bundle["model"], list(bundle["classes"])
    try:
        X = np.array([graybox_vector(evidences[i]) for i in delegated])
        probs = model.predict_proba(X)
    except Exception:
        return
    for i, p in zip(delegated, probs):
        adj, conf = graybox_decide(p, classes)
        results[i][1]["adjudication"] = adj
        results[i][1]["confidence"] = conf


def main(input_dir, output_path):
    pdfs = sorted(str(p) for p in Path(input_dir).glob("*.pdf"))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        evidences = list(pool.map(_worker, pdfs, chunksize=2))

    dates = []
    for evidence in evidences:
        merged = resolve_fields(evidence)
        d, score = merged.get("arrival_date", (None, 0))[:2]
        if d and score >= 0.9:
            dates.append(d)
    cutoff = batch_stale_cutoff(dates)

    results = []
    for pdf, evidence in zip(pdfs, evidences):
        try:
            record, bucket = build_record(evidence, cutoff)
        except Exception:
            record = {
                "case_id": evidence["case_id"],
                "applicant_name": "unknown", "species_code": "unknown",
                "home_world": "unknown", "visa_class": "unknown",
                "sponsor_id": "SPN-0000", "arrival_date": "2026-01-01",
                "declared_purpose": "unknown", "risk_flags": "none",
                "fee_status": "paid", "adjudication": "NEEDS_REVIEW",
                "confidence": 0.4,
            }
            bucket = "error"
        results.append([pdf, record, bucket])

    apply_graybox_delegation(results, evidences)

    with open(out, "w") as f:
        for _, record, _ in results:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: solution.py <input_pdf_dir> <output_path>")
    main(sys.argv[1], sys.argv[2])
