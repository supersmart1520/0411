#!/usr/bin/env python3
"""
pdf_to_images.py — robust, high-quality PDF → image extraction.

Two modes:

  render   (default)  Render EACH PAGE of the PDF to a crisp raster image at a
                      configurable DPI (default 300). This is the clean
                      "고화질" path: it never deals with how images are stored
                      inside the PDF, so it avoids the embedded-image "tiling"
                      problem (big pictures split into many horizontal strips).

  embedded            Pull the raster images that are actually embedded in the
                      PDF and AUTO-REASSEMBLE tiled strips back into whole
                      images. Many PDFs store a single large picture as a run
                      of consecutive equal-width strips stacked on the page;
                      this mode detects those runs and stitches them back
                      together vertically.

Engine selection (automatic, printed at runtime):

  * Prefer PyMuPDF (import name `pymupdf`, legacy `fitz`) when importable.
    - render   : page.get_pixmap() at the requested DPI.
    - embedded : extract embedded images AND their on-page placement boxes,
                 then group strips by true geometry (strips whose boxes are
                 vertically adjacent and share the same horizontal span belong
                 to one image). This needs no per-PDF tuning.

  * Fall back to Poppler command-line tools when PyMuPDF is unavailable.
    - render   : `pdftoppm`
    - embedded : `pdfimages` (geometry is NOT available from these tools, so
                 strips are regrouped by runs of equal dimensions and chunked
                 to the aspect ratio closest to --target-ratio).

Security notes:
  * The input PDF path is treated as untrusted data: it is only ever passed as
    an argument to the PDF libraries / CLIs. We never chdir into an untrusted
    directory and never import anything from it. Run this script with
    `python3 -I pdf_to_images.py ...` to also isolate the interpreter
    (ignore PYTHON* env and the user site dir) when handling untrusted PDFs.

Usage:
    python3 pdf_to_images.py INPUT.pdf [-o OUTDIR] [--mode render|embedded]
            [--dpi 300] [--format png|jpg] [--quality 95] [--prefix page]
            [--target-ratio 1.3333]

Examples:
    # high-quality per-page render (default)
    python3 pdf_to_images.py report.pdf
    # higher resolution
    python3 pdf_to_images.py report.pdf --dpi 400
    # JPEG output
    python3 pdf_to_images.py report.pdf --format jpg --quality 95
    # extract embedded images, auto-stitching tiled strips
    python3 pdf_to_images.py report.pdf --mode embedded
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


# --------------------------------------------------------------------------- #
# Engine / dependency detection
# --------------------------------------------------------------------------- #

def _import_pymupdf():
    """Return the PyMuPDF module (as `pymupdf`), or None if not importable."""
    try:
        import pymupdf  # modern name
        return pymupdf
    except Exception:
        pass
    try:
        import fitz  # legacy name, same library
        return fitz
    except Exception:
        return None


def _import_pil():
    """Return PIL.Image, or None. Needed to stitch strips and to write JPEG."""
    try:
        from PIL import Image
        return Image
    except Exception:
        return None


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _try_pip_install() -> None:
    """Best-effort install of the preferred stack. Silent on failure."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "PyMuPDF", "Pillow"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=600,
        )
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _ext_for(fmt: str) -> str:
    return "jpg" if fmt == "jpg" else "png"


def _save_pil(img, path: str, fmt: str, quality: int) -> None:
    """Save a PIL image as png or jpg with the requested quality."""
    if fmt == "jpg":
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(path, "JPEG", quality=quality, optimize=True)
    else:
        img.save(path, "PNG")


# --------------------------------------------------------------------------- #
# RENDER mode
# --------------------------------------------------------------------------- #

def render_pymupdf(pymupdf, Image, pdf_path, outdir, dpi, fmt, quality, prefix):
    """Render each page to a raster image using PyMuPDF."""
    doc = pymupdf.open(pdf_path)
    try:
        n = doc.page_count
        width = max(3, len(str(n)))
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        results = []
        for pno in range(n):
            page = doc[pno]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            name = f"{prefix}_{pno + 1:0{width}d}.{_ext_for(fmt)}"
            path = os.path.join(outdir, name)
            if Image is not None:
                # Route through PIL so PNG/JPEG and --quality behave identically.
                mode = "RGB" if pix.n < 4 else "RGBA"
                img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                _save_pil(img, path, fmt, quality)
            else:
                # PIL unavailable: PyMuPDF can still write PNG directly.
                if fmt == "jpg":
                    pix.pil_save(path, format="JPEG", quality=quality)
                else:
                    pix.save(path)
            results.append((name, os.path.getsize(path)))
        return results
    finally:
        doc.close()


def render_poppler(pdf_path, outdir, dpi, fmt, quality, prefix):
    """Render each page using Poppler's pdftoppm, then normalize file names."""
    stem = os.path.join(outdir, "__render")
    cmd = ["pdftoppm", "-r", str(dpi)]
    if fmt == "jpg":
        cmd += ["-jpeg", "-jpegopt", f"quality={quality}"]
    else:
        cmd += ["-png"]
    cmd += [pdf_path, stem]
    subprocess.run(cmd, check=True)

    produced = sorted(
        f for f in os.listdir(outdir)
        if f.startswith("__render") and (f.endswith(".png") or f.endswith(".jpg"))
    )
    ext = _ext_for(fmt)
    width = max(3, len(str(len(produced))))
    results = []
    for i, fname in enumerate(produced, start=1):
        src = os.path.join(outdir, fname)
        name = f"{prefix}_{i:0{width}d}.{ext}"
        dst = os.path.join(outdir, name)
        os.replace(src, dst)
        results.append((name, os.path.getsize(dst)))
    return results


# --------------------------------------------------------------------------- #
# EMBEDDED mode — reassembly of tiled strips
# --------------------------------------------------------------------------- #

def _group_by_geometry(placements, x_tol=3.0, gap_tol=6.0):
    """
    Group strip placements that form one logical image.

    `placements` is a page-ordered list of dicts with keys:
        img  : PIL image of the strip
        x0,y0,x1,y1 : on-page bounding box (points)
        page : page index

    A strip continues the current group when, on the SAME page, it shares the
    current group's horizontal span (left/right within x_tol) and its top edge
    meets the previous strip's bottom edge (gap within gap_tol). Otherwise it
    starts a new group. A lone strip is simply a group of one.
    """
    groups = []
    cur = []
    for p in placements:
        if cur:
            prev = cur[-1]
            same_col = (abs(p["x0"] - prev["x0"]) <= x_tol
                        and abs(p["x1"] - prev["x1"]) <= x_tol)
            adjacent = (p["page"] == prev["page"]
                        and -x_tol <= (p["y0"] - prev["y1"]) <= gap_tol)
            if same_col and adjacent:
                cur.append(p)
                continue
            groups.append(cur)
            cur = [p]
        else:
            cur = [p]
    if cur:
        groups.append(cur)
    return groups


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _group_by_dimension_runs(strips, target_ratio):
    """
    Geometry-free regrouping (Poppler fallback).

    `strips` is an ordered list of dicts with keys: img, w, h.
    Detect maximal runs of consecutive equal-(w,h) strips. Within each run of
    length N, pick a chunk size c (a divisor of N) so the stitched image's
    aspect ratio w / (c*h) is as close as possible to `target_ratio`; split the
    run into consecutive chunks of c strips. This reconstructs tiled images
    (e.g. 5 strips of 2640x396 -> one 2640x1980 image) without page geometry.
    """
    # 1) split into runs of equal dimensions
    runs = []
    cur = []
    for s in strips:
        if cur and (s["w"], s["h"]) == (cur[0]["w"], cur[0]["h"]):
            cur.append(s)
        else:
            if cur:
                runs.append(cur)
            cur = [s]
    if cur:
        runs.append(cur)

    # 2) chunk each run to the best aspect ratio
    groups = []
    for run in runs:
        w, h = run[0]["w"], run[0]["h"]
        n = len(run)
        best_c = min(
            _divisors(n),
            key=lambda c: abs((w / float(c * h)) - target_ratio),
        )
        for k in range(0, n, best_c):
            groups.append(run[k:k + best_c])
    return groups


def _stack(Image, group):
    """Stack a group of strips vertically into one RGB image."""
    imgs = [g["img"] for g in group]
    width = max(im.width for im in imgs)
    total_h = sum(im.height for im in imgs)
    canvas = Image.new("RGB", (width, total_h), (255, 255, 255))
    y = 0
    for im in imgs:
        canvas.paste(im.convert("RGB"), (0, y))
        y += im.height
    return canvas


def embedded_pymupdf(pymupdf, Image, pdf_path, outdir, fmt, quality, prefix,
                     target_ratio):
    """Extract embedded images with PyMuPDF and reassemble by true geometry."""
    import io
    doc = pymupdf.open(pdf_path)
    placements = []
    try:
        for pno in range(doc.page_count):
            page = doc[pno]
            # placement boxes, in page content order, with the source xref
            for info in page.get_image_info(xrefs=True):
                xref = info.get("xref", 0)
                if not xref:
                    continue
                raw = doc.extract_image(xref)
                img = Image.open(io.BytesIO(raw["image"]))
                img.load()
                b = info["bbox"]
                placements.append({
                    "img": img, "page": pno,
                    "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
                })
    finally:
        doc.close()

    if placements:
        groups = _group_by_geometry(placements)
    else:
        groups = []

    return _write_groups(Image, groups, outdir, fmt, quality, prefix)


def embedded_poppler(Image, pdf_path, outdir, fmt, quality, prefix, target_ratio):
    """Extract embedded images with Poppler's pdfimages; regroup by dimension."""
    if Image is None:
        raise RuntimeError(
            "embedded mode needs Pillow to read/stitch strips, but it is not "
            "installed and could not be installed."
        )
    import tempfile
    tmp = tempfile.mkdtemp(prefix="pdfimages_", dir=outdir)
    try:
        subprocess.run(["pdfimages", "-all", pdf_path, os.path.join(tmp, "img")],
                       check=True)
        files = sorted(
            os.path.join(tmp, f) for f in os.listdir(tmp)
            if os.path.isfile(os.path.join(tmp, f))
        )
        strips = []
        for f in files:
            try:
                im = Image.open(f)
                im.load()
            except Exception:
                continue  # skip non-image sidecar files
            strips.append({"img": im, "w": im.width, "h": im.height})
        groups = _group_by_dimension_runs(strips, target_ratio) if strips else []
        return _write_groups(Image, groups, outdir, fmt, quality, prefix)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_groups(Image, groups, outdir, fmt, quality, prefix):
    """Stitch each group and write it out as image_NNN.<ext>."""
    ext = _ext_for(fmt)
    width = max(3, len(str(len(groups))))
    results = []
    for i, group in enumerate(groups, start=1):
        img = group[0]["img"] if len(group) == 1 else _stack(Image, group)
        name = f"image_{i:0{width}d}.{ext}"
        path = os.path.join(outdir, name)
        _save_pil(img, path, fmt, quality)
        results.append((name, os.path.getsize(path)))
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf_to_images.py",
        description="Extract images from a PDF (high-quality page render, or "
                    "embedded-image extraction with automatic tiled-strip "
                    "reassembly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("pdf", metavar="INPUT.pdf", help="path to the input PDF")
    p.add_argument("-o", "--outdir", default=None,
                   help="output directory (default: <pdfname>_images next to "
                        "the input)")
    p.add_argument("--mode", choices=("render", "embedded"), default="render",
                   help="render each page, or extract embedded images")
    p.add_argument("--dpi", type=int, default=300,
                   help="render resolution in DPI (render mode only)")
    p.add_argument("--format", choices=("png", "jpg"), default="png",
                   dest="format", help="output image format")
    p.add_argument("--quality", type=int, default=95,
                   help="JPEG quality 1-100 (jpg format only)")
    p.add_argument("--prefix", default="page",
                   help="file name prefix for render mode (page_001.ext ...)")
    p.add_argument("--target-ratio", type=float, default=4.0 / 3.0,
                   help="Poppler-fallback only: target width/height aspect "
                        "ratio used to re-chunk tiled strips")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    pdf_path = os.path.abspath(args.pdf)
    if not os.path.isfile(pdf_path):
        print(f"ERROR: input PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    # Output directory (never derived from untrusted PDF *contents*; only its
    # name on disk). Created if missing; we never chdir into it.
    if args.outdir:
        outdir = os.path.abspath(args.outdir)
    else:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        outdir = os.path.join(os.path.dirname(pdf_path), f"{stem}_images")
    os.makedirs(outdir, exist_ok=True)

    # Resolve engines. Try to ensure the preferred stack is present.
    pymupdf = _import_pymupdf()
    Image = _import_pil()
    if pymupdf is None or Image is None:
        _try_pip_install()
        pymupdf = _import_pymupdf()
        Image = _import_pil()

    # Decide engine + run.
    results = None
    engine = None
    try:
        if args.mode == "render":
            if pymupdf is not None:
                engine = "PyMuPDF"
                results = render_pymupdf(pymupdf, Image, pdf_path, outdir,
                                         args.dpi, args.format, args.quality,
                                         args.prefix)
            elif _have("pdftoppm"):
                engine = "Poppler (pdftoppm)"
                results = render_poppler(pdf_path, outdir, args.dpi,
                                         args.format, args.quality, args.prefix)
            else:
                print("ERROR: no render engine available (install PyMuPDF or "
                      "poppler-utils).", file=sys.stderr)
                return 3
        else:  # embedded
            if pymupdf is not None and Image is not None:
                engine = "PyMuPDF (+Pillow, geometry reassembly)"
                results = embedded_pymupdf(pymupdf, Image, pdf_path, outdir,
                                           args.format, args.quality,
                                           args.prefix, args.target_ratio)
            elif _have("pdfimages"):
                engine = "Poppler (pdfimages, dimension-run reassembly)"
                results = embedded_poppler(Image, pdf_path, outdir, args.format,
                                           args.quality, args.prefix,
                                           args.target_ratio)
            else:
                print("ERROR: no embedded-image engine available (install "
                      "PyMuPDF+Pillow or poppler-utils).", file=sys.stderr)
                return 3
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: engine command failed: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:  # pragma: no cover - surfaced to the user
        print(f"ERROR: extraction failed: {exc}", file=sys.stderr)
        return 4

    # Summary.
    label = "pages rendered" if args.mode == "render" else "images reassembled"
    print("=" * 60)
    print("PDF -> images: done")
    print(f"  input   : {pdf_path}")
    print(f"  engine  : {engine}")
    print(f"  mode    : {args.mode}" + (f"  (dpi={args.dpi})"
                                        if args.mode == "render" else ""))
    print(f"  format  : {args.format}" + (f" q{args.quality}"
                                          if args.format == "jpg" else ""))
    print(f"  outdir  : {outdir}")
    print(f"  {label}: {len(results)}")
    print("-" * 60)
    total = 0
    for name, size in results:
        total += size
        print(f"    {name:<16} {_human_size(size):>12}")
    print("-" * 60)
    print(f"  total size: {_human_size(total)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
