"""Large drawings are kept in git as .dxf.gz (GitHub's web upload stops at 25 MB per file); unpack on first use."""
import gzip
import os
import shutil


def unpack_sources(folder):
    """Write <name>.dxf next to every <name>.dxf.gz that has no unpacked copy yet."""
    if not os.path.isdir(folder):
        return
    for f in os.listdir(folder):
        if f.lower().endswith(".dxf.gz"):
            out = os.path.join(folder, f[:-3])
            if not os.path.exists(out):
                with gzip.open(os.path.join(folder, f), "rb") as a, open(out + ".part", "wb") as b:
                    shutil.copyfileobj(a, b)
                os.replace(out + ".part", out)
