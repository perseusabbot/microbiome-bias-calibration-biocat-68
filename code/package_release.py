"""Regenerate the SHA-256 manifest and deterministic deliverable ZIP.

Run from any directory. The archive is placed beside the package directory.
Only files beneath the package are included; git, caches, and virtual
environments are excluded. No local absolute paths or owner IDs are stored.
"""
from pathlib import Path
import hashlib,zipfile

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','__pycache__','.venv','.mpl-cache','.DS_Store'}

def package():
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file()
           and not set(p.relative_to(ROOT).parts)&SKIP and p.suffix!='.pyc'
           and p.name!='MANIFEST.sha256']
    manifest=ROOT/'MANIFEST.sha256'
    manifest.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in files))
    archive=ROOT.parent/'microbiome-calibration-reproducible.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(files+[manifest]):
            info=zipfile.ZipInfo('microbiome-calibration/'+p.relative_to(ROOT).as_posix(),(2026,10,17,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,p.read_bytes(),compresslevel=9)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in files+[manifest]:
            assert z.read('microbiome-calibration/'+p.relative_to(ROOT).as_posix())==p.read_bytes()
    print(f'Verified {len(files)+1} archived files; {archive.stat().st_size:,} bytes')
    print('Archive SHA-256:',hashlib.sha256(archive.read_bytes()).hexdigest())

if __name__=='__main__':package()
