# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

project_root = Path.cwd()
src_path = project_root / "src"


def _merge_unique(items):
    seen = set()
    merged = []
    for item in items:
        key = tuple(item) if isinstance(item, (list, tuple)) else item
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


datas = []
binaries = []
hiddenimports = []

for package_name in ["cvxpy", "scipy", "osqp", "ecos", "scs", "clarabel"]:
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    hiddenimports.extend(pkg_hiddenimports)

datas = _merge_unique(datas)
binaries = _merge_unique(binaries)
hiddenimports = _merge_unique(hiddenimports)


a = Analysis(
    ["run_app.py"],
    pathex=[str(src_path)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ConvexOptAgent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ConvexOptAgent",
)
