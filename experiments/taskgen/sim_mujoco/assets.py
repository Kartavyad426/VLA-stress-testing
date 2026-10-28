"""TaskGen MuJoCo layer: the asset packs (MJCF) as a deterministic lookup (docs/TASKGEN_DESIGN.html §3a).
Name match by experiments/asset_pack/catalogue.match (res31/AssetPack's code, read-only); sizes at real scale from
build_scene.asset_footprint. Packs: kitchen + gso. YCB is left out: its catalogue writes MJCF files into the shared
data folder on first use. Exploratory, UNVERIFIED.
"""
from functools import lru_cache

from core.kinds import PACK_KIND

PACKS = "kitchen,gso"


@lru_cache(maxsize=1)
def _cat():
    import catalogue as packs
    return packs.pack_catalogue(PACKS)


@lru_cache(maxsize=None)
def info(key):
    from build_scene import asset_footprint
    e = _cat()[key]
    half_xy, zmin, zmax, _ = asset_footprint(e["xml"])
    cat = e["category"] or ""
    kind = PACK_KIND.get(cat) or next((v for t, v in PACK_KIND.items() if t in e["tokens"]), "item")
    return dict(key=key, pack=e["pack"], category=cat or None, kind=kind, licence=e["licence"],
                size_m=(float(2 * half_xy[0]), float(2 * half_xy[1]), float(zmax - zmin)))


def search(query, limit=8):
    """Catalogue keys matching a noun, in catalogue.match's deterministic order."""
    import catalogue as packs
    return packs.match(query, _cat())[:limit]


def exists(key):
    return key in _cat()
