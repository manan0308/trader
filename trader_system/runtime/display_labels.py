from __future__ import annotations


ASSET_LABELS = {
    "NIFTY": "NIFTY",
    "MIDCAP": "MIDCAP",
    "SMALLCAP": "SMALLCAP",
    "GOLD": "GOLD",
    "SILVER": "SILVER",
    "US": "MON100",
    "CASH": "LIQUIDBEES",
}


def asset_label(asset: str) -> str:
    return ASSET_LABELS.get(asset, asset)
