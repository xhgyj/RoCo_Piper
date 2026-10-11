from __future__ import annotations
from typing import Mapping

def expanded_lego_library(base: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    library = {str(key): dict(value) for key, value in base.items()}
    pairs = ((12, 13), (2, 14), (3, 15), (4, 16), (5, 17), (6, 18), (9, 19), (10, 20))
    for source_id, target_id in pairs:
        source = library[str(source_id)]
        enlarged = dict(source)
        enlarged['height'] = int(source['height']) * 4
        enlarged['width'] = int(source['width']) * 4
        if 'mass' in source:
            enlarged['mass'] = float(source['mass']) * 16
        library[str(target_id)] = enlarged
    return library
