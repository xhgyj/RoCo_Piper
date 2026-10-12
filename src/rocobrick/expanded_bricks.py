from __future__ import annotations
from typing import Mapping

def expanded_lego_library(base: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    library = {str(key): dict(value) for key, value in base.items()}
    pairs4 = ((12, 13), (2, 14), (3, 15), (4, 16), (5, 17), (6, 18), (9, 19), (10, 20))
    pairs2 = ((12, 21), (2, 22), (3, 23), (4, 24), (5, 25), (6, 26), (9, 27), (10, 28))
    for source_id, target_id in pairs4:
        source = library[str(source_id)]; enlarged = dict(source)
        enlarged['height'] = int(source['height']) * 4; enlarged['width'] = int(source['width']) * 4; enlarged['layers'] = 4
        if 'mass' in source: enlarged['mass'] = float(source['mass']) * 64
        library[str(target_id)] = enlarged
    for source_id, target_id in pairs2:
        source = library[str(source_id)]; enlarged = dict(source)
        enlarged['height'] = int(source['height']) * 2; enlarged['width'] = int(source['width']) * 2; enlarged['layers'] = 2
        if 'mass' in source: enlarged['mass'] = float(source['mass']) * 8
        library[str(target_id)] = enlarged
    return library
