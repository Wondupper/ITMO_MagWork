"""Адаптер FoR / for-2seconds (PROJECT_ARCHITECTURE.md §6.2, Приложение А).

Протокола нет — «крайний случай» контракта (ничего, кроме имён папок). Метка из
имени папки (`real`→bonafide=0, `fake`→spoof=1), сплит из верхней папки
(`training/validation/testing` → `train/dev/test`). Дикторов/атак нет → NULL.

utt_id выводится из относительного пути (без расширения, `/`→`__`), чтобы быть
уникальным внутри датасета и безопасным как часть ключа кэша (§6.5): имена файлов
между папками real/fake могут совпадать, поэтому одного stem мало.
"""
from __future__ import annotations

import pandas as pd

from ..schema import LABEL_BONAFIDE, LABEL_SPOOF
from .common import codec_from_suffix, iter_audio
from .base import DatasetAdapter, register


@register
class FoR2Seconds(DatasetAdapter):
    dataset_id = "for-2seconds"

    _SPLIT_DIR = {"training": "train", "validation": "dev", "testing": "test"}
    _CLASS_LABEL = {"real": LABEL_BONAFIDE, "fake": LABEL_SPOOF}

    def build(self) -> pd.DataFrame:
        rows = []
        for split_dir, split in self._SPLIT_DIR.items():
            for cls, label in self._CLASS_LABEL.items():
                root = self.raw_dir / split_dir / cls
                if not root.exists():
                    continue  # допускаем отсутствие какого-то класса/сплита
                for audio in iter_audio(root):
                    rel = audio.relative_to(self.raw_dir).as_posix()
                    utt = rel.rsplit(".", 1)[0].replace("/", "__")
                    rows.append({
                        "dataset_id": self.dataset_id,
                        "utt_id": utt,
                        "path": rel,
                        "label": label,
                        "split": split,
                        "codec": codec_from_suffix(audio),
                    })
        if not rows:
            raise FileNotFoundError(
                f"[{self.dataset_id}] не найдено аудио под {self.raw_dir} "
                "(ожидались папки training/validation/testing с real/fake)"
            )
        return pd.DataFrame(rows)