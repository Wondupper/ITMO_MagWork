"""Адаптер ASVspoof 2021 DF (PROJECT_ARCHITECTURE.md §6.2, Приложение А).

Аудио скачано ЧАСТИЧНО (part00 из трёх) — адаптер индексирует присутствующее
аудио и берёт метку из ПОЛНЫХ ключей (Приложение А). Ключи вне папки датасета —
`raw/DF-keys-full/keys/DF/CM/trial_metadata.txt`.
Формат trial_metadata (13 полей):
  `speaker utt codec source attack label trim phase vocoder + 4 поля VCC`.
У bonafide поле атаки = `-` → NULL. Весь набор — родной eval → канон `test`.

Заполняются:
  `condition` ← поле 3: условие сжатия. Наблюдаемые значения (9 шт.):
                nocodec; {high,low}_{mp3,m4a,ogg}; mp3m4a, oggm4a (двойное
                перекодирование). Это главная ось для разбивки EER по кодекам.
  `vocoder`   ← поле 9: вокодер спуф-системы; `-` у части строк → NULL.

ВНИМАНИЕ: `condition` — сжатие СИГНАЛА, а колонка `codec` — формат КОНТЕЙНЕРА
файла на диске (здесь всегда flac). Разные вещи, см. §6.1.

Дропается: `source` (4), `trim` (7), `phase` (8), VCC-мета (10–13) — колонок под
них нет, добавить при надобности (§9).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import (
    codec_from_suffix,
    iter_audio,
    label_to_int,
    norm_attack,
    norm_condition,
    protocol_table,
)
from .base import DatasetAdapter, register


@register
class ASVspoof2021DF(DatasetAdapter):
    dataset_id = "asvspoof2021_df"

    _AUDIO_SUBDIR = "flac"
    _KEYS_RELPATH = Path("DF-keys-full") / "keys" / "DF" / "CM" / "trial_metadata.txt"
    # индексы с нуля: speaker(1) utt(2) codec(3) source(4) attack(5) label(6)
    #                 trim(7) phase(8) vocoder(9)
    _FIELDS = {
        "speaker_id": 0,
        "utt_id": 1,
        "condition": 2,
        "attack": 4,
        "label": 5,
        "vocoder": 8,
    }

    def _keys_path(self) -> Path:
        return self.config.paths.raw / self._KEYS_RELPATH

    def build(self) -> pd.DataFrame:
        keys = protocol_table(
            self._keys_path(), fields=self._FIELDS, dataset_id=self.dataset_id
        )
        rows = []
        for audio in iter_audio(self.raw_dir / self._AUDIO_SUBDIR):
            utt = audio.stem
            if utt not in keys:
                raise KeyError(f"[{self.dataset_id}] {audio.name} отсутствует в trial_metadata")
            f = keys[utt]
            label = label_to_int(f["label"])
            rows.append({
                "dataset_id": self.dataset_id,
                "utt_id": utt,
                "path": audio.relative_to(self.raw_dir).as_posix(),
                "label": label,
                "split": "test",  # eval → test (§6.3)
                "attack_type": norm_attack(f["attack"], label),
                "condition": norm_condition(f["condition"]),
                "vocoder": norm_condition(f["vocoder"]),
                "speaker_id": f["speaker_id"],
                "codec": codec_from_suffix(audio),
            })
        return pd.DataFrame(rows)