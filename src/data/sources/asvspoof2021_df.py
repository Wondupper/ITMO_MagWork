"""Адаптер ASVspoof 2021 DF (PROJECT_ARCHITECTURE.md §6.2, Приложение А).

Аудио скачано ЧАСТИЧНО (part00 из трёх) — адаптер индексирует присутствующее
аудио и берёт метку из ПОЛНЫХ ключей (Приложение А). Ключи вне папки датасета —
`raw/DF-keys-full/keys/DF/CM/trial_metadata.txt`.
Формат trial_metadata (13 полей):
  `speaker utt codec source attack label trim phase vocoder + 4 поля VCC`.
У bonafide поле атаки = `-` → NULL. Весь набор — родной eval → канон `test`.

Дропаются (пока нет колонок, §9): codec-условие (3), source (4), phase (8),
vocoder (9) — vocoder удобен для разбивки EER, добавить опц. колонкой при надобности.
Колонка `codec` = кодек контейнера (flac), не поле 3.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import codec_from_suffix, iter_audio, label_to_int, norm_attack, read_protocol
from .base import DatasetAdapter, register


@register
class ASVspoof2021DF(DatasetAdapter):
    dataset_id = "asvspoof2021_df"

    _AUDIO_SUBDIR = "flac"
    _KEYS_RELPATH = Path("DF-keys-full") / "keys" / "DF" / "CM" / "trial_metadata.txt"

    def _keys_path(self) -> Path:
        return self.config.paths.raw / self._KEYS_RELPATH

    def _parse_keys(self) -> dict[str, tuple[str, str, str]]:
        table: dict[str, tuple[str, str, str]] = {}
        for f in read_protocol(self._keys_path()):
            if len(f) < 6:
                raise ValueError(f"[{self.dataset_id}] короткая строка keys: {f}")
            speaker, utt, attack, label = f[0], f[1], f[4], f[5]
            table[utt] = (speaker, attack, label)
        return table

    def build(self) -> pd.DataFrame:
        keys = self._parse_keys()
        rows = []
        for audio in iter_audio(self.raw_dir / self._AUDIO_SUBDIR):
            utt = audio.stem
            if utt not in keys:
                raise KeyError(f"[{self.dataset_id}] {audio.name} отсутствует в trial_metadata")
            speaker, attack, label_str = keys[utt]
            label = label_to_int(label_str)
            rows.append({
                "dataset_id": self.dataset_id,
                "utt_id": utt,
                "path": audio.relative_to(self.raw_dir).as_posix(),
                "label": label,
                "split": "test",  # eval → test (§6.3)
                "attack_type": norm_attack(attack, label),
                "speaker_id": speaker,
                "codec": codec_from_suffix(audio),
            })
        return pd.DataFrame(rows)