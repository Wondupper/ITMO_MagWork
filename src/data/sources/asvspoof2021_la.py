"""Адаптер ASVspoof 2021 LA (PROJECT_ARCHITECTURE.md §6.2, Приложение А).

Аудио: `raw/asvspoof2021_la/flac/`. Ключи лежат ВНЕ папки датасета —
`raw/LA-keys-full/keys/LA/CM/trial_metadata.txt` (это датасет-специфика §6.2).
Формат trial_metadata (8 полей): `speaker utt codec transmission attack label trim phase`.
Весь набор — родной eval → канон `test`.

Условия тракта пишутся в ДВЕ колонки, а не в одну склейку:
  `condition`    ← поле 3: кодек сигнала. Наблюдаемые значения:
                   none, alaw, ulaw, g722, gsm, opus, pstn.
  `transmission` ← поле 4: канал передачи. Наблюдаемые значения:
                   loc_tx, sin_tx, ita_tx, mad_tx, `-` (у none) → NULL.
Колонки атомарные (одно поле — один факт): разбивку по паре легко получить через
groupby обеих, а вот разобрать склеенную строку обратно — уже парсинг. Поля
связаны не свободно (`pstn` встречается только с `mad_tx`, `none` — только с `-`),
поэтому пара кодек×канал НЕ полный декартов набор; на группировку это не влияет.

ВНИМАНИЕ: `condition` — кодек СИГНАЛА, а колонка `codec` — формат КОНТЕЙНЕРА
файла на диске (здесь всегда flac). Разные вещи, см. §6.1.

Дропается: `trim` (7) и `phase` (8, progress/eval/hidden) — колонок под них нет,
добавить при надобности считать метрики на конкретной фазе (§9).
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
class ASVspoof2021LA(DatasetAdapter):
    dataset_id = "asvspoof2021_la"

    _AUDIO_SUBDIR = "flac"
    _KEYS_RELPATH = Path("LA-keys-full") / "keys" / "LA" / "CM" / "trial_metadata.txt"
    # индексы с нуля: speaker(1) utt(2) codec(3) transmission(4) attack(5) label(6)
    _FIELDS = {
        "speaker_id": 0,
        "utt_id": 1,
        "condition": 2,
        "transmission": 3,
        "attack": 4,
        "label": 5,
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
                "transmission": norm_condition(f["transmission"]),
                "speaker_id": f["speaker_id"],
                "codec": codec_from_suffix(audio),
            })
        return pd.DataFrame(rows)