"""Адаптер ASVspoof 2019 LA (PROJECT_ARCHITECTURE.md §6.2, Приложение А).

Индексация ведётся ПО CM-ПРОТОКОЛУ (`ASVspoof2019_LA_cm_protocols/`, по одному
файлу на сплит) — манифест = ровно CM-триалы. В папках flac у dev/eval лежат
также ASV-enrollment записи (напр. `LA_D_A…`), которых в CM-протоколе нет; они в
манифест НЕ попадают. Сплит определяется парой (протокол ↔ папка аудио), а не
префиксом utt_id. Формат CM-строки: `speaker utt - attack label` (поле 3 = `-`).

Условий записи (кодек/канал) в 2019 LA нет — колонки `condition`/`transmission`
остаются NULL, и это содержательно: «поля нет в датасете», а не «неизвестно».
Отличать этот NULL от `condition = "none"` у LA-2021 (там сжатие сознательно не
применяли) обязательно — иначе разбивка EER по условиям смешает разные вещи.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import codec_from_suffix, iter_audio, label_to_int, norm_attack, protocol_table
from .base import DatasetAdapter, register


@register
class ASVspoof2019LA(DatasetAdapter):
    dataset_id = "asvspoof2019_la"

    _PROTO_DIR = "ASVspoof2019_LA_cm_protocols"
    # родной сплит датасета (папка аудио) → канон {train, dev, test}, §6.3
    _SPLIT_SUBDIR = {
        "train": "ASVspoof2019_LA_train",
        "dev": "ASVspoof2019_LA_dev",
        "test": "ASVspoof2019_LA_eval",   # eval → test
    }
    # ключевое слово для выбора файла CM-протокола данного сплита
    _SPLIT_PROTO_KEYWORD = {"train": "train", "dev": "dev", "test": "eval"}
    # CM-строка: speaker(1) utt(2) -(3) attack(4) label(5); индексы с нуля
    _FIELDS = {"speaker_id": 0, "utt_id": 1, "attack": 3, "label": 4}

    def _protocol_for(self, split: str) -> Path:
        proto_dir = self.raw_dir / self._PROTO_DIR
        kw = self._SPLIT_PROTO_KEYWORD[split]
        cands = sorted(p for p in proto_dir.glob("*.txt") if kw in p.name.lower())
        if len(cands) != 1:
            raise FileNotFoundError(
                f"[{self.dataset_id}] ожидался ровно 1 CM-протокол с '{kw}' в "
                f"{proto_dir}, найдено {len(cands)}: {[p.name for p in cands]}"
            )
        return cands[0]

    def build(self) -> pd.DataFrame:
        rows = []
        for split, subdir in self._SPLIT_SUBDIR.items():
            proto = protocol_table(
                self._protocol_for(split),
                fields=self._FIELDS,
                dataset_id=self.dataset_id,
            )
            flac_dir = self.raw_dir / subdir / "flac"
            # Индексируем ТОЛЬКО записи CM-протокола. В папках flac у dev/eval лежат
            # ещё и ASV-enrollment файлы (напр. LA_D_A…), которых нет в CM-протоколе —
            # для детекции спуфинга они не нужны и в манифест не идут.
            present = {p.stem: p for p in iter_audio(flac_dir)}
            missing = [u for u in proto if u not in present]
            if missing:
                raise FileNotFoundError(
                    f"[{self.dataset_id}] {split}: {len(missing)} записей CM-протокола "
                    f"без аудио в {flac_dir}, напр. {missing[:3]}"
                )
            for utt, f in proto.items():
                audio = present[utt]
                label = label_to_int(f["label"])
                rows.append({
                    "dataset_id": self.dataset_id,
                    "utt_id": utt,
                    "path": audio.relative_to(self.raw_dir).as_posix(),
                    "label": label,
                    "split": split,
                    "attack_type": norm_attack(f["attack"], label),
                    "speaker_id": f["speaker_id"],
                    "codec": codec_from_suffix(audio),
                })
        return pd.DataFrame(rows)