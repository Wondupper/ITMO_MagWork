"""Адаптер ASVspoof 2019 LA (PROJECT_ARCHITECTURE.md §6.2, Приложение А).

Метки — из CM-протоколов (`ASVspoof2019_LA_cm_protocols/`), по одному файлу на
сплит. Сплит берётся из ПАПКИ аудио (`_train/_dev/_eval`), а не из префикса utt_id.
Формат CM-строки: `speaker utt - attack label` (поле 3 всегда `-`).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import codec_from_suffix, iter_audio, label_to_int, norm_attack, read_protocol
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

    def _parse_protocol(self, path: Path) -> dict[str, tuple[str, str, str]]:
        table: dict[str, tuple[str, str, str]] = {}
        for f in read_protocol(path):
            if len(f) < 5:
                raise ValueError(f"[{self.dataset_id}] короткая CM-строка: {f}")
            speaker, utt, attack, label = f[0], f[1], f[3], f[4]  # f[2] == '-'
            table[utt] = (speaker, attack, label)
        return table

    def build(self) -> pd.DataFrame:
        rows = []
        for split, subdir in self._SPLIT_SUBDIR.items():
            proto = self._parse_protocol(self._protocol_for(split))
            for audio in iter_audio(self.raw_dir / subdir / "flac"):
                utt = audio.stem
                if utt not in proto:
                    raise KeyError(
                        f"[{self.dataset_id}] {audio.name} отсутствует в протоколе '{split}'"
                    )
                speaker, attack, label_str = proto[utt]
                label = label_to_int(label_str)
                rows.append({
                    "dataset_id": self.dataset_id,
                    "utt_id": utt,
                    "path": audio.relative_to(self.raw_dir).as_posix(),
                    "label": label,
                    "split": split,
                    "attack_type": norm_attack(attack, label),
                    "speaker_id": speaker,
                    "codec": codec_from_suffix(audio),
                })
        return pd.DataFrame(rows)