"""Адаптер `mini` — крошечный датасет для smoke-прогонов (§6.2, §6.4).

Отличается от остальных адаптеров только происхождением папки: `data/raw/mini/`
не скачивается, а собирается командой `python -m src.data.mini` из уже
проиндексированных датасетов (по 5 подлинных + 5 спуф на источник). Для самого
адаптера это неважно — он, как и любой другой, просто читает родной протокол
своей папки, здесь это `_index.csv`, и выдаёт канонический манифест.

Метка, сплит и `attack_type` берутся из протокола готовыми: их уже разрешил
адаптер источника при построении его манифеста, и разрешать заново означало бы
завести вторую точку истины. `speaker_id` / `condition` / `vocoder` намеренно НЕ
переносятся — на 40 записях они бесполезны, а NULL здесь честнее (§6.1).

⚠️ Метрики с `mini` в работу не идут: записи взяты из всех сплитов источников и
переразмечены заново, то есть утечка между train и test заведомо есть. Датасет
проверяет, что код работает, а не насколько хорошо работает метод (§6.4).
"""
from __future__ import annotations

import pandas as pd

from ..mini import INDEX_NAME, MINI_DATASET_ID
from .base import DatasetAdapter, register
from .common import codec_from_suffix


@register
class Mini(DatasetAdapter):
    dataset_id = MINI_DATASET_ID

    def build(self) -> pd.DataFrame:
        index_path = self.raw_dir / INDEX_NAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"[{self.dataset_id}] нет протокола {index_path} — сначала соберите "
                "датасет: python -m src.data.mini"
            )
        df = pd.read_csv(index_path, dtype=str, keep_default_na=True)

        rows = []
        for r in df.itertuples(index=False):
            audio = self.raw_dir / str(r.path)
            if not audio.exists():
                raise FileNotFoundError(
                    f"[{self.dataset_id}] протокол ссылается на отсутствующий файл "
                    f"{audio} — пересоберите: python -m src.data.mini --force"
                )
            rows.append({
                "dataset_id": self.dataset_id,
                "utt_id": str(r.utt_id),
                "path": str(r.path),
                "label": int(r.label),
                "split": str(r.split),
                # NaN из CSV → NULL (§6.1); у подлинных записей атаки нет по смыслу.
                "attack_type": pd.NA if pd.isna(r.attack_type) else str(r.attack_type),
                "codec": codec_from_suffix(audio),
            })
        if not rows:
            raise ValueError(
                f"[{self.dataset_id}] протокол {index_path} пуст — пересоберите датасет."
            )
        return pd.DataFrame(rows)