"""Ленивый Dataset поверх кэша признаков + коллатор (§6.6, §7).

Читающая сторона контракта признак↔модель. Стадия 1 разложила признаки по
per-file `.npy` с ключом (dataset_id, utt_id) (§6.5); здесь они лениво,
по одному примеру, читаются в обучении — в памяти только текущий батч, не датасет
(§3, §7). `__init__` держит ТОЛЬКО метаданные манифеста (id и метки), не данные:
иначе каждый воркер DataLoader продублировал бы их в памяти (§7).

Длина T у примеров разная. Стадия 2 приводит каждый пример к общей длине
`t_fixed` (кроп длинного / повтор-паддинг короткого) — это и собирает батч в один
тензор, и держит память батча предсказуемой (§3). Приведение к общей длине —
стандарт анти-спуфинга (RawNet2/LCNN фиксируют длину входа). Контракт §6.6 при
этом соблюдён: `lengths` возвращается (при повтор-паддинге все кадры валидны,
lengths == t_fixed), интерфейс модели не меняется; при переходе к честной
переменной длине достаточно сменить коллатор, не трогая модели.

Перенос на устройство здесь НЕ делается — это единственная точка цикла обучения
(§6.6). Dataset и коллатор работают на CPU, отдают обычные тензоры.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def _fix_length(x: np.ndarray, t_fixed: int, *, random_crop: bool = False) -> np.ndarray:
    """Привести [T, C] к [t_fixed, C]: кроп длинного, повтор-паддинг короткого.

    Повтор (а не нулевой паддинг) — доменный стандарт: короткая запись повторяется
    до нужной длины, все кадры остаются «настоящими», пулинг не разбавляется нулями.

    Кроп длинного файла:
      • random_crop=False (валидация/оценка) — детерминированное окно с начала:
        воспроизводимо, один и тот же вход при каждом проходе;
      • random_crop=True (обучение) — случайное окно из записи. Лёгкая аугментация:
        за эпохи модель видит РАЗНЫЕ фрагменты длинных файлов, а не всегда первые
        t_fixed кадров → больше эффективного разнообразия и лучше обобщение (доменный
        стандарт анти-спуфинга). Смещение берётся из torch-ГСЧ, который DataLoader
        сеет по воркерам и эпохам, — прогон остаётся воспроизводимым от общего seed
        (§7), но окно закономерно меняется от эпохи к эпохе.
    Короткий файл повторяется с начала в обоих режимах: весь сигнал и так виден,
    рандомизировать нечего.
    """
    t = x.shape[0]
    if t == t_fixed:
        return x
    if t > t_fixed:
        start = int(torch.randint(0, t - t_fixed + 1, (1,)).item()) if random_crop else 0
        return x[start:start + t_fixed]
    reps = -(-t_fixed // t)                       # ceil(t_fixed / t)
    return np.tile(x, (reps, 1))[:t_fixed]


class FeatureCacheDataset(Dataset):
    """torch Dataset поверх кэша признаков одной конфигурации (§6.5, §6.6).

    Параметры
    ---------
    manifest : DataFrame канонической схемы (§6.1); может быть подвыборкой (§6.4)
        или склейкой нескольких датасетов. Нужны колонки dataset_id, utt_id, label.
    cache_dir : папка кэша признаков `data/features/<name>-<hash>/`
        (= extractor.cache_dir(paths)). Путь примера — cache_dir/<ds>/<utt>.npy,
        в точности как пишет Стадия 1 (features.output_path).
    expected_channels : ожидаемое C (= extractor.channels). Каждый прочитанный
        массив проверяется на [., C] — дешёвый аналог валидатора (§6.6): ловит
        рассинхрон «признак ≠ модель» на уровне данных, а не молча.
    t_fixed : общая длина, к которой приводится каждый пример.
    random_crop : случайное окно из длинных файлов (аугментация обучения) vs окно с
        начала (валидация/оценка). Train-загрузчик ставит True, val — False.

    Отсутствующие в кэше файлы (напр. сбойные на Стадии 1 → в `_failures.csv`, без
    `.npy`) отфильтровываются в `__init__` с предупреждением: так обучение не падает
    на первом же битом ключе, а работает по фактически посчитанному кэшу.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_dir: str | Path,
        expected_channels: int,
        t_fixed: int,
        random_crop: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.expected_channels = int(expected_channels)
        self.t_fixed = int(t_fixed)
        self.random_crop = bool(random_crop)

        present = self._filter_present(manifest)
        # Держим только лёгкие метаданные как numpy-массивы (не DataFrame) — так
        # воркеры DataLoader не тащат pandas-объект целиком (§7).
        self.dataset_ids = present["dataset_id"].astype(str).to_numpy()
        self.utt_ids = present["utt_id"].astype(str).to_numpy()
        self.labels = present["label"].astype(np.float32).to_numpy()  # BCE ждёт float

    def _npy_path(self, dataset_id: str, utt_id: str) -> Path:
        return self.cache_dir / dataset_id / f"{utt_id}.npy"

    def _filter_present(self, manifest: pd.DataFrame) -> pd.DataFrame:
        exists = [
            self._npy_path(str(d), str(u)).exists()
            for d, u in zip(manifest["dataset_id"], manifest["utt_id"])
        ]
        n_missing = len(exists) - int(np.sum(exists))
        if n_missing:
            print(
                f"    ВНИМАНИЕ: {n_missing}/{len(exists)} примеров без .npy в "
                f"{self.cache_dir.name} — пропущены (сбои Стадии 1?). Осталось "
                f"{int(np.sum(exists))}."
            )
        out = manifest.loc[exists].reset_index(drop=True)
        if len(out) == 0:
            raise RuntimeError(
                f"в кэше {self.cache_dir} нет ни одного файла из манифеста — "
                "сначала постройте признаки (Стадия 1) для этих датасетов."
            )
        return out

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        arr = np.load(self._npy_path(self.dataset_ids[i], self.utt_ids[i]))  # [T, C]
        if arr.ndim != 2 or arr.shape[1] != self.expected_channels:
            raise ValueError(
                f"{self.dataset_ids[i]}/{self.utt_ids[i]}: ожидалось [T, "
                f"{self.expected_channels}], получено {tuple(arr.shape)} — "
                "кэш не соответствует признаку модели (§6.6)."
            )
        x = _fix_length(np.ascontiguousarray(arr, dtype=np.float32), self.t_fixed,
                        random_crop=self.random_crop)
        length = self.t_fixed  # при повтор-паддинге все кадры валидны
        return (
            torch.from_numpy(x),                       # [t_fixed, C] float32
            torch.tensor(self.labels[i], dtype=torch.float32),
            length,
        )


def collate(batch):
    """Собрать список примеров в батч (§6.6).

    Все примеры уже длины t_fixed → простой stack. `lengths` возвращается для
    контракта §6.6 (модель делает масочный пулинг); при единой длине маска
    тривиальна. Перенос на device — в цикле обучения, не здесь.
    """
    xs, ys, lengths = zip(*batch)
    x = torch.stack(xs, dim=0)                         # [B, t_fixed, C]
    y = torch.stack(ys, dim=0)                         # [B]
    length = torch.tensor(lengths, dtype=torch.long)   # [B]
    return x, y, length