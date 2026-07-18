"""Стадия 1 — извлечение признаков: манифест → per-file кэш `.npy` (§4, §6.5, §6.6).

Три вещи, ради которых всё устроено именно так:

1. **Реестр экстракторов** — зеркало реестра адаптеров (`sources/base.py`):
   `@register_extractor` кладёт класс в таблицу по имени, выбор идёт по имени
   без if-цепочек. Новый признак = новый класс + декоратор, ничего ниже не трогая.

2. **Контракт формы (§6.6).** Каждый экстрактор отдаёт `float32`-массив `[T, C]`
   (T — кадры по времени, C — каналы). Длину НЕ нормализуем здесь: паддинг/пулинг —
   обязанность коллатора/модели (Стадия 2), кэш хранит сырые `[T, C]`. Так контракт
   «любые признаки → любая модель» остаётся стабильным. C у каждого экстрактора
   статически известен (`channels`) и проверяется на каждом выходе — дешёвый аналог
   валидатора схемы из §6.1.

3. **Возобновляемость (§7), паттерн «пропустить готовое».** Путь выхода
   детерминирован: (dataset_id, utt_id) + хэш конфига признаков. Есть файл →
   пропуск. Иначе: грузим аудио → извлекаем → атомарная запись (tmp + os.replace,
   `utils.io`). Хэш конфига в пути кэша разводит разные наборы признаков по разным
   папкам — при возврате к прежнему набору пересчёта нет.

Аудио грузится РОВНО ОДИН РАЗ в раннере (soundfile → моно → ресемпл в target_sr
экстрактора), поэтому сами экстракторы — чистая математика над сигналом и легко
тестируются. Загрузка одного файла держит в памяти единицы–десятки МБ (§3): пик
памяти не зависит от размера датасета. Декодируем напрямую через soundfile (не
librosa.load): при сбое видно НАСТОЯЩУЮ ошибку libsndfile, а не обёртку audioread,
и нет скрытой зависимости от ffmpeg. Достаточно для flac/wav/ogg (текущие датасеты).

Кэш самоописателен: рядом с признаками пишется `_spec.json` (все параметры +
версии библиотек — воспроизводимость, критерий №2) и, если были сбои,
`<dataset_id>/_failures.csv` с реальной причиной по каждому файлу.

Запуск (как модуль пакета, из корня проекта):
    python -m src.data.features asvspoof2019_la -f lfcc     # один датасет
    python -m src.data.features --all -f logmel             # все с готовым манифестом
    python -m src.data.features --list                      # экстракторы + датасеты
    python -m src.data.features for-2seconds -f lfcc --limit 200   # smoke-тест
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Iterator

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fftpack import dct

from ..config import Config, Paths, default_config
from ..utils.io import atomic_write_npy


# =============================================================================
# Контракт экстрактора (§6.6)
# =============================================================================
class FeatureExtractor(ABC):
    """База экстрактора. Подкласс = frozen dataclass со своими гиперпараметрами
    (они и есть конфиг признаков) + атрибут класса `name` + `extract`/`channels`.

    Параметры-поля dataclass'а автоматически попадают в `spec()` → в хэш кэша, так
    что менять их безопасно: другой набор параметров = другой хэш = другая папка
    кэша, старое не перезаписывается и не путается.
    """

    #: Ключ реестра; совпадает с именем в CLI (-f) и первой частью папки кэша.
    name: ClassVar[str] = ""

    @property
    @abstractmethod
    def target_sr(self) -> int:
        """Частота, к которой раннер ресемплит аудио перед extract()."""

    @property
    @abstractmethod
    def channels(self) -> int:
        """Ожидаемое C выхода [T, C] — статически известно из параметров."""

    @abstractmethod
    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Сигнал (моно float32 @ target_sr) → массив признаков `[T, C]` float32."""

    # --- общее для всех экстракторов ---------------------------------------
    def spec(self) -> dict:
        """Канонический словарь для хэша: имя + все поля-параметры."""
        return {"name": self.name, **asdict(self)}

    def cache_hash(self, n: int = 10) -> str:
        """Короткий sha1 от отсортированного JSON конфига признаков (§6.5)."""
        blob = json.dumps(self.spec(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:n]

    def cache_dir(self, paths: Paths) -> Path:
        """data/features/<name>-<hash>/ — одна папка на конфигурацию признаков."""
        return paths.features / f"{self.name}-{self.cache_hash()}"

    def output_path(self, paths: Paths, dataset_id: str, utt_id: str) -> Path:
        """Детерминированный путь одного примера: ключ (dataset_id, utt_id)."""
        return self.cache_dir(paths) / dataset_id / f"{utt_id}.npy"


# =============================================================================
# Реестр экстракторов (зеркало sources/base.py)
# =============================================================================
_EXTRACTORS: dict[str, type[FeatureExtractor]] = {}


def register_extractor(cls: type[FeatureExtractor]) -> type[FeatureExtractor]:
    """Декоратор: зарегистрировать класс экстрактора по его `name`."""
    if not cls.name:
        raise ValueError(f"{cls.__name__}: без атрибута name нельзя зарегистрировать")
    if cls.name in _EXTRACTORS:
        raise ValueError(f"экстрактор '{cls.name}' уже зарегистрирован")
    _EXTRACTORS[cls.name] = cls
    return cls


def get_extractor(name: str, **overrides) -> FeatureExtractor:
    """Собрать экстрактор по имени. `overrides` — точка входа для будущего
    YAML-слоя оверрайдов (§7); без них берутся дефолтные гиперпараметры."""
    if name not in _EXTRACTORS:
        raise KeyError(f"нет экстрактора '{name}'. Зарегистрированы: {available_extractors()}")
    return _EXTRACTORS[name](**overrides)


def available_extractors() -> list[str]:
    return sorted(_EXTRACTORS)


# =============================================================================
# Экстракторы
# =============================================================================
@register_extractor
@dataclass(frozen=True)
class LogMelSpectrogram(FeatureExtractor):
    """Лог-мел-спектрограмма — крайний «простой» случай контракта: C = n_mels.

    power_to_db с ref=1.0 и top_db=None намеренно: без per-file нормировки к пику,
    чтобы кэш был воспроизводим и сравним между файлами (нормализацию, если нужна,
    делает Стадия 2, а не кэш).
    """
    name: ClassVar[str] = "logmel"

    target_sr: int = 16000       # стандарт ASVspoof (§ Приложение А)
    n_fft: int = 512
    hop_length: int = 160        # 10 мс @ 16 кГц
    win_length: int = 400        # 25 мс @ 16 кГц
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float | None = None    # None → Найквист

    @property
    def channels(self) -> int:
        return self.n_mels

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax,
        )  # [n_mels, T], мощность
        logmel = librosa.power_to_db(mel, ref=1.0, top_db=None)  # [n_mels, T], дБ
        return logmel.T.astype(np.float32)  # [T, n_mels]


@register_extractor
@dataclass(frozen=True)
class LFCC(FeatureExtractor):
    """Linear Frequency Cepstral Coefficients — сквозной бейзлайн анти-спуфинга
    (LFCC-GMM/LCNN есть даже в ключах 2021, Приложение А).

    Отличие от MFCC — ЛИНЕЙНЫЙ треугольный фильтробанк вместо мел. Конвейер:
    STFT-мощность → линейный фильтробанк → log → DCT-II (оставляем n_lfcc) →
    [опц.] Δ и ΔΔ по времени. При with_deltas=True C = 3·n_lfcc (по умолч. 60 —
    классический LFCC-60).
    """
    name: ClassVar[str] = "lfcc"

    target_sr: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    n_filter: int = 20           # число линейных треугольных фильтров
    n_lfcc: int = 20             # сколько кепстральных коэффициентов оставить
    with_deltas: bool = True     # добавить Δ и ΔΔ → C = 3·n_lfcc
    delta_width: int = 9         # окно вычисления дельт (нечётное)
    log_eps: float = 1e-10       # стабилизатор log(0)

    @property
    def channels(self) -> int:
        return self.n_lfcc * (3 if self.with_deltas else 1)

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        S = np.abs(librosa.stft(
            y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
        )) ** 2  # [1+n_fft/2, T], мощность
        fb = _linear_filterbank(sr, self.n_fft, self.n_filter)  # [n_filter, 1+n_fft/2]
        filtered = fb @ S                                       # [n_filter, T]
        log_energy = np.log(filtered + self.log_eps)
        coeffs = dct(log_energy, type=2, axis=0, norm="ortho")[: self.n_lfcc]  # [n_lfcc, T]
        if self.with_deltas:
            w = _odd_width(self.delta_width, coeffs.shape[1])
            d1 = librosa.feature.delta(coeffs, width=w, order=1)
            d2 = librosa.feature.delta(coeffs, width=w, order=2)
            coeffs = np.concatenate([coeffs, d1, d2], axis=0)  # [3·n_lfcc, T]
        return coeffs.T.astype(np.float32)  # [T, C]


@lru_cache(maxsize=8)
def _linear_filterbank(sr: int, n_fft: int, n_filter: int) -> np.ndarray:
    """Линейный треугольный фильтробанк `[n_filter, 1+n_fft/2]`.

    Аналог мел-банка, но центры фильтров равномерны по частоте (не по мел-шкале).
    Кэшируется: параметры одни на весь прогон, строится один раз.
    """
    n_bins = 1 + n_fft // 2
    bin_freqs = np.linspace(0.0, sr / 2.0, n_bins)
    edges = np.linspace(0.0, sr / 2.0, n_filter + 2)  # n_filter центров + 2 края
    fb = np.zeros((n_filter, n_bins), dtype=np.float64)
    for m in range(1, n_filter + 1):
        left, center, right = edges[m - 1], edges[m], edges[m + 1]
        rise = (bin_freqs >= left) & (bin_freqs <= center)
        fall = (bin_freqs >= center) & (bin_freqs <= right)
        fb[m - 1, rise] = (bin_freqs[rise] - left) / max(center - left, 1e-12)
        fb[m - 1, fall] = (right - bin_freqs[fall]) / max(right - center, 1e-12)
    return fb.astype(np.float32)


def _odd_width(width: int, t: int) -> int:
    """Ужать окно дельт до нечётного ≥3 и ≤ T. Слишком короткий сигнал → ошибка
    (файл будет помечён как сбойный, а не даст выход с несовместимым C)."""
    w = min(width, t)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        raise ValueError(f"сигнал слишком короткий для дельт: T={t}")
    return w


# =============================================================================
# Стадия 1 — прогон по датасету
# =============================================================================
@dataclass
class FeatureRunStats:
    """Итог прогона одного датасета — для сводки и кода возврата раннера."""
    dataset_id: str
    extractor: str
    cache_dir: Path
    total: int
    written: int
    skipped: int
    failures: list[tuple[str, str, str]]   # (utt_id, path, error)
    failures_path: Path | None = None

    @property
    def ok(self) -> bool:
        # Дефект уровня датасета = непустой вход, но 0 успехов (систематический сбой).
        # Единичные битые файлы (failures) сами по себе прогон не роняют.
        succeeded = self.written + self.skipped
        return not (self.total > 0 and succeeded == 0)

    def summary(self) -> str:
        lines = [
            f"    всего: {self.total}  записано: {self.written}  "
            f"пропущено(готово): {self.skipped}  ошибок: {len(self.failures)}"
        ]
        for utt, _path, msg in self.failures[:5]:
            lines.append(f"      FAIL {utt}: {msg}")
        if len(self.failures) > 5:
            lines.append(f"      … ещё {len(self.failures) - 5} (полный список: {self.failures_path})")
        return "\n".join(lines)


def _validate_feature(arr: np.ndarray, extractor: FeatureExtractor) -> np.ndarray:
    """Контракт §6.6: ровно [T, C], объявленное C, float32, конечные значения."""
    if arr.ndim != 2:
        raise ValueError(f"ожидалось [T, C], получено ndim={arr.ndim}, shape={arr.shape}")
    if arr.shape[1] != extractor.channels:
        raise ValueError(f"C={arr.shape[1]} ≠ объявленного {extractor.channels}")
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("в признаках NaN/Inf")
    return arr


def _load_audio(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """Декодировать аудио → моно float32 @ target_sr. Ошибку декодера НЕ прячем —
    она уходит наверх как есть (LibsndfileError с внятным текстом), в отличие от
    librosa.load, который маскирует её обёрткой audioread."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:                      # многоканальное → моно усреднением
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)  # тот же soxr
    return np.ascontiguousarray(y, dtype=np.float32), target_sr


def _write_spec(extractor: FeatureExtractor, cache_dir: Path) -> None:
    """Положить `_spec.json` в корень папки кэша — паспорт конфигурации признаков.
    Делает папку самоописательной: по хэшу видно, чем именно она получена."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "spec": extractor.spec(),
        "channels": extractor.channels,
        "cache_hash": extractor.cache_hash(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "librosa": librosa.__version__,
            "soundfile": sf.__version__,
            "libsndfile": sf.__libsndfile_version__,
            "numpy": np.__version__,
        },
    }
    (cache_dir / "_spec.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_failures(failures: list[tuple[str, str, str]], cache_dir: Path, dataset_id: str) -> Path | None:
    """Сохранить список сбойных файлов с реальной причиной → честная цифра «N не
    декодировалось» для работы, а не обрезанный вывод в консоль. Пусто → чистим
    старый файл (сбои исправлены). Отражает последний прогон по этому датасету."""
    out = cache_dir / dataset_id / "_failures.csv"
    if not failures:
        if out.exists():
            out.unlink()
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["utt_id", "path", "error"]).to_csv(out, index=False)
    return out


def extract_features(
    dataset_id: str,
    extractor: FeatureExtractor,
    config: Config | None = None,
    *,
    limit: int | None = None,
    log_every: int = 2000,
) -> FeatureRunStats:
    """Извлечь признаки для одного датасета: манифест → per-file `.npy` кэш.

    Читает data/manifests/<dataset_id>.parquet (только колонки-метаданные — память
    не зависит от размера аудио). Строки стабильно сортируются по (dataset_id,
    utt_id) → детерминизм прогресса и воспроизводимость `--limit`. Возобновляемо:
    готовые файлы пропускаются; единичные сбойные файлы логируются и не роняют
    прогон.
    """
    config = config or default_config()
    manifest_path = config.paths.manifest_path(dataset_id)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"нет манифеста {manifest_path} — сначала Стадия 0: "
            f"python -m src.data.manifest {dataset_id}"
        )

    # Только метаданные, нужные для загрузки и ключа кэша (§7: не тянем лишнего).
    df = pd.read_parquet(manifest_path, columns=["dataset_id", "utt_id", "path"])
    df = df.sort_values(["dataset_id", "utt_id"], kind="stable").reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)

    raw_root = config.paths.raw_dataset(dataset_id)
    cache_dir = extractor.cache_dir(config.paths)
    _write_spec(extractor, cache_dir)  # паспорт конфигурации — до тяжёлой работы
    total = len(df)
    written = skipped = 0
    failures: list[tuple[str, str, str]] = []
    proc = _maybe_process()

    for i, row in enumerate(_progress(df.itertuples(index=False), total, dataset_id, extractor.name)):
        out = extractor.output_path(config.paths, dataset_id, row.utt_id)
        if out.exists():
            skipped += 1
            continue
        try:
            y, sr = _load_audio(str(raw_root / row.path), extractor.target_sr)
            arr = _validate_feature(extractor.extract(y, sr), extractor)
            atomic_write_npy(arr, out)
            written += 1
        except Exception as e:  # noqa: BLE001 — один битый файл не должен ронять прогон
            failures.append((str(row.utt_id), str(row.path), f"{type(e).__name__}: {e}"))
        if proc is not None and log_every and (i + 1) % log_every == 0:
            rss = proc.memory_info().rss / 2 ** 20
            print(f"    … {i + 1}/{total}  RSS={rss:.0f} МБ", flush=True)

    failures_path = _write_failures(failures, cache_dir, dataset_id)
    return FeatureRunStats(dataset_id, extractor.name, cache_dir, total, written, skipped,
                           failures, failures_path)


# --- необязательные удобства прогона (мягкие зависимости) --------------------
def _maybe_process():
    """psutil.Process для лога RSS, если psutil доступен; иначе None (§7)."""
    try:
        import psutil
        return psutil.Process()
    except Exception:  # noqa: BLE001
        return None


def _progress(it: Iterator, total: int, dataset_id: str, name: str) -> Iterator:
    """Обернуть в tqdm, если он есть; иначе — как есть."""
    try:
        from tqdm import tqdm
        return tqdm(it, total=total, desc=f"[{dataset_id}] {name}", unit="file")
    except Exception:  # noqa: BLE001
        return it


# =============================================================================
# Тонкий раннер стадии (python -m src.data.features ...)
# =============================================================================
# Это НЕ глобальная точка входа/диспетчер (§4 её отвергает), а независимый запуск
# одной стадии — как раннер Стадии 0. У Стадии 1 два измерения: датасет(ы) и один
# экстрактор за прогон (одна конфигурация признаков = одна папка кэша).

def _available_datasets(config: Config) -> list[str]:
    """Датасеты с уже построенным манифестом (вход Стадии 1). subsets/ не попадают
    (glob по одному уровню)."""
    d = config.paths.manifests
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.parquet"))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.data.features",
        description="Стадия 1 (признаки): манифест → per-file .npy кэш.",
    )
    parser.add_argument("datasets", nargs="*", help="dataset_id (один или несколько)")
    parser.add_argument("-f", "--features", metavar="NAME",
                        help="имя экстрактора (список — в --list)")
    parser.add_argument("--all", action="store_true",
                        help="все датасеты с готовым манифестом")
    parser.add_argument("--list", action="store_true",
                        help="показать экстракторы и доступные датасеты, выйти")
    parser.add_argument("--limit", type=int, default=None,
                        help="только первые N примеров (smoke-тест; НЕ для итоговых метрик)")
    parser.add_argument("--root", type=Path, default=None,
                        help="корень проекта (по умолчанию — автоопределение)")
    args = parser.parse_args(argv)

    config = Config(paths=Paths(root=args.root)) if args.root else default_config()

    if args.list:
        print("Экстракторы:")
        for n in available_extractors():
            ex = get_extractor(n)
            print(f"  {n:<10} C={ex.channels}  sr={ex.target_sr}  hash={ex.cache_hash()}")
        ds = _available_datasets(config)
        print("Датасеты с готовым манифестом:")
        if ds:
            for d in ds:
                print(f"  {d}")
        else:
            print("  (нет — сначала Стадия 0: python -m src.data.manifest ...)")
        return 0

    if not args.features:
        parser.error("укажите экстрактор через -f/--features (список: --list)")
    try:
        extractor = get_extractor(args.features)
    except KeyError as e:
        parser.error(str(e))

    targets = _available_datasets(config) if args.all else args.datasets
    if not targets:
        parser.error("укажите dataset_id, либо --all, либо --list")

    print(f"экстрактор: {extractor.name}  C={extractor.channels}  "
          f"sr={extractor.target_sr}  hash={extractor.cache_hash()}")
    print(f"кэш: {extractor.cache_dir(config.paths)}")
    if args.limit is not None:
        print(f"ВНИМАНИЕ: --limit {args.limit} — только smoke-тест, не для итоговых метрик")

    failed: list[str] = []
    for ds in targets:
        print(f"[{ds}] Стадия 1 ({extractor.name})…", flush=True)
        try:
            stats = extract_features(ds, extractor, config, limit=args.limit)
        except Exception as e:  # noqa: BLE001 — раннеру важно не падать на первом датасете
            print(f"[{ds}] ОШИБКА: {type(e).__name__}: {e}")
            failed.append(ds)
            continue
        print(stats.summary())
        if not stats.ok:
            failed.append(ds)

    if failed:
        print(f"\nне удалось: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())