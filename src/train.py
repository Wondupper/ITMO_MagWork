"""Стадия 2 — обучение: кэш признаков → веса модели (§4, §6.6, §7).

Цикл обучения МОДЕЛЬ-АГНОСТИЧЕН: он не знает конкретной сети и признака, а берёт их
из реестров по имени (`models/`, `features.py`) и работает через контракты §6.6.
Смена архитектуры/признака = правка одной строки в experiments/<exp>.yaml, код
цикла не меняется. Это и есть смысл Стадии 2 (§1, §9).

Что делает `run(config, spec)`:
  1. резолвит признак → его C и папку кэша; PRE-FLIGHT сверяет C кэша с C, который
     объявляет модель (§6.6) — рассинхрон «признак ≠ модель» ловится ДО обучения,
     как валидатор схемы на Стадии 0 (§6.1);
  2. читает train/val-манифесты по селекторам (dataset_id, split), при нескольких —
     склеивает (§3); опционально берёт детерминированную подвыборку (§6.4);
  3. лениво грузит признаки через Dataset/DataLoader — в памяти только батч (§3,§7);
  4. учит; в конце каждой эпохи считает val-EER (общая утилита, §8), пишет строку в
     metrics.jsonl и чекпойнт;
  5. ВОЗОБНОВЛЯЕМ (§7): при рестарте подхватывает свежий чекпойнт и продолжает;
     ротация последних K чекпойнтов, атомарная запись; отдельно хранит лучшие по
     val-EER веса (best.pt) и финальные (final.pt).

Артефакты — в experiments/<name>/ (§6.5): metrics.jsonl, ckpt_epoch*.pt, best.pt,
final.pt. `device`/`batch_size`/`num_workers`/`seed` — параметры (§2,§3), берутся из
config.runtime и переопределяются секцией runtime в YAML; перенос батча на device —
единственной точкой в цикле (§6.6).

Запуск (модуль пакета, из корня проекта — как Стадии 0/1, §4):
    python -m src.train --config experiments/smoke_lfcc_statpool.yaml
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config, Paths, Runtime, default_config
from .data.dataset import FeatureCacheDataset, collate
from .data.features import available_extractors, get_extractor
from .data.subsets import DEFAULT_STRATIFY, make_subset
from .models import available_models, get_model
from .utils.metrics import compute_eer

# Колонки манифеста, реально нужные Стадии 2 (§7: не тянем лишнего).
_BASE_COLUMNS = ["dataset_id", "utt_id", "label", "split"]


# =============================================================================
# Воспроизводимость и устройство
# =============================================================================
def _set_seed(seed: int) -> None:
    """Зафиксировать ГСЧ (§7). Бит-в-бит детерминизм не самоцель — нужен
    воспроизводимый прогон, который можно приложить к работе (критерий №2)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(name: str) -> torch.device:
    """Уважать конфиг, но не падать, если cuda запрошена, а её нет (§3)."""
    if name == "cuda" and not torch.cuda.is_available():
        print("    ВНИМАНИЕ: запрошено device=cuda, но CUDA недоступна — беру cpu.")
        return torch.device("cpu")
    return torch.device(name)


# =============================================================================
# Данные: манифесты → (подвыборка) → Dataset/DataLoader
# =============================================================================
def _load_selectors(paths: Paths, selectors: list[dict], stratify: Iterable[str]) -> pd.DataFrame:
    """Прочитать нужные строки манифеста(ов) по списку (dataset_id, split) и склеить.

    Читаются только колонки, нужные Стадии 2 плюс колонки стратификации (§7).
    Единственные законные операции над dataset_id — фильтр по split и concat
    (правило не-ветвления, §3): никакого `if dataset_id == …`.
    """
    need = list(dict.fromkeys([*_BASE_COLUMNS, *stratify]))  # уникальные, порядок стабилен
    frames = []
    for sel in selectors:
        ds, sp = sel["dataset_id"], sel["split"]
        mpath = paths.manifest_path(ds)
        if not mpath.exists():
            raise FileNotFoundError(
                f"нет манифеста {mpath} — сначала Стадия 0: "
                f"python -m src.data.manifest {ds}"
            )
        df = pd.read_parquet(mpath, columns=[c for c in need if c is not None])
        df = df[df["split"] == sp]
        if df.empty:
            raise ValueError(f"[{ds}] нет строк со split='{sp}' — проверьте селектор.")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _maybe_subset(
    df: pd.DataFrame, n: int | None, seed: int, stratify: Iterable[str]
) -> pd.DataFrame:
    """Подвыборка (§6.4), если задан n; иначе полный df (стабильно отсортированный)."""
    return make_subset(df, n, seed=seed, stratify=tuple(stratify))


def _make_loader(
    manifest: pd.DataFrame, cache_dir: Path, channels: int, t_fixed: int,
    batch_size: int, num_workers: int, *, shuffle: bool, random_crop: bool = False,
) -> DataLoader:
    ds = FeatureCacheDataset(manifest, cache_dir, channels, t_fixed, random_crop=random_crop)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate, pin_memory=False, drop_last=False,
    )


# =============================================================================
# Дисбаланс классов: pos_weight для BCE (§ доменные заметки, метки spoof=1)
# =============================================================================
def _pos_weight(labels: np.ndarray, mode: Any) -> torch.Tensor | None:
    """Вес положительного класса для BCEWithLogitsLoss.

    В train 2019 LA спуфа (1) заметно больше, чем bonafide (0); без коррекции модель
    смещается к «всё спуф». `pos_weight = N_neg / N_pos` уравнивает вклад классов —
    одна величина, память/скорость не меняет. mode: "auto" | "none" | число.
    Считается по ФАКТИЧЕСКИ используемым (в т.ч. подвыборочным) train-меткам.
    """
    if mode in (None, "none", False):
        return None
    if isinstance(mode, (int, float)) and not isinstance(mode, bool):
        return torch.tensor([float(mode)])
    # auto
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        print("    ВНИМАНИЕ: в train представлен один класс — pos_weight не применяю.")
        return None
    w = n_neg / n_pos
    print(f"    pos_weight(auto) = N_neg/N_pos = {n_neg}/{n_pos} = {w:.3f}")
    return torch.tensor([w])


# =============================================================================
# Чекпойнты: атомарная запись + ротация последних K (§7)
# =============================================================================
def _atomic_torch_save(state: dict, path: Path) -> None:
    """torch.save во временный файл рядом, затем os.replace — как atomic_write_* (§7).

    Падение посреди записи не оставит битый чекпойнт, который при рестарте сочли бы
    готовым и попытались загрузить."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _ckpt_path(exp_dir: Path, epoch: int) -> Path:
    return exp_dir / f"ckpt_epoch{epoch:04d}.pt"


def _list_ckpts(exp_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in exp_dir.glob("ckpt_epoch*.pt"):
        try:
            out.append((int(p.stem.replace("ckpt_epoch", "")), p))
        except ValueError:
            continue
    return sorted(out)


def _rotate_ckpts(exp_dir: Path, keep_last: int) -> None:
    """Оставить последние keep_last чекпойнтов, старые удалить (§7).

    keep_last ≥ 1 гарантируется вызывающим кодом (клампится в run). Защитно: при
    некорректном keep_last < 1 НИЧЕГО не удаляем — остаться без чекпойнта хуже, чем
    сохранить лишний, ведь это ломает возобновляемость."""
    if keep_last < 1:
        return
    ckpts = _list_ckpts(exp_dir)
    for _epoch, p in ckpts[:-keep_last]:
        try:
            p.unlink()
        except OSError:
            pass


def _latest_ckpt(exp_dir: Path) -> Path | None:
    ckpts = _list_ckpts(exp_dir)
    return ckpts[-1][1] if ckpts else None


# =============================================================================
# Лог метрик: append + дедуп по эпохе при возобновлении (§7)
# =============================================================================
def _truncate_metrics(path: Path, last_epoch: int) -> None:
    """Оставить в metrics.jsonl только строки с epoch ≤ last_epoch (§7).

    При возобновлении с чекпойнта эпохи E строки эпох > E могли быть записаны до
    падения — их убираем, чтобы лог не задваивался и совпадал с чекпойнтом."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if int(rec.get("epoch", -1)) <= last_epoch:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def _append_metric(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# Один проход обучения / оценки
# =============================================================================
def _train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, count = 0.0, 0
    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        total += float(loss.item()) * bs
        count += bs
    return total / max(count, 1)


@torch.no_grad()
def _evaluate(model, loader, device):
    """Собрать скоры/метки по val и посчитать EER (§8). Скор = сырой логит (спуф=1)."""
    model.eval()
    scores, labels = [], []
    for x, y, lengths in loader:
        x, lengths = x.to(device), lengths.to(device)
        logits = model(x, lengths)
        scores.append(logits.detach().cpu().numpy())
        labels.append(y.numpy())
    scores = np.concatenate(scores) if scores else np.array([])
    labels = np.concatenate(labels) if labels else np.array([])
    return compute_eer(scores, labels)


# =============================================================================
# Основная стадия
# =============================================================================
def run(config: Config, spec: dict) -> Path:
    """Прогнать Стадию 2 по конфигу прогона `spec` (разобранный YAML). Вернуть путь
    к лучшим весам (best.pt). Идемпотентна к возобновлению (§7)."""
    paths = config.paths
    rt = config.runtime
    device = _resolve_device(rt.device)
    _set_seed(rt.seed)

    # --- имя эксперимента и его папка (§6.5) ---
    name = spec.get("name") or "unnamed_exp"
    exp_dir = paths.experiments / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.jsonl"

    # --- признак → C и папка кэша; PRE-FLIGHT сверка (§6.6) ---
    feat = spec["feature"]
    extractor = get_extractor(feat["name"], **feat.get("overrides", {}))
    channels = extractor.channels
    cache_dir = extractor.cache_dir(paths)
    _preflight_cache(cache_dir, extractor, channels)

    # --- данные: селекторы → (подвыборка) → загрузчики ---
    data = spec["data"]
    t_fixed = int(data["t_fixed"])
    sub = data.get("subset") or {}
    sub_seed = int(sub.get("seed", rt.seed))

    train_sel = data["train"]
    val_sel = data["val"]
    # Стратификация: из YAML или дефолт; при склейке >1 датасета добавляем dataset_id.
    strat_train = _stratify_for(sub.get("stratify"), train_sel)
    strat_val = _stratify_for(sub.get("stratify"), val_sel)

    train_df = _load_selectors(paths, train_sel, strat_train)
    val_df = _load_selectors(paths, val_sel, strat_val)
    train_df = _maybe_subset(train_df, sub.get("n_train"), sub_seed, strat_train)
    val_df = _maybe_subset(val_df, sub.get("n_val"), sub_seed, strat_val)
    print(f"    train: {len(train_df)} примеров, val: {len(val_df)} примеров "
          f"(признак {extractor.name}, C={channels}, t_fixed={t_fixed})")

    train_loader = _make_loader(train_df, cache_dir, channels, t_fixed,
                                rt.batch_size, rt.num_workers, shuffle=True, random_crop=True)
    val_loader = _make_loader(val_df, cache_dir, channels, t_fixed,
                              rt.batch_size, rt.num_workers, shuffle=False, random_crop=False)

    # --- модель / лосс / оптимизатор ---
    mspec = spec["model"]
    model = get_model(mspec["name"], in_channels=channels, **mspec.get("params", {})).to(device)
    print(f"    модель {mspec['name']}: {model.num_parameters():,} параметров")

    tcfg = spec.get("train", {})
    pos_weight = _pos_weight(train_loader.dataset.labels, tcfg.get("pos_weight", "auto"))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    epochs = int(tcfg.get("epochs", 3))
    ckpt_every = int(tcfg.get("ckpt_every", 1))
    keep_last = max(1, int(tcfg.get("keep_last", 3)))  # ≥1: без чекпойнта нет resume (§7)

    # --- возобновление с последнего чекпойнта (§7) ---
    start_epoch, best_eer = _maybe_resume(exp_dir, model, optimizer, device, metrics_path)

    # --- цикл по эпохам ---
    for epoch in range(start_epoch, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = _evaluate(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        _append_metric(metrics_path, {
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "val_eer": None if np.isnan(val.eer) else round(val.eer, 6),
            "val_threshold": None if np.isnan(val.threshold) else round(val.threshold, 6),
            "lr": lr,
        })
        eer_str = "nan" if np.isnan(val.eer) else f"{val.eer:.4f}"
        print(f"  эпоха {epoch}/{epochs}  loss={train_loss:.4f}  val_EER={eer_str}")

        # лучшие веса по val-EER (NaN не считается улучшением)
        if not np.isnan(val.eer) and val.eer < best_eer:
            best_eer = val.eer
            _atomic_torch_save(_state(epoch, model, optimizer, best_eer, extractor,
                                      mspec["name"], channels, t_fixed),
                               exp_dir / "best.pt")

        # периодический чекпойнт + ротация
        if ckpt_every > 0 and (epoch % ckpt_every == 0 or epoch == epochs):
            _atomic_torch_save(_state(epoch, model, optimizer, best_eer, extractor,
                                      mspec["name"], channels, t_fixed),
                               _ckpt_path(exp_dir, epoch))
            _rotate_ckpts(exp_dir, keep_last)

    # финальные веса (последняя эпоха)
    _atomic_torch_save(_state(epochs, model, optimizer, best_eer, extractor,
                              mspec["name"], channels, t_fixed),
                       exp_dir / "final.pt")
    best_path = exp_dir / "best.pt"
    best_disp = "nan" if np.isnan(best_eer) else f"{best_eer:.4f}"
    print(f"готово. лучший val_EER={best_disp}. веса: {best_path if best_path.exists() else exp_dir / 'final.pt'}")
    return best_path if best_path.exists() else exp_dir / "final.pt"


def _stratify_for(yaml_stratify, selectors: list[dict]) -> tuple[str, ...]:
    """Колонки стратификации подвыборки. Из YAML или дефолт (§6.4); при склейке
    нескольких датасетов обязательно добавляем dataset_id (§6.4)."""
    strat = tuple(yaml_stratify) if yaml_stratify else DEFAULT_STRATIFY
    datasets = {s["dataset_id"] for s in selectors}
    if len(datasets) > 1 and "dataset_id" not in strat:
        strat = ("dataset_id", *strat)
    return strat


def _preflight_cache(cache_dir: Path, extractor, channels: int) -> None:
    """Проверить, что кэш признаков существует и собран этим же конфигом (§6.6).

    Дешёвый аналог валидатора схемы (§6.1): ловим «нет кэша» и «C кэша ≠ C
    признака» ДО долгого обучения. C, объявленный экстрактором, и есть C, который
    получит модель, — так что сверяем on-disk `_spec.json` как независимую страховку
    (папка реально построена ожидаемым признаком, а не осталась от другого)."""
    spec_path = cache_dir / "_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"нет кэша признаков {cache_dir} (_spec.json отсутствует) — сначала "
            f"Стадия 1: python -m src.data.features <dataset(s)> -f {extractor.name}"
        )
    doc = json.loads(spec_path.read_text(encoding="utf-8"))
    cached_c = doc.get("channels")
    if cached_c != channels:
        raise ValueError(
            f"C кэша ({cached_c}) ≠ C признака {extractor.name} ({channels}) — "
            "кэш собран другим конфигом признаков (§6.6). Проверьте feature в YAML."
        )


def _state(epoch, model, optimizer, best_eer, extractor, model_name, channels, t_fixed) -> dict:
    """Содержимое чекпойнта (§7): веса + состояние оптимизатора + метаданные прогона."""
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_eer": best_eer,
        "meta": {
            "model_name": model_name,
            "feature": extractor.name,
            "feature_hash": extractor.cache_hash(),
            "in_channels": channels,
            "t_fixed": t_fixed,
        },
    }


def _maybe_resume(exp_dir, model, optimizer, device, metrics_path) -> tuple[int, float]:
    """Подхватить последний чекпойнт, если есть (§7). Вернуть (start_epoch, best_eer)."""
    ckpt = _latest_ckpt(exp_dir)
    if ckpt is None:
        return 1, float("inf")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    done = int(state["epoch"])
    best_eer = float(state.get("best_eer", float("inf")))
    _truncate_metrics(metrics_path, done)  # лог согласуем с чекпойнтом
    print(f"    возобновление с чекпойнта {ckpt.name}: продолжаю с эпохи {done + 1} "
          f"(лучший val_EER={best_eer:.4f})" if best_eer != float("inf")
          else f"    возобновление с {ckpt.name}: продолжаю с эпохи {done + 1}")
    return done + 1, best_eer


# =============================================================================
# Тонкий раннер стадии (python -m src.train --config ...) — не глобальный
# диспетчер (§4 его отвергает), а независимый запуск одной стадии, как Стадии 0/1.
# =============================================================================
def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:  # noqa: F841
        raise SystemExit(
            "нужен PyYAML для чтения конфигов Стадии 2: pip install pyyaml"
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_runtime_overrides(base: Runtime, spec_runtime: dict | None) -> Runtime:
    """Секция runtime в YAML переопределяет дефолты config.py (§2, §3)."""
    if not spec_runtime:
        return base
    allowed = {"device", "batch_size", "num_workers", "seed"}
    overrides = {k: v for k, v in spec_runtime.items() if k in allowed}
    return replace(base, **overrides)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description="Стадия 2 (обучение): кэш признаков → веса модели.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="experiments/<exp>.yaml — описание прогона")
    parser.add_argument("--root", type=Path, default=None,
                        help="корень проекта (по умолчанию — автоопределение)")
    parser.add_argument("--list", action="store_true",
                        help="показать доступные признаки и модели, выйти")
    args = parser.parse_args(argv)

    if args.list:
        print("Признаки:", ", ".join(available_extractors()))
        print("Модели:  ", ", ".join(available_models()))
        return 0

    if args.config is None:
        parser.error("укажите --config experiments/<exp>.yaml (или --list)")

    spec = _load_yaml(args.config)
    base = default_config() if not args.root else Config(paths=Paths(root=args.root))
    runtime = _apply_runtime_overrides(base.runtime, spec.get("runtime"))
    config = Config(paths=base.paths, runtime=runtime)

    print(f"эксперимент: {spec.get('name', '(без имени)')}  "
          f"device={runtime.device}  batch={runtime.batch_size}  seed={runtime.seed}")
    run(config, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())