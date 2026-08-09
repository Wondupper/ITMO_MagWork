"""Стадия 3 — оценка: веса + кэш признаков → скоры и метрики (§4, §6.5, §8).

Стадия отвечает на единственный вопрос работы: насколько хорошо метод отличает
синтетическую речь от подлинной, выраженное числом (критерий №2). Всё остальное —
графики, таблицы, разбивки — строится ноутбуком из артефактов этой стадии, ничего
не пересчитывая (§6).

Что делает `run(config, spec)`:
  1. резолвит веса (`best.pt` по умолчанию, §6.5) и восстанавливает модель ПО
     МЕТАДАННЫМ ЧЕКПОЙНТА — имя, гиперпараметры, C, t_fixed берутся оттуда, а не
     из YAML: так оценка гарантированно идёт той же архитектурой и тем же входом,
     что были при обучении;
  2. PRE-FLIGHT (§6.6), строже, чем на Стадии 2: сверяются C кэша, C модели и
     ХЭШ КОНФИГА ПРИЗНАКА из чекпойнта. Последнее ловит случай «C совпал, а
     параметры признака другие», который одной сверкой C не отличить;
  3. читает тестовые манифесты по селекторам (§3), опц. берёт подвыборку (§6.4);
  4. лениво прогоняет модель по кэшу, скор = сырой логит (§8, `utils.inference`);
  5. ВОЗОБНОВЛЯЕМ (§7): скоры пишутся шардами в `scores_parts/`, при рестарте уже
     посчитанные ключи (dataset_id, utt_id) пропускаются; в конце шарды
     склеиваются в `scores.parquet` и удаляются;
  6. считает метрики ПО КАЖДОМУ ДАТАСЕТУ ОТДЕЛЬНО: EER + бутстрэп-CI (§6.4),
     матрицу ошибок, разбивку EER по атакам (§8).

Почему нет общего EER по смеси датасетов: он смешал бы разные домены и разные
классовые пропорции в одно число, которое ничего не измеряет. Кросс-датасетная
постановка (§8) — это как раз сравнение отдельных чисел между собой.

Артефакты — в experiments/<name>/ (§6.5):
    scores.parquet     — dataset_id, utt_id, split, label, attack_type, score
    eval_metrics.json  — агрегаты: EER, CI, пороги, матрицы ошибок, EER по атакам

Запуск (модуль пакета, из корня проекта — как Стадии 0/1/2, §4):
    python -m src.evaluate --config experiments/smoke_lfcc_statpool.yaml
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config, Paths, Runtime, default_config
from .data.dataset import FeatureCacheDataset, collate
from .data.features import available_extractors, get_extractor
from .data.selectors import load_selectors, stratify_for
from .data.subsets import make_subset
from .models import available_models, get_model
from .utils.inference import iter_batch_scores
from .utils.io import atomic_write_parquet
from .utils.metrics import Confusion, bootstrap_eer, compute_eer, confusion_at

#: Колонки выгрузки скоров (§6.5). Порядок фиксирован — на него смотрит viz.
SCORE_COLUMNS = ["dataset_id", "utt_id", "split", "label", "attack_type", "score"]

#: Конвенция скора — записывается в eval_metrics.json, чтобы число нельзя было
#: прочитать наоборот. ВНИМАНИЕ: в официальных протоколах ASVspoof CM-скор
#: ориентирован ОБРАТНО (больше ⇒ bonafide); при сравнении с чужими скор-файлами
#: знак надо инвертировать (§8).
SCORE_CONVENTION = "сырой логит модели; больше ⇒ спуф (label=1, §6.1)"

#: Имя подпапки с промежуточными шардами скоров (механизм возобновления, §7).
_PARTS_DIR = "scores_parts"


# =============================================================================
# Веса и модель
# =============================================================================
def _resolve_weights(exp_dir: Path, which: str) -> Path:
    """`best` / `final` → файл в папке эксперимента; иначе — путь как есть (§6.5)."""
    path = exp_dir / f"{which}.pt" if which in ("best", "final") else Path(which)
    if not path.exists():
        raise FileNotFoundError(
            f"нет весов {path} — сначала Стадия 2: python -m src.train --config <...>"
        )
    return path


def _build_model(meta: dict, state: dict, channels: int, device: torch.device):
    """Восстановить модель по метаданным чекпойнта и загрузить веса.

    Гиперпараметры берутся из `meta["model_params"]`. У чекпойнтов, записанных до
    появления этого поля, его нет — тогда модель собирается на дефолтах класса, и
    несовпадение с конфигом обучения проявится ошибкой `load_state_dict`. Это
    честнее, чем молча подставить YAML: веса и конфиг могут быть от разных прогонов.
    """
    params = meta.get("model_params")
    if params is None:
        print("    ВНИМАНИЕ: в чекпойнте нет model_params (старый формат) — "
              "модель собирается на дефолтных гиперпараметрах.")
        params = {}
    model = get_model(meta["model_name"], in_channels=channels, **params)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def _preflight(cache_dir: Path, extractor, meta: dict) -> None:
    """Сверить кэш, признак и чекпойнт ДО инференса (§6.6).

    Три независимые сверки: (1) кэш вообще существует и самоописателен; (2) C кэша
    равен C признака из YAML; (3) хэш конфига признака, записанный при обучении,
    совпадает с текущим — то есть модель оценивается на ТЕХ ЖЕ признаках, на
    которых училась. Третья проверка строже сверки одного C и есть только здесь:
    на Стадии 2 сравнивать было не с чем.
    """
    spec_path = cache_dir / "_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"нет кэша признаков {cache_dir} (_spec.json отсутствует) — сначала "
            f"Стадия 1: python -m src.data.features <dataset(s)> -f {extractor.name}"
        )
    doc = json.loads(spec_path.read_text(encoding="utf-8"))
    if doc.get("channels") != extractor.channels:
        raise ValueError(
            f"C кэша ({doc.get('channels')}) ≠ C признака {extractor.name} "
            f"({extractor.channels}) — кэш собран другим конфигом признаков (§6.6)."
        )
    trained_hash = meta.get("feature_hash")
    if trained_hash and trained_hash != extractor.cache_hash():
        raise ValueError(
            f"модель обучена на признаке с хэшем {trained_hash}, а в конфиге "
            f"{extractor.name} даёт {extractor.cache_hash()} — оценка шла бы на ДРУГИХ "
            "признаках. Приведите секцию feature в YAML к конфигу обучения (§6.6)."
        )
    if meta.get("in_channels") not in (None, extractor.channels):
        raise ValueError(
            f"модель ожидает C={meta['in_channels']}, признак даёт "
            f"C={extractor.channels} (§6.6)."
        )


# =============================================================================
# Возобновляемость: шарды скоров (§7)
# =============================================================================
def _fingerprint(weights: Path, meta: dict, selectors: list[dict],
                 n_test, seed: int, window: str) -> dict:
    """Отпечаток условий прогона. Меняются условия — прежние шарды невалидны.

    «Пропустить готовое» безопасно только пока готовое посчитано ТЕМ ЖЕ. Без
    отпечатка смена весов или тестового набора при существующих шардах молча
    смешала бы скоры двух разных прогонов в один файл.
    """
    return {
        "weights": weights.name,
        "epoch": meta.get("epoch"),
        "model_name": meta.get("model_name"),
        "feature_hash": meta.get("feature_hash"),
        "t_fixed": meta.get("t_fixed"),
        "window": window,
        "selectors": sorted(f"{s['dataset_id']}:{s['split']}" for s in selectors),
        "n_test": n_test,
        "seed": seed,
    }


def _resume(parts_dir: Path, fingerprint: dict) -> tuple[set[tuple[str, str]], int]:
    """Вернуть уже посчитанные ключи и номер следующего шарда (§7).

    Несовпадение отпечатка — не ошибка, а нормальная ситуация «перезапустили с
    другими весами»: старые шарды удаляются с предупреждением.
    """
    state_path = parts_dir / "_state.json"
    if parts_dir.exists() and state_path.exists():
        try:
            known = json.loads(state_path.read_text(encoding="utf-8"))
        except ValueError:
            known = None
        if known != fingerprint:
            print("    ВНИМАНИЕ: условия прогона изменились — прежние шарды скоров "
                  "удалены, оценка считается заново.")
            shutil.rmtree(parts_dir)
    elif parts_dir.exists():
        shutil.rmtree(parts_dir)  # шарды без паспорта — доверять нечему

    parts_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True),
                          encoding="utf-8")

    done: set[tuple[str, str]] = set()
    next_idx = 0
    for p in sorted(parts_dir.glob("part_*.parquet")):
        df = pd.read_parquet(p, columns=["dataset_id", "utt_id"])
        done.update(zip(df["dataset_id"].astype(str), df["utt_id"].astype(str)))
        next_idx = max(next_idx, int(p.stem.split("_")[1]) + 1)
    if done:
        print(f"    возобновление: {len(done)} примеров уже оценены — пропускаю.")
    return done, next_idx


def _score_remaining(model, ds: FeatureCacheDataset, loader: DataLoader,
                     device: torch.device, parts_dir: Path, next_idx: int,
                     shard_rows: int) -> int:
    """Прогнать модель и писать скоры шардами. Вернуть число оценённых примеров.

    Связь «скор ↔ ключ» держится на порядке: загрузчик собран с shuffle=False и
    drop_last=False, поэтому i-й скор соответствует i-му примеру Dataset, а его
    ключ лежит в `ds.dataset_ids[i]` / `ds.utt_ids[i]` (§ inference).
    """
    buf: list[np.ndarray] = []
    pos = 0        # индекс начала буфера в порядке Dataset
    scored = 0

    def flush(start: int, chunk: np.ndarray, idx: int) -> None:
        end = start + len(chunk)
        atomic_write_parquet(
            pd.DataFrame({
                "dataset_id": ds.dataset_ids[start:end],
                "utt_id": ds.utt_ids[start:end],
                "score": chunk.astype(np.float32),
            }),
            parts_dir / f"part_{idx:05d}.parquet",
        )

    total = len(ds)
    for scores, _labels in iter_batch_scores(model, loader, device):
        buf.append(scores)
        scored += len(scores)
        if sum(len(b) for b in buf) >= shard_rows:
            chunk = np.concatenate(buf)
            flush(pos, chunk, next_idx)
            pos += len(chunk)
            next_idx += 1
            buf = []
            print(f"    … {pos}/{total}", flush=True)
    if buf:
        flush(pos, np.concatenate(buf), next_idx)
    return scored


def _collect_scores(parts_dir: Path) -> pd.DataFrame:
    """Склеить шарды в один кадр (dataset_id, utt_id, score)."""
    parts = sorted(parts_dir.glob("part_*.parquet"))
    if not parts:
        return pd.DataFrame({"dataset_id": [], "utt_id": [], "score": []})
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return df.drop_duplicates(subset=["dataset_id", "utt_id"], keep="last")


# =============================================================================
# Метрики (§8)
# =============================================================================
def _rates(c: Confusion) -> dict:
    """FPR (ложная тревога на подлинных) и FNR (пропуск спуфа) — как в compute_eer."""
    neg, pos = c.tn + c.fp, c.fn + c.tp
    return {
        "confusion": {"tn": c.tn, "fp": c.fp, "fn": c.fn, "tp": c.tp},
        "fpr": (c.fp / neg) if neg else float("nan"),
        "fnr": (c.fn / pos) if pos else float("nan"),
    }


def _eer_by_attack(df: pd.DataFrame) -> dict:
    """EER по каждой атаке отдельно: все подлинные датасета против спуфа этой атаки.

    Уточнение контракта §6.1 (иначе правило читается буквально и ломается): «строки
    с NULL в attack_type пропускать» относится к СПУФ-строкам с неизвестной атакой.
    У подлинных записей attack_type всегда NULL по конвенции, и они обязаны
    участвовать — без отрицательного класса EER не определён.

    Смысл разбивки (§8): агрегированный EER скрывает, на каких атаках метод
    проседает; именно эта таблица показывает слабые места, а не одно число.
    """
    bona = df[df["label"] == 0]
    spoof = df[(df["label"] == 1) & df["attack_type"].notna()]
    out: dict[str, dict] = {}
    for attack, g in spoof.groupby("attack_type", observed=True):
        scores = np.concatenate([bona["score"].to_numpy(), g["score"].to_numpy()])
        labels = np.concatenate([np.zeros(len(bona), int), np.ones(len(g), int)])
        e = compute_eer(scores, labels)
        out[str(attack)] = {
            "n_spoof": int(len(g)),
            "eer": None if np.isnan(e.eer) else round(float(e.eer), 6),
        }
    return dict(sorted(out.items()))


def _dataset_metrics(df: pd.DataFrame, *, n_boot: int, boot_seed: int, ci: float,
                     val_threshold: float | None) -> dict:
    """Метрики одного датасета: EER + CI, матрицы ошибок, разбивка по атакам."""
    scores = df["score"].to_numpy()
    labels = df["label"].to_numpy().astype(int)
    n_bona = int((labels == 0).sum())
    n_spoof = int((labels == 1).sum())

    eer = compute_eer(scores, labels)
    boot = bootstrap_eer(scores, labels, n_boot=n_boot, seed=boot_seed, ci=ci)

    out: dict = {
        "n": int(len(df)),
        "n_bonafide": n_bona,
        "n_spoof": n_spoof,
        "n_spoof_without_attack_type": int(
            ((labels == 1) & df["attack_type"].isna().to_numpy()).sum()
        ),
        "eer": None if np.isnan(eer.eer) else round(float(eer.eer), 6),
        "eer_threshold": None if np.isnan(eer.threshold) else float(eer.threshold),
        "eer_ci": None if np.isnan(boot.lo) else [round(boot.lo, 6), round(boot.hi, 6)],
        "eer_ci_level": ci,
        "bootstrap_n": boot.n_boot,
    }
    if n_bona and n_spoof:
        # Порог подобран на САМОМ тесте: матрица иллюстрирует, как распределены
        # ошибки в точке равного риска, но не является оценкой обобщения (§6.5).
        out["at_eer_threshold"] = _rates(confusion_at(scores, labels, eer.threshold))
    if val_threshold is not None:
        # Честная рабочая точка: порог выбран на валидации, до теста.
        out["at_val_threshold"] = {
            "threshold": float(val_threshold),
            **_rates(confusion_at(scores, labels, val_threshold)),
        }
    out["by_attack"] = _eer_by_attack(df)
    return out


def _val_threshold(metrics_path: Path, epoch) -> float | None:
    """Val-порог той эпохи, чьи веса оцениваются (из metrics.jsonl Стадии 2, §6.5)."""
    if epoch is None or not metrics_path.exists():
        return None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if int(rec.get("epoch", -1)) == int(epoch):
            thr = rec.get("val_threshold")
            return None if thr is None else float(thr)
    return None


def _write_json(doc: dict, path: Path) -> Path:
    """Атомарная запись JSON — тот же приём, что у остальных артефактов (§7)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


# =============================================================================
# Основная стадия
# =============================================================================
def run(config: Config, spec: dict, *, weights: str | None = None,
        allow_missing: bool = False) -> Path:
    """Прогнать Стадию 3 по конфигу прогона `spec`. Вернуть путь к scores.parquet."""
    paths = config.paths
    rt = config.runtime
    device = torch.device("cpu") if rt.device == "cuda" and not torch.cuda.is_available() \
        else torch.device(rt.device)

    name = spec.get("name") or "unnamed_exp"
    exp_dir = paths.experiments / name
    if not exp_dir.exists():
        raise FileNotFoundError(f"нет папки эксперимента {exp_dir} — сначала Стадия 2.")

    ecfg = spec.get("eval") or {}
    window = str(ecfg.get("window", "fixed"))
    if window != "fixed":
        raise NotImplementedError(
            f"eval.window='{window}' не реализован. Сейчас поддержан только 'fixed' — "
            "окно t_fixed кадров с начала записи, ровно как на валидации Стадии 2. "
            "'full' (вся запись) и 'sliding' (скользящие окна с агрегацией) — "
            "содержательные варианты метода, а не настройки инфраструктуры."
        )

    # --- веса и метаданные обучения ---
    wpath = _resolve_weights(exp_dir, weights or str(ecfg.get("weights", "best")))
    state = torch.load(wpath, map_location=device, weights_only=False)
    meta = dict(state.get("meta", {}))
    meta["epoch"] = state.get("epoch")
    t_fixed = int(meta["t_fixed"])

    # --- признак и pre-flight (§6.6) ---
    feat = spec["feature"]
    extractor = get_extractor(feat["name"], **feat.get("overrides", {}))
    channels = extractor.channels
    cache_dir = extractor.cache_dir(paths)
    _preflight(cache_dir, extractor, meta)

    model = _build_model(meta, state, channels, device)
    print(f"    веса: {wpath.name} (эпоха {meta['epoch']}), модель "
          f"{meta.get('model_name')}, признак {extractor.name} C={channels}, "
          f"t_fixed={t_fixed}, окно={window}")

    # --- тестовый манифест ---
    data = spec["data"]
    if not data.get("test"):
        raise KeyError(
            "в конфиге нет data.test — добавьте список селекторов, напр.\n"
            "  test:\n    - {dataset_id: asvspoof2021_la, split: test}"
        )
    sub = data.get("subset") or {}
    n_test = sub.get("n_test")
    sub_seed = int(sub.get("seed", rt.seed))
    strat = stratify_for(sub.get("stratify"), data["test"])

    test_df = load_selectors(paths, data["test"], ("attack_type", *strat))
    test_df = make_subset(test_df, n_test, seed=sub_seed, stratify=strat)
    n_requested = len(test_df)
    print(f"    тест: {n_requested} примеров из {len(data['test'])} селектор(ов)")
    if n_test is not None:
        print("    ВНИМАНИЕ: оценка на ПОДВЫБОРКЕ — для отладки, не для итоговых "
              "чисел (§6.4).")

    # --- инференс с возобновлением (§7) ---
    parts_dir = exp_dir / _PARTS_DIR
    fingerprint = _fingerprint(wpath, meta, data["test"], n_test, sub_seed, window)
    done, next_idx = _resume(parts_dir, fingerprint)

    todo = test_df
    if done:
        keys = list(zip(test_df["dataset_id"].astype(str), test_df["utt_id"].astype(str)))
        todo = test_df.loc[[k not in done for k in keys]].reset_index(drop=True)

    if len(todo):
        try:
            ds = FeatureCacheDataset(todo, cache_dir, channels, t_fixed, random_crop=False)
        except RuntimeError:
            # Ни одного .npy на оставшиеся ключи. Не падаем: считать нечего, а
            # достаточность покрытия проверяется ниже единым правилом.
            print("    ВНИМАНИЕ: для оставшихся примеров нет признаков в кэше.")
            ds = None
        if ds is not None:
            loader = DataLoader(ds, batch_size=rt.batch_size, shuffle=False,
                                num_workers=rt.num_workers, collate_fn=collate,
                                pin_memory=False, drop_last=False)
            _score_remaining(model, ds, loader, device, parts_dir, next_idx,
                             int(ecfg.get("shard_rows", 20000)))

    # --- склейка шардов + метаданные манифеста → scores.parquet (§6.5) ---
    scores = _collect_scores(parts_dir)
    out = test_df.merge(scores, on=["dataset_id", "utt_id"], how="inner")
    out = out[SCORE_COLUMNS].sort_values(["dataset_id", "utt_id"], kind="stable")
    scores_path = atomic_write_parquet(out.reset_index(drop=True),
                                       exp_dir / "scores.parquet")
    shutil.rmtree(parts_dir, ignore_errors=True)

    # --- покрытие: молчаливая потеря примеров меняет знаменатель метрики ---
    n_scored = len(out)
    ratio = n_scored / n_requested if n_requested else 0.0
    min_cov = float(ecfg.get("min_coverage", 0.99))
    if n_scored < n_requested:
        print(f"    ВНИМАНИЕ: оценено {n_scored}/{n_requested} ({ratio:.4f}) — "
              "часть примеров отсутствует в кэше признаков (Стадия 1).")
    if ratio < min_cov and not allow_missing:
        raise RuntimeError(
            f"покрытие {ratio:.4f} < min_coverage {min_cov}: метрика считалась бы по "
            f"неполному тесту. Доизвлеките признаки (python -m src.data.features "
            f"<ds> -f {extractor.name}) либо запустите с --allow-missing, честно "
            "оговорив покрытие в работе."
        )

    # --- метрики по каждому датасету отдельно (§8) ---
    val_thr = _val_threshold(exp_dir / "metrics.jsonl", meta.get("epoch"))
    n_boot = int(ecfg.get("bootstrap", 1000))
    boot_seed = int(ecfg.get("bootstrap_seed", rt.seed))
    ci = float(ecfg.get("ci", 0.95))
    if n_boot:
        print(f"    бутстрэп-CI: {n_boot} повторов на датасет…", flush=True)

    doc = {
        "experiment": name,
        "weights": wpath.name,
        "checkpoint": {k: meta.get(k) for k in
                       ("epoch", "model_name", "model_params", "feature",
                        "feature_hash", "in_channels", "t_fixed")},
        "window": window,
        "score_convention": SCORE_CONVENTION,
        "val_threshold": val_thr,
        "coverage": {"n_requested": int(n_requested), "n_scored": int(n_scored),
                     "ratio": round(ratio, 6)},
        "subset": {"n_test": n_test, "seed": sub_seed} if n_test is not None else None,
        "per_dataset": {},
    }
    for ds_id, g in out.groupby("dataset_id", observed=True):
        doc["per_dataset"][str(ds_id)] = _dataset_metrics(
            g, n_boot=n_boot, boot_seed=boot_seed, ci=ci, val_threshold=val_thr
        )
    metrics_path = _write_json(doc, exp_dir / "eval_metrics.json")

    _print_summary(doc)
    print(f"готово. скоры: {scores_path}\n         метрики: {metrics_path}")
    return scores_path


def _print_summary(doc: dict) -> None:
    """Компактная сводка — сразу видно, есть ли смысл идти в ноутбук."""
    print(f"\n{'датасет':<22}{'N':>8}{'EER':>9}   CI")
    for ds_id, m in doc["per_dataset"].items():
        eer = "nan" if m["eer"] is None else f"{m['eer'] * 100:.2f}%"
        ci = "—" if not m["eer_ci"] else \
            f"[{m['eer_ci'][0] * 100:.2f}%, {m['eer_ci'][1] * 100:.2f}%]"
        print(f"{ds_id:<22}{m['n']:>8}{eer:>9}   {ci}")


# =============================================================================
# Тонкий раннер стадии (python -m src.evaluate --config ...) — как Стадии 0/1/2
# =============================================================================
def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit("нужен PyYAML для чтения конфигов Стадии 3: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_runtime_overrides(base: Runtime, spec_runtime: dict | None) -> Runtime:
    """Секция runtime в YAML переопределяет дефолты config.py (§2, §3)."""
    if not spec_runtime:
        return base
    allowed = {"device", "batch_size", "num_workers", "seed"}
    return replace(base, **{k: v for k, v in spec_runtime.items() if k in allowed})


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Стадия 3 (оценка): веса + кэш признаков → скоры и метрики.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="experiments/<exp>.yaml — тот же конфиг, что у Стадии 2")
    parser.add_argument("--weights", default=None,
                        help="best | final | путь к .pt (по умолчанию — eval.weights)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="не падать при покрытии ниже eval.min_coverage")
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
    base = Config(paths=Paths(root=args.root)) if args.root else default_config()
    config = Config(paths=base.paths,
                    runtime=_apply_runtime_overrides(base.runtime, spec.get("runtime")))

    print(f"оценка: {spec.get('name', '(без имени)')}  device={config.runtime.device}  "
          f"batch={config.runtime.batch_size}")
    run(config, spec, weights=args.weights, allow_missing=args.allow_missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())