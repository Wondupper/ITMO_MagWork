"""Сравнение нескольких прогонов между собой (§8). Импортируется ноутбуком 05.

Зачем отдельный модуль. `results.py` описывает ОДИН эксперимент и читает его
артефакты; здесь вопрос другой — «стало ли лучше», и он по устройству
межэкспериментный: ни одна Стадия 3 такого числа не считает, потому что каждый её
прогон знает только про себя. Разделение модулей повторяет разделение ноутбуков:
04 — предусловие «есть один прогон», 05 — «есть два и больше».

Про правило §6 «ноутбуки ничего не пересчитывают». Оно здесь не нарушено, но
держится не само собой, а проверкой: `align_scores` сверяет EER, пересчитанный из
`scores.parquet`, с числом из `eval_metrics.json` того же прогона и падает при
расхождении. То есть новое (разность и её интервал) считается, а старое —
сверяется, и разъехаться числу в таблице с числом на графике по-прежнему нечем.

Все функции возвращают `pandas.DataFrame` или `matplotlib.Figure`; сохраняет
ноутбук через `viz.style.save_figure`.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import Paths
from ..utils.metrics import compute_eer, paired_bootstrap_eer_delta
from .results import _DET_TICKS, _curve, load_metrics, load_scores

#: Допуск при сверке пересчитанного EER с записанным в JSON. Не ноль, потому что
#: в JSON число округлено до 6 знаков (см. evaluate._dataset_metrics).
_EER_TOL = 1e-6


# =============================================================================
# Сводные таблицы
# =============================================================================
def compare_table(experiments: list[str], paths: Paths | None = None) -> pd.DataFrame:
    """Строка на пару (эксперимент, датасет): N, EER и его CI — всё из JSON.

    Основная таблица работы: столбец EER читается сверху вниз внутри датасета.
    Сравнение датасетов между собой по этой таблице делать нельзя (разные домены и
    разные классовые пропорции, §8) — сравниваются строки одного датасета.

    ВАЖНО про чтение: пересекающиеся CI двух экспериментов НЕ означают, что разницы
    нет. Утверждение о разнице делается по `delta_table`, где интервал построен у
    самой разности.
    """
    rows = []
    for exp in experiments:
        m = load_metrics(exp, paths)
        for ds, d in m["per_dataset"].items():
            ci = d.get("eer_ci")
            rows.append({
                "experiment": exp,
                "dataset_id": ds,
                "N": d["n"],
                "EER, %": None if d["eer"] is None else round(d["eer"] * 100, 2),
                "CI нижн., %": None if not ci else round(ci[0] * 100, 2),
                "CI верхн., %": None if not ci else round(ci[1] * 100, 2),
                "признак": (m.get("checkpoint") or {}).get("feature"),
                "модель": (m.get("checkpoint") or {}).get("model_name"),
            })
    return pd.DataFrame(rows).sort_values(["dataset_id", "experiment"],
                                          kind="stable").reset_index(drop=True)


# =============================================================================
# Выравнивание двух выгрузок
# =============================================================================
def align_scores(exp_a: str, exp_b: str, dataset_id: str,
                 paths: Paths | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Скоры двух прогонов на общих записях одного датасета + метки.

    Возвращает `(scores_a, scores_b, labels)`, где i-й элемент всех трёх массивов
    относится к одной и той же записи. Без этого парный бутстрэп считал бы разность
    между скорами разных файлов.

    Соединение — inner join по (dataset_id, utt_id): это глобальный ключ проекта
    (§3), а не порядок строк, на который полагаться нельзя. Если тестовые наборы
    прогонов совпадают не полностью (например, у одного покрытие кэша было ниже),
    сравнение честно идёт по пересечению, о чём печатается предупреждение — иначе
    разность посчиталась бы на разных знаменателях.

    Здесь же сверка с §6: EER, пересчитанный из выгрузки, сравнивается с числом из
    `eval_metrics.json`. Сверка выполняется только при полном совпадении наборов —
    на пересечении EER законно отличается, и требовать равенства было бы неверно.
    """
    out = []
    for exp in (exp_a, exp_b):
        s = load_scores(exp, paths)
        s = s[s["dataset_id"].astype(str) == str(dataset_id)]
        if s.empty:
            raise ValueError(f"[{exp}] в выгрузке нет датасета '{dataset_id}'")
        out.append(s[["utt_id", "label", "score"]].copy())
    a, b = out

    merged = a.merge(b, on="utt_id", how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError(
            f"у '{exp_a}' и '{exp_b}' нет общих записей в {dataset_id} — "
            "проверьте, что оба прогона оценивались на одном тесте."
        )
    if (merged["label_a"] != merged["label_b"]).any():
        raise ValueError(
            f"метки одних и тех же utt_id разошлись между прогонами в {dataset_id} — "
            "выгрузки построены по разным манифестам, сравнивать нельзя."
        )
    if len(merged) < max(len(a), len(b)):
        print(f"    ВНИМАНИЕ [{dataset_id}]: сравнение по пересечению "
              f"{len(merged)} записей (в '{exp_a}' {len(a)}, в '{exp_b}' {len(b)}).")

    labels = merged["label_a"].to_numpy().astype(int)
    sa = merged["score_a"].to_numpy()
    sb = merged["score_b"].to_numpy()

    # Сверка с числом Стадии 3 (§6): расхождение = выгрузка и метрики от разных прогонов.
    if len(merged) == len(a) == len(b):
        for exp, sc in ((exp_a, sa), (exp_b, sb)):
            stored = (load_metrics(exp, paths)["per_dataset"].get(str(dataset_id)) or {}).get("eer")
            got = compute_eer(sc, labels).eer
            if stored is not None and not np.isnan(got) and abs(got - stored) > _EER_TOL:
                raise ValueError(
                    f"[{exp}/{dataset_id}] EER из scores.parquet ({got:.6f}) не совпал с "
                    f"eval_metrics.json ({stored:.6f}) — артефакты от разных прогонов, "
                    "перезапустите Стадию 3."
                )
    return sa, sb, labels


def delta_table(exp_a: str, exp_b: str, *, paths: Paths | None = None,
                n_boot: int = 1000, seed: int = 1337, ci: float = 0.95,
                datasets: list[str] | None = None) -> pd.DataFrame:
    """ΔEER = EER(B) − EER(A) с парным бутстрэп-CI, по каждому общему датасету.

    Это и есть числовое утверждение работы: отрицательная дельта = B лучше, а
    интервал показывает, насколько уверенно. Колонка «значимо» отмечает случай,
    когда интервал целиком лежит по одну сторону нуля.

    Оговорка, которую надо держать в тексте (§8): интервал покрывает конечность
    ТЕСТА, но не случайность обучения. Это сравнение двух конкретных обученных
    моделей, а не двух методов вообще.
    """
    ma, mb = load_metrics(exp_a, paths), load_metrics(exp_b, paths)
    shared = datasets or [d for d in ma["per_dataset"] if d in mb["per_dataset"]]
    if not shared:
        raise ValueError(f"у '{exp_a}' и '{exp_b}' нет общих датасетов в метриках.")

    rows = []
    for ds in shared:
        sa, sb, labels = align_scores(exp_a, exp_b, ds, paths)
        d = paired_bootstrap_eer_delta(sa, sb, labels, n_boot=n_boot, seed=seed, ci=ci)
        signif = None if np.isnan(d.lo) else bool(d.lo * d.hi > 0)
        rows.append({
            "dataset_id": ds,
            "N": len(labels),
            f"EER {exp_a}, %": round(d.eer_a * 100, 2),
            f"EER {exp_b}, %": round(d.eer_b * 100, 2),
            "ΔEER, п.п.": round(d.delta * 100, 2),
            "CI Δ нижн.": None if np.isnan(d.lo) else round(d.lo * 100, 2),
            "CI Δ верхн.": None if np.isnan(d.hi) else round(d.hi * 100, 2),
            "значимо": signif,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Графики сравнения
# =============================================================================
def plot_eer_compare(experiments: list[str], paths: Paths | None = None,
                     title: str = "EER по датасетам") -> plt.Figure:
    """Столбики EER с усами CI: группы — датасеты, цвет — эксперимент.

    Обзорная картинка всей серии экспериментов на одном листе. Усы — тот же
    интервал, что в таблице; он про конечность теста, поэтому по пересечению усов
    выводов о значимости различия делать нельзя (для этого `delta_table`).
    """
    metrics = {e: load_metrics(e, paths) for e in experiments}
    datasets = list(dict.fromkeys(
        ds for m in metrics.values() for ds in m["per_dataset"]
    ))
    x = np.arange(len(datasets), dtype=float)
    width = min(0.8 / max(len(experiments), 1), 0.35)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(datasets) + 2.0), 4.2))
    for k, exp in enumerate(experiments):
        vals, lo, hi = [], [], []
        for ds in datasets:
            d = metrics[exp]["per_dataset"].get(ds) or {}
            e = d.get("eer")
            c = d.get("eer_ci")
            vals.append(np.nan if e is None else e * 100)
            lo.append(0.0 if not c or e is None else max(e * 100 - c[0] * 100, 0.0))
            hi.append(0.0 if not c or e is None else max(c[1] * 100 - e * 100, 0.0))
        off = (k - (len(experiments) - 1) / 2) * width
        ax.bar(x + off, vals, width=width * 0.92, label=exp,
               yerr=[lo, hi], capsize=3, error_kw={"lw": 1.0, "ecolor": "0.35"})

    ax.set_xticks(x, datasets, rotation=15 if len(datasets) > 3 else 0)
    ax.set_ylabel("EER, %")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_det_compare(experiments: list[str], dataset_id: str,
                     paths: Paths | None = None) -> plt.Figure:
    """DET-кривые нескольких прогонов на ОДНОМ датасете, на общих осях.

    Дополняет таблицу тем, чего в одном числе нет: EER — это одна точка кривой, и
    два метода с равным EER могут по-разному вести себя в области малых ошибок,
    где как раз и работает реальная система (§8). Кривая ниже и левее — лучше.

    Точки на диагонали — EER из `eval_metrics.json` соответствующего прогона.
    """
    fig, ax = plt.subplots()
    lim = norm.ppf([_DET_TICKS.min(), _DET_TICKS.max()])

    for exp in experiments:
        scores = load_scores(exp, paths)
        g = scores[scores["dataset_id"].astype(str) == str(dataset_id)]
        if g.empty or g["label"].nunique() < 2:
            print(f"    {exp}: в {dataset_id} нет обоих классов — пропуск.")
            continue
        fpr, fnr = _curve(g)
        m = (fpr > 0) & (fnr > 0)
        line, = ax.plot(norm.ppf(fpr[m]), norm.ppf(fnr[m]), label=exp)
        eer = (load_metrics(exp, paths)["per_dataset"].get(str(dataset_id)) or {}).get("eer")
        if eer is not None and eer > 0:
            ax.plot(norm.ppf(eer), norm.ppf(eer), "o", ms=5, color=line.get_color())

    ax.plot(lim, lim, ls=":", lw=0.9, color="0.5", zorder=0)
    for setter, getter in ((ax.set_xticks, ax.set_xticklabels),
                           (ax.set_yticks, ax.set_yticklabels)):
        setter(norm.ppf(_DET_TICKS))
        getter([f"{t * 100:g}" for t in _DET_TICKS])
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Ложная тревога FPR, % (подлинную приняли за спуф)")
    ax.set_ylabel("Пропуск FNR, % (спуф приняли за подлинную)")
    ax.set_title(f"DET: {dataset_id}")
    ax.legend()
    return fig


def plot_group_compare(experiments: list[str], dataset_id: str,
                       column: str = "attack_type",
                       paths: Paths | None = None) -> plt.Figure:
    """EER по разрезу (атака / кодек / вокодер) для нескольких прогонов сразу.

    Отвечает на вопрос, который не виден в средних числах: улучшение равномерное
    или метод просто выучил одно семейство атак, просев на остальных. Берутся
    группы, посчитанные во ВСЕХ сравниваемых прогонах, — иначе столбцы означали бы
    разное.
    """
    metrics = {e: load_metrics(e, paths) for e in experiments}
    blocks = {}
    for exp, m in metrics.items():
        d = (m["per_dataset"].get(str(dataset_id)) or {}).get(f"by_{column}")
        if not d:
            raise ValueError(f"[{exp}] у {dataset_id} нет разбивки по '{column}'.")
        blocks[exp] = d

    names = sorted(set.intersection(*(set(b) for b in blocks.values())))
    if not names:
        raise ValueError(f"у прогонов нет общих значений '{column}' в {dataset_id}.")

    x = np.arange(len(names), dtype=float)
    width = min(0.8 / len(experiments), 0.35)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(names) + 2.0), 4.2))
    for k, exp in enumerate(experiments):
        vals = [np.nan if blocks[exp][n]["eer"] is None else blocks[exp][n]["eer"] * 100
                for n in names]
        ax.bar(x + (k - (len(experiments) - 1) / 2) * width, vals,
               width=width * 0.92, label=exp)

    ax.set_xticks(x, names, rotation=90 if len(names) > 10 else 0)
    ax.set_ylabel("EER, %")
    ax.set_title(f"{dataset_id}: EER по разрезу '{column}'")
    ax.legend()
    return fig
