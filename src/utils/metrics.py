"""Метрики детекции. EER — сквозная метрика проекта (§8).

Одна функция EER, общая для Стадии 2 (val-EER по эпохам) и Стадии 3 (итоговый EER
на тесте). Держать её в одном месте важно принципиально: если val и финал считать
разным кодом, числа окажутся несопоставимы — а сопоставимость и есть критерий №2.

Конвенция скора: БОЛЬШЕ скор ⇒ увереннее «это spoof» (положительный класс = 1,
§6.1). Модель Стадии 2 отдаёт один логит на пример; сырой логит годится как скор
напрямую (монотонная сигмоида порядок не меняет).

Зависимости — только numpy и scikit-learn (обе обязательны, §2); scipy не тянем.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.metrics import roc_curve


class EER(NamedTuple):
    """Результат: сама метрика и порог, на котором FPR = FNR."""
    eer: float
    threshold: float


def compute_eer(scores, labels) -> EER:
    """Equal Error Rate: точка, где доли ложного пропуска и ложной тревоги равны.

    Определения при положительном классе spoof=1:
      • FPR (ложная тревога) — доля bonafide (0), которым поставили скор ≥ порога;
      • FNR (ложный пропуск) — доля spoof (1) со скором < порога.
    EER — значение этих долей в пороге, где кривые FPR(θ) и FNR(θ) пересекаются.
    Порог между узлами ROC находится линейной интерполяцией по точке пересечения
    (FNR − FPR меняет знак) — оценка точнее, чем ближайший узел.

    Вырожденный случай (в батче/подвыборке представлен один класс) EER не
    определён: возвращаем NaN, не роняя прогон. На Стадии 2 это нормально для
    крохотного smoke-val — эпоха логируется с val_eer=NaN, обучение продолжается;
    итоговый EER всё равно берётся на полном тесте (§6.4).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)

    if len(np.unique(labels)) < 2:
        return EER(float("nan"), float("nan"))

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    diff = fnr - fpr  # монотонно убывает от + к − при движении по узлам ROC

    # Узел, ближайший к пересечению; при точном совпадении узла хватает и его.
    i = int(np.nanargmin(np.abs(diff)))
    if diff[i] == 0 or i + 1 >= len(diff):
        return EER(float((fpr[i] + fnr[i]) / 2.0), float(thresholds[i]))

    # Выбираем соседний узел так, чтобы пересечение лежало между i и j.
    j = i + 1 if diff[i] * diff[i + 1] < 0 else i - 1
    if j < 0 or diff[i] * diff[j] > 0:
        return EER(float((fpr[i] + fnr[i]) / 2.0), float(thresholds[i]))

    # Линейная интерполяция точки diff = 0 на отрезке [i, j].
    t = diff[i] / (diff[i] - diff[j])
    eer = float(fpr[i] + t * (fpr[j] - fpr[i]))
    threshold = float(thresholds[i] + t * (thresholds[j] - thresholds[i]))
    return EER(eer, threshold)


class EERInterval(NamedTuple):
    """Точечная оценка EER и её доверительный интервал (§6.4)."""
    eer: float
    lo: float
    hi: float
    n_boot: int


def bootstrap_eer(scores, labels, *, n_boot: int = 1000, seed: int = 1337,
                  ci: float = 0.95) -> EERInterval:
    """Перцентильный бутстрэп-CI для EER, стратифицированный по классу (§6.4).

    Зачем. Одно число EER без интервала не даёт понять, значимо ли отличие двух
    методов: на малом или несбалансированном тесте разброс велик. Интервал — это
    ровно то, что превращает «улучшение» в измеримое утверждение (критерий №2).

    Почему стратифицированно. Пересэмплируются НЕЗАВИСИМО подлинные и спуф-записи,
    с сохранением их исходных количеств. EER определяется парой условных
    распределений скора (при label=0 и при label=1), а соотношение классов в
    тестовом наборе — свойство его дизайна, а не случайная величина. Общий
    ресэмплинг «в кучу» добавил бы к интервалу дисперсию классового баланса,
    которой в реальности нет.

    Что интервал НЕ покрывает (важно оговорить в работе): неопределённость из-за
    конечности обучающей выборки и случайности самого обучения. Это разброс одной
    обученной модели на данном тесте, а не разброс метода.

    Стоимость — n_boot прогонов ROC. Для теста в сотни тысяч примеров это минуты;
    `n_boot=0` отключает вычисление и возвращает NaN-интервал.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    point = compute_eer(scores, labels).eer

    neg = np.flatnonzero(labels == 0)
    pos = np.flatnonzero(labels == 1)
    if n_boot <= 0 or len(neg) == 0 or len(pos) == 0:
        return EERInterval(point, float("nan"), float("nan"), 0)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n_boot)):
        idx = np.concatenate([
            rng.choice(neg, size=len(neg), replace=True),
            rng.choice(pos, size=len(pos), replace=True),
        ])
        e = compute_eer(scores[idx], labels[idx]).eer
        if not np.isnan(e):
            draws.append(e)
    if not draws:
        return EERInterval(point, float("nan"), float("nan"), 0)

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(draws, [alpha, 1.0 - alpha])
    return EERInterval(point, float(lo), float(hi), len(draws))


class Confusion(NamedTuple):
    """Матрица ошибок при фиксированном пороге (спуф = положительный класс)."""
    tn: int
    fp: int
    fn: int
    tp: int


def confusion_at(scores, labels, threshold: float) -> Confusion:
    """Матрица ошибок для решающего правила «скор ≥ порога ⇒ спуф».

    Знак сравнения тот же, что в определении FPR/FNR у `compute_eer`, — иначе
    матрица на EER-пороге не совпала бы с самим EER.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    pred = scores >= threshold
    return Confusion(
        tn=int(((labels == 0) & ~pred).sum()),
        fp=int(((labels == 0) & pred).sum()),
        fn=int(((labels == 1) & ~pred).sum()),
        tp=int(((labels == 1) & pred).sum()),
    )