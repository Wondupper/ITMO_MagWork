"""Пакет моделей: реестр архитектур (§6.6).

Импорт модулей-моделей здесь нужен ради side-effect регистрации: `@register_model`
срабатывает при импорте класса, поэтому `get_model` видит все архитектуры, как
только импортирован пакет `models`. Новая архитектура = новый модуль + строка
импорта ниже (или просто импорт в train.py) — цикл обучения не трогается.
"""
from .model_base import AntispoofModel, available_models, get_model, register_model
from . import statpool  # noqa: F401 — импорт ради регистрации StatPoolMLP

__all__ = ["AntispoofModel", "available_models", "get_model", "register_model"]