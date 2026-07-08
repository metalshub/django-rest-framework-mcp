"""Advertise a ViewSet's list filtering, ordering and pagination as an MCP ``query`` input.

A normal DRF ``list`` endpoint is narrowed with a query string (``?ordering=``, filterset fields,
``?page=``). MCP tool calls don't carry a query string, so for ``list`` tools we advertise an optional
``query`` object mirroring what the same endpoint accepts over HTTP, derived from the ViewSet's own
configuration:

- **ordering** -- when :class:`rest_framework.filters.OrderingFilter` is active and ``ordering_fields``
  is a concrete list (not ``"__all__"``).
- **filters** -- when :class:`django_filters.rest_framework.DjangoFilterBackend` is active and the
  ViewSet declares a ``filterset_class`` or ``filterset_fields``. ``django-filter`` is an optional
  dependency: if it is not installed, filter fields are simply not advertised.
- **pagination** -- ``page`` / ``page_size`` when the ViewSet has a ``pagination_class``.

The builders are pure functions of the ViewSet class (no request state), so the schema can be generated
and tested independently of the endpoint. :meth:`MCPView.execute_tool` writes an incoming ``query`` back
onto the request's query string and lets the ViewSet's real filter backends run, so filtering, ordering
and pagination behave exactly as they do over HTTP.
"""

import logging
from typing import Any, Dict, Optional, Type

from rest_framework.filters import OrderingFilter
from rest_framework.viewsets import GenericViewSet

logger = logging.getLogger(__name__)

try:
    from django_filters.rest_framework import DjangoFilterBackend, FilterSet
    from django_filters.rest_framework import filters as df_filters
except ImportError:  # pragma: no cover - exercised only when django-filter is absent
    DjangoFilterBackend = None
    FilterSet = None
    df_filters = None


def build_list_query_schema(
    viewset_class: Type[GenericViewSet],
) -> Optional[Dict[str, Any]]:
    """Build the optional ``query`` object schema for a ``list`` tool.

    Returns ``None`` when the ViewSet exposes nothing to order, filter or paginate by, so the input is
    only advertised where it actually does something.
    """
    properties: Dict[str, Any] = {}

    ordering = _ordering_schema(viewset_class)
    if ordering:
        properties["ordering"] = ordering

    properties.update(_filter_schema(viewset_class))

    if getattr(viewset_class, "pagination_class", None):
        properties["page"] = {
            "type": "integer",
            "description": "1-based page number to fetch.",
        }
        properties["page_size"] = {
            "type": "integer",
            "description": "Number of results per page.",
        }

    if not properties:
        return None

    return {
        "type": "object",
        "description": (
            "Optional filtering, ordering and pagination for this list, applied exactly like the web "
            "API's query string. Omit it to get the first page in the default order."
        ),
        "properties": properties,
    }


def _has_backend(viewset_class: Type[GenericViewSet], backend: type) -> bool:
    """Whether ``backend`` (or a subclass) is among the ViewSet's configured ``filter_backends``."""
    backends = getattr(viewset_class, "filter_backends", None) or []
    return any(isinstance(b, type) and issubclass(b, backend) for b in backends)


def _ordering_schema(viewset_class: Type[GenericViewSet]) -> Optional[Dict[str, Any]]:
    """Enum of the ViewSet's orderable fields, each in ascending and ``-`` descending form."""
    if not _has_backend(viewset_class, OrderingFilter):
        return None
    fields = getattr(viewset_class, "ordering_fields", None)
    if not fields or fields == "__all__":
        return None

    options = []
    for field in fields:
        for value in (field, f"-{field}"):
            if value not in options:
                options.append(value)

    return {
        "type": "string",
        "enum": options,
        "description": (
            "Field to sort by. Prefix with '-' for descending order -- e.g. '-created_date' returns the "
            "most recent first."
        ),
    }


def _resolve_filterset_class(viewset_class: Type[GenericViewSet]):
    """The ViewSet's ``filterset_class``, or one built from ``filterset_fields`` as the backend would.

    Returns ``None`` when neither is configured (or when a ``filterset_fields`` model can't be
    determined from the ViewSet's ``queryset`` class attribute).
    """
    filterset_class = getattr(viewset_class, "filterset_class", None)
    if filterset_class is not None:
        return filterset_class

    filterset_fields = getattr(viewset_class, "filterset_fields", None)
    if not filterset_fields:
        return None
    queryset = getattr(viewset_class, "queryset", None)
    model = getattr(queryset, "model", None)
    if model is None:
        return None
    # Mirror DjangoFilterBackend's auto-generated FilterSet from filterset_fields.
    meta = type("Meta", (), {"model": model, "fields": filterset_fields})
    return type("AutoFilterSet", (FilterSet,), {"Meta": meta})


def _filter_schema(viewset_class: Type[GenericViewSet]) -> Dict[str, Any]:
    """Map the ViewSet's filterset fields to JSON-schema properties (best effort)."""
    if DjangoFilterBackend is None or not _has_backend(
        viewset_class, DjangoFilterBackend
    ):
        return {}
    filterset_class = _resolve_filterset_class(viewset_class)
    if filterset_class is None:
        return {}

    properties: Dict[str, Any] = {}
    for name, filter_ in filterset_class.base_filters.items():
        try:
            properties[name] = _filter_field_schema(name, filter_)
        except Exception as exc:  # noqa: BLE001 - one odd filter must not break tool discovery
            logger.warning(
                "Skipping MCP filter %s on %s: %s",
                name,
                filterset_class.__name__,
                exc,
            )
    return properties


def _filter_field_schema(name: str, filter_: Any) -> Dict[str, Any]:
    """Best-effort JSON schema for a single django-filters ``Filter``.

    Subclass order matters: the multiple/model variants must be checked before their bases.
    """
    extra = getattr(filter_, "extra", {}) or {}
    choices = extra.get("choices")
    choices = (
        [str(choice[0]) for choice in choices]
        if choices and not callable(choices)
        else None
    )
    to_field = extra.get("to_field_name")
    model_value_is_id = to_field in (None, "pk", "id")

    schema: Dict[str, Any]
    if isinstance(filter_, df_filters.BooleanFilter):
        schema = {"type": "boolean"}
    elif isinstance(filter_, df_filters.NumberFilter):
        schema = {"type": "number"}
    elif isinstance(filter_, df_filters.ModelMultipleChoiceFilter):
        schema = {
            "type": "array",
            "items": {"type": "integer" if model_value_is_id else "string"},
        }
    elif isinstance(filter_, df_filters.ModelChoiceFilter):
        schema = {"type": "integer" if model_value_is_id else "string"}
    elif isinstance(filter_, df_filters.MultipleChoiceFilter):
        item: Dict[str, Any] = {"type": "string"}
        if choices:
            item["enum"] = choices
        schema = {"type": "array", "items": item}
    elif isinstance(filter_, df_filters.ChoiceFilter):
        schema = {"type": "string"}
        if choices:
            schema["enum"] = choices
    else:
        schema = {"type": "string"}

    if filter_.label:
        schema["description"] = str(filter_.label)
    elif isinstance(
        filter_, (df_filters.ModelChoiceFilter, df_filters.ModelMultipleChoiceFilter)
    ):
        schema["description"] = f"Filter by {name} id."
    return schema
