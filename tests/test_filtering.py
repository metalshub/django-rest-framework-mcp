"""Unit tests for djangorestframework_mcp.filtering.build_list_query_schema.

End-to-end behaviour (an incoming ``query`` applied through the ViewSet's filter backends) lives in
tests/test_integration.py.
"""

from django.test import TestCase
from rest_framework.filters import OrderingFilter

from djangorestframework_mcp.filtering import build_list_query_schema

from .views import (
    FilterableCustomerViewSet,
    FilterFieldsCustomerViewSet,
    PlainCustomerViewSet,
)


class BuildListQuerySchemaTests(TestCase):
    """The ``query`` schema builder derives ordering/filters/pagination from the ViewSet."""

    def test_returns_none_when_nothing_to_narrow_by(self):
        assert build_list_query_schema(PlainCustomerViewSet) is None

    def test_advertises_ordering_filters_and_pagination(self):
        schema = build_list_query_schema(FilterableCustomerViewSet)

        assert schema["type"] == "object"
        props = schema["properties"]
        # Ordering enum carries both ascending and descending forms.
        assert set(props["ordering"]["enum"]) == {"age", "-age", "name", "-name"}
        # FilterSet fields are typed from the filter class.
        assert props["is_active"]["type"] == "boolean"
        assert props["min_age"]["type"] == "number"
        # Pagination knobs.
        assert props["page"]["type"] == "integer"
        assert props["page_size"]["type"] == "integer"

    def test_ordering_requires_ordering_filter_backend(self):
        class NoOrderingBackend(FilterableCustomerViewSet):
            filter_backends = [
                b
                for b in FilterableCustomerViewSet.filter_backends
                if b is not OrderingFilter
            ]

        schema = build_list_query_schema(NoOrderingBackend)

        assert "ordering" not in schema["properties"]
        assert "is_active" in schema["properties"]

    def test_all_ordering_is_not_enumerated(self):
        class AllOrdering(FilterableCustomerViewSet):
            ordering_fields = "__all__"

        schema = build_list_query_schema(AllOrdering)

        assert "ordering" not in schema["properties"]

    def test_filters_require_django_filter_backend(self):
        class NoFilterBackend(FilterableCustomerViewSet):
            filter_backends = [OrderingFilter]

        schema = build_list_query_schema(NoFilterBackend)

        assert "is_active" not in schema["properties"]
        assert "ordering" in schema["properties"]

    def test_filterset_fields_are_advertised(self):
        schema = build_list_query_schema(FilterFieldsCustomerViewSet)

        props = schema["properties"]
        assert props["is_active"]["type"] == "boolean"
        assert props["age"]["type"] == "number"
