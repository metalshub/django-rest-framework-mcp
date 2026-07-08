"""Test views for django-rest-framework-mcp."""

from django_filters import rest_framework as django_filters
from rest_framework import mixins, viewsets
from rest_framework.authentication import (
    BasicAuthentication,
    SessionAuthentication,
    TokenAuthentication,
)
from rest_framework.filters import OrderingFilter
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import BasePermission, IsAuthenticated
from rest_framework.response import Response

from djangorestframework_mcp.decorators import mcp_viewset

from .models import Customer, Product
from .serializers import CustomerSerializer, ProductSerializer


@mcp_viewset()
class CustomerViewSet(viewsets.ModelViewSet):
    """ViewSet for Customer model with MCP support."""

    queryset = Customer.objects.all()
    serializer_class = CustomerSerializer


@mcp_viewset(basename="products")
class ProductViewSet(viewsets.ModelViewSet):
    """ViewSet for Product model with custom MCP name."""

    queryset = Product.objects.all()
    serializer_class = ProductSerializer


# Authentication test helpers
class CustomAuthentication(TokenAuthentication):
    """Custom authentication class for testing."""

    keyword = "Custom"


class CustomPermission(IsAuthenticated):
    """Custom permission class for testing."""

    message = "Custom permission denied"


class AlwaysDenyPermission(BasePermission):
    """Permission class that always denies but doesn't require auth."""

    message = "Custom permission denied"

    def has_permission(self, request, view):
        from rest_framework.exceptions import PermissionDenied

        raise PermissionDenied(detail=self.message)


# Authentication test ViewSets
@mcp_viewset()
class AuthenticatedViewSet(viewsets.GenericViewSet):
    """Test ViewSet with authentication requirements."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def list(self, request):
        return Response([{"id": 1, "name": "Authenticated Item"}])


@mcp_viewset()
class MultipleAuthViewSet(viewsets.GenericViewSet):
    """Test ViewSet with multiple authentication methods."""

    authentication_classes = [
        TokenAuthentication,
        SessionAuthentication,
        BasicAuthentication,
    ]
    permission_classes = [IsAuthenticated]

    def list(self, request):
        return Response([{"id": 1, "name": "Multi-auth Item"}])


@mcp_viewset()
class UnauthenticatedViewSet(viewsets.GenericViewSet):
    """Test ViewSet without authentication requirements."""

    def list(self, request):
        return Response([{"id": 1, "name": "Public Item"}])


@mcp_viewset()
class CustomAuthViewSet(viewsets.GenericViewSet):
    """Test ViewSet with custom authentication and permission classes."""

    authentication_classes = [CustomAuthentication]
    permission_classes = [CustomPermission]

    def list(self, request):
        return Response([{"id": 1, "name": "Custom Auth Item"}])


@mcp_viewset()
class CustomPermissionViewSet(viewsets.GenericViewSet):
    """Test ViewSet with custom permission that always denies."""

    permission_classes = [AlwaysDenyPermission]

    def list(self, request):
        return Response([{"id": 1, "name": "Should never reach here"}])


# ---------------------------------------------------------------------------
# Fixtures for list filtering/ordering/pagination and view hooks. These are
# registered manually (not decorated) by the tests that use them.
# ---------------------------------------------------------------------------


class CustomerFilterSet(django_filters.FilterSet):
    """FilterSet exercising a boolean filter and a numeric (renamed) filter."""

    is_active = django_filters.BooleanFilter()
    min_age = django_filters.NumberFilter(field_name="age", lookup_expr="gte")

    class Meta:
        model = Customer
        fields = ["is_active", "min_age"]


class SmallPagePagination(PageNumberPagination):
    """Tiny page size so pagination is observable with only a couple of rows."""

    page_size = 2
    page_size_query_param = "page_size"


class FilterableCustomerViewSet(mixins.ListModelMixin, viewsets.GenericViewSet):
    """Customers with filtering, ordering and pagination enabled."""

    queryset = Customer.objects.all().order_by("id")
    serializer_class = CustomerSerializer
    filter_backends = [django_filters.DjangoFilterBackend, OrderingFilter]
    filterset_class = CustomerFilterSet
    ordering_fields = ["age", "name"]
    pagination_class = SmallPagePagination


class FilterFieldsCustomerViewSet(mixins.ListModelMixin, viewsets.GenericViewSet):
    """Uses ``filterset_fields`` (auto-generated FilterSet) instead of a ``filterset_class``."""

    queryset = Customer.objects.all()
    serializer_class = CustomerSerializer
    filter_backends = [django_filters.DjangoFilterBackend]
    filterset_fields = ["is_active", "age"]


class PlainCustomerViewSet(mixins.ListModelMixin, viewsets.GenericViewSet):
    """Customers with no filtering, ordering or pagination configured."""

    queryset = Customer.objects.all()
    serializer_class = CustomerSerializer


class DefaultActiveFilterSet(django_filters.FilterSet):
    """Defaults ``is_active=True`` when unset, so an unfiltered call filters out inactive rows.

    Mirrors real-world filtersets that inject a default in ``__init__`` -- used to prove that omitting
    the ``query`` skips the filter backends entirely rather than running them with their defaults.
    """

    is_active = django_filters.BooleanFilter()

    class Meta:
        model = Customer
        fields = ["is_active"]

    def __init__(self, data=None, *args, **kwargs):
        data = (data or {}).copy()
        data.setdefault("is_active", "true")
        super().__init__(data, *args, **kwargs)


class DefaultFilteringCustomerViewSet(mixins.ListModelMixin, viewsets.GenericViewSet):
    """Customers whose FilterSet defaults to is_active=True."""

    queryset = Customer.objects.all().order_by("id")
    serializer_class = CustomerSerializer
    filter_backends = [django_filters.DjangoFilterBackend]
    filterset_class = DefaultActiveFilterSet
    pagination_class = SmallPagePagination


class EchoViewSet(viewsets.GenericViewSet):
    """Echoes back context attached to the request by an MCPView hook."""

    def list(self, request, *args, **kwargs):
        return Response(
            {
                "injected": getattr(request, "injected", None),
                "from_params": getattr(request, "from_params", None),
            }
        )
