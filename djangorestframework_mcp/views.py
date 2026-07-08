"""MCP HTTP endpoint views."""

import inspect
import json
import logging
from http import HTTPStatus
from typing import Any, Dict, Optional, Type

from django.http import HttpRequest, HttpResponse, JsonResponse, QueryDict
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from rest_framework import exceptions
from rest_framework.authentication import BaseAuthentication
from rest_framework.parsers import JSONParser
from rest_framework.request import Request
from rest_framework.viewsets import GenericViewSet

from .registry import registry
from .schema import generate_tool_schema
from .settings import mcp_settings
from .types import MCPTool

logger = logging.getLogger(__name__)

# The HTTP verb each ViewSet action maps to. MCP is RPC, not REST, but the ViewSet's permissions and
# authentication may be method-aware (SAFE_METHODS, CSRF), so we give the rebuilt request the verb the
# routed call would have carried.
ACTION_METHODS = {
    "list": "GET",
    "retrieve": "GET",
    "create": "POST",
    "update": "PUT",
    "partial_update": "PATCH",
    "destroy": "DELETE",
}


@method_decorator(csrf_exempt, name="dispatch")
class MCPView(View):
    """Main MCP HTTP endpoint handler."""

    # Override the definition of these to enforce authentication on the MCP endpoint
    authentication_classes: list[Type[BaseAuthentication]] = []

    def has_mcp_permission(self, request: HttpRequest) -> bool:
        """
        Override this method to implement custom permission logic for the MCP endpoint.

        Args:
            request: The Django HttpRequest object (.user and .auth will be set if authenticated)

        Returns:
            bool: True if the request should be allowed, False otherwise

        Default behavior: Allow all requests (return True)
        """
        return True

    def post(self, request):
        """Handle MCP requests."""
        try:
            # Parse the JSON-RPC request
            body = json.loads(request.body)

            # Extract the method and params
            method = body.get("method")
            params = body.get("params", {})
            request_id = body.get("id")

            # Perform authentication and permission checks for the MCP endpoint
            self.perform_mcp_authentication_and_permissions_check(request)

            # Route to appropriate handler
            if method == "initialize":
                result = self.handle_initialize()
            elif method == "notifications/initialized":
                # Sent by the client to acknowledge the receipt of our response to its initialize handshake
                # No response is expected
                return HttpResponse(status=HTTPStatus.NO_CONTENT)
            elif method == "tools/list":
                result = self.handle_tools_list()
            elif method == "tools/call":
                result = self.handle_tools_call(params, request)
            else:
                # Method not found
                return self.error_response(
                    request_id, -32601, f"Method not found: {method}"
                )

            # Return JSON-RPC response
            return JsonResponse({"jsonrpc": "2.0", "result": result, "id": request_id})

        except json.JSONDecodeError:
            return self.error_response(None, -32700, "Parse error")
        except (
            exceptions.AuthenticationFailed,
            exceptions.NotAuthenticated,
            exceptions.PermissionDenied,
        ) as exc:
            return self.handle_auth_error(
                exc, body.get("id") if "body" in locals() else None
            )
        except Exception as e:
            return self.error_response(
                body.get("id") if "body" in locals() else None,
                -32603,
                f"Internal error: {str(e)}",
            )

    def handle_initialize(self) -> Dict[str, Any]:
        """Handle initialize request."""
        # The only capabilities we currently support are tool calling (without listChanged notifications)
        return {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "django-rest-framework-mcp", "version": "0.1.0a2"},
        }

    def handle_tools_list(self) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = []
        for tool in registry.get_all_tools():
            entry = self.build_tool_definition(tool)
            if entry is not None:
                tools.append(entry)
        return {"tools": tools}

    def build_tool_definition(self, tool: MCPTool) -> Optional[Dict[str, Any]]:
        """Build the tools/list entry for ``tool``, or ``None`` if it can't be advertised.

        A ViewSet whose serializer drf-mcp can't introspect (custom fields, ``serializer_class = None``,
        ...) raises during schema generation; rather than break ``tools/list`` for the whole server we
        log and skip that one tool so every other tool is still discoverable. Subclasses can call this
        and then extend the returned entry (e.g. add extra inputs to ``inputSchema``).
        """
        try:
            tool_schema = generate_tool_schema(tool)
        except Exception as exc:  # noqa: BLE001 - one bad serializer must not break discovery
            logger.warning(
                "Skipping MCP tool %s: input schema generation failed (%s)",
                tool.name,
                exc,
            )
            return None

        entry: Dict[str, Any] = {
            "name": tool.name,
            "description": self.get_tool_description(tool),
            "inputSchema": tool_schema["inputSchema"],
        }
        if tool.title:
            entry["title"] = tool.title
        annotations = self.get_tool_annotations(tool)
        if annotations:
            entry["annotations"] = annotations
        return entry

    def get_tool_description(self, tool: MCPTool) -> Optional[str]:
        """The description advertised for ``tool`` in tools/list.

        Prefers an explicit ``@mcp_tool(description=...)``. Otherwise, when the description was
        auto-generated, falls back to the ViewSet's docstring summary (its first paragraph) if it has
        one -- which reads far better for an LLM than the robotic ``"<Action> <basename>"`` default, and
        lets two ViewSets over the same model be told apart. Falls back to the generated default when
        there is no docstring.
        """
        if getattr(tool, "description_is_auto", False):
            # The ViewSet's *own* docstring only -- never one inherited from a DRF mixin (e.g.
            # ListModelMixin's "List a queryset."), which would be misleading.
            doc = tool.viewset_class.__dict__.get("__doc__")
            if doc:
                doc = inspect.cleandoc(doc)
                summary = " ".join(doc.split("\n\n", 1)[0].split())
                if summary:
                    return summary
        return tool.description

    def get_tool_annotations(self, tool: MCPTool) -> Dict[str, Any]:
        """MCP tool ``annotations`` hints derived from the action.

        ``readOnlyHint`` lets clients group tools (e.g. Claude's connector UI only buckets a tool under
        "Read-only tools" when it is present and true). Write actions additionally advertise
        ``destructiveHint`` / ``idempotentHint``. ``openWorldHint`` is ``False``: a ViewSet tool operates
        on the application's own data, not an open-ended external system.
        """
        read_only = tool.action in ("list", "retrieve")
        annotations: Dict[str, Any] = {
            "readOnlyHint": read_only,
            "openWorldHint": False,
        }
        if not read_only:
            annotations["destructiveHint"] = tool.action == "destroy"
            annotations["idempotentHint"] = tool.action in (
                "update",
                "partial_update",
                "destroy",
            )
        return annotations

    def handle_tools_call(
        self, params: Dict[str, Any], original_request: HttpRequest
    ) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        tool_params = params.get("arguments", {})

        try:
            # Find the tool
            if not tool_name:
                raise Exception("Tool name is required")
            tool = registry.get_tool_by_name(tool_name)
            if not tool:
                # This should be handled as a protocol-level error, not a tool execution error
                raise Exception(f"Tool not found: {tool_name}")

            # Execute the tool
            result = self.execute_tool(tool, tool_params, original_request)

            # Per latest MCP specification (2025-06-18), JSON should be returned in both
            # structured content and as stringified text content (the latter for backwards compatibility)
            response = {
                "content": [{"type": "text", "text": json.dumps(result, default=str)}]
            }
            # Add structured content if result is JSON-serializable
            try:
                # Test if result can be JSON serialized (for structuredContent validation)
                json.dumps(result)
                response["structuredContent"] = result
            except (TypeError, ValueError):
                # If result contains non-JSON-serializable data, skip structuredContent
                # The text content will still contain the string representation
                pass

            return response

        except (
            exceptions.AuthenticationFailed,
            exceptions.NotAuthenticated,
            exceptions.PermissionDenied,
        ) as exc:
            # Re-raise authentication/permission errors to be handled at HTTP level
            raise exc
        except Exception as e:
            return {
                "content": [
                    {"type": "text", "text": f"Error executing tool: {str(e)}"}
                ],
                "isError": True,
            }

    def perform_mcp_authentication_and_permissions_check(self, request: HttpRequest):
        """Perform authentication for the MCP endpoint."""
        authenticators = [auth() for auth in self.authentication_classes]

        # Convert HttpRequest to DRF Request for authentication
        drf_request = Request(
            request,
            parsers=[JSONParser()],
            authenticators=authenticators,
        )

        try:
            # Trigger authentication by accessing the user property. This runs through all
            # authenticators and (via DRF's Request.user setter) also sets user/auth on the underlying
            # request, so has_mcp_permission below can rely on request.user / request.auth.
            _ = drf_request.user

            # Check permissions
            if not self.has_mcp_permission(request):
                # If request is not permitted, determine what kind of exception to raise.
                if authenticators and not drf_request.successful_authenticator:
                    raise exceptions.NotAuthenticated()
                raise exceptions.PermissionDenied()
        except (
            exceptions.AuthenticationFailed,
            exceptions.NotAuthenticated,
        ) as exc:
            # Add WWW-Authenticate header if we have authenticators
            if authenticators:
                exc.auth_header = authenticators[0].authenticate_header(drf_request)  # type: ignore[union-attr]
            raise

        # Copy authenticated user/auth to the original request
        request.user = drf_request.user
        request.auth = drf_request.auth

    def handle_auth_error(
        self, exc: exceptions.APIException, request_id: Optional[Any]
    ) -> JsonResponse:
        """Handle authentication/permission errors with proper HTTP status and headers."""
        headers = {}

        # Build error message with additional context
        error_message = str(exc.detail)
        try:
            error_message = f"{HTTPStatus(exc.status_code).phrase}: {error_message}"
        except ValueError:
            # If exc.status_code not in HTTPStatus, just continue
            pass

        # Add WWW-Authenticate info to the message for LLM context
        if getattr(exc, "auth_header", None):
            error_message += f" (WWW-Authenticate: {exc.auth_header})"
            if not mcp_settings.RETURN_200_FOR_ERRORS:
                headers["WWW-Authenticate"] = exc.auth_header

        # Determine HTTP status code based on RETURN_200_FOR_ERRORS setting
        http_status = (
            HTTPStatus.OK if mcp_settings.RETURN_200_FOR_ERRORS else exc.status_code
        )
        response = JsonResponse(
            {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{"type": "text", "text": error_message}],
                    "isError": True,
                },
                "id": request_id,
            },
            status=http_status,
        )

        # Add HTTP headers
        for key, value in headers.items():
            response[key] = value

        return response

    def error_response(
        self, request_id: Optional[Any], code: int, message: str
    ) -> JsonResponse:
        """Create a JSON-RPC error response."""
        return JsonResponse(
            {
                "jsonrpc": "2.0",
                "error": {"code": code, "message": message},
                "id": request_id,
            }
        )

    def build_tool_request(
        self,
        tool: MCPTool,
        original_request: HttpRequest,
        *,
        body: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
    ) -> Request:
        """Build a DRF ``Request`` equivalent to the routed API call for ``tool``.

        Carries over the original request's META and the user/auth authenticated at the MCP endpoint,
        sets the HTTP method implied by the action, installs ``body`` as the JSON payload, and applies an
        optional list ``query`` as a real query string. Used by :meth:`execute_tool`, and available to
        subclasses that need to rebuild a request per tool (e.g. probing permissions during discovery).
        """
        bypass_viewset_auth = mcp_settings.BYPASS_VIEWSET_AUTHENTICATION

        # Create a new HttpRequest that represents the equivalent API call
        body_bytes = json.dumps(body).encode("utf-8") if body else b"{}"
        request = HttpRequest()

        # Carry over META and authenticated user info from the original request
        for key, value in original_request.META.items():
            request.META[key] = value
        request.method = ACTION_METHODS.get(tool.action, "GET")
        if hasattr(original_request, "user"):
            request.user = original_request.user
        if hasattr(original_request, "auth"):
            request.auth = original_request.auth

        # Replace the body with the body that was passed in via params
        request.META["HTTP_CONTENT_TYPE"] = "application/json"
        request.META["HTTP_CONTENT_LENGTH"] = str(len(body_bytes))
        request._body = body_bytes
        # We aren't creating a proper stream. Marking it as started tells the parser it does not need to
        # read it as a stream.
        request._read_started = True

        if query:
            self._apply_query_params(request, query)

        # Based on `rest_framework.views.APIView.initialize_request`, but without content negotiation
        # since that doesn't apply to an MCP request.
        authenticators = (
            [] if bypass_viewset_auth else tool.viewset_class().get_authenticators()
        )
        drf_request = Request(
            request,
            parsers=[JSONParser()],  # MCP always uses JSON
            authenticators=authenticators,
        )

        # If bypassing ViewSet auth, carry over the user authenticated at the MCP endpoint.
        if bypass_viewset_auth:
            if hasattr(request, "user"):
                drf_request.user = request.user
            if hasattr(request, "auth"):
                drf_request.auth = request.auth

        # Mark request as coming from MCP
        drf_request.is_mcp_request = True
        return drf_request

    def build_viewset(
        self,
        tool: MCPTool,
        drf_request: Request,
        *,
        method_kwargs: Optional[Dict[str, Any]] = None,
    ) -> GenericViewSet:
        """Instantiate and initialise the ViewSet for ``tool``, as DRF's dispatch would for this action."""
        viewset = tool.viewset_class()
        # From `rest_framework.viewsets.ViewSetMixin.initialize_request` / APIView.dispatch:
        viewset.action = tool.action
        viewset.args = ()
        viewset.kwargs = dict(method_kwargs or {})
        viewset.headers = {}  # In the future, this will be passed in via a headers param.
        viewset.request = drf_request
        viewset.format_kwarg = None
        return viewset

    @staticmethod
    def _apply_query_params(request: HttpRequest, query: Dict[str, Any]) -> None:
        """Write an MCP ``query`` object onto ``request`` as a real query string.

        List-valued entries (multi-choice filters) become repeated params so django-filter reads them as
        a list, matching how the same call arrives over HTTP. ``None`` values are dropped.
        """
        query_dict = QueryDict(mutable=True)
        for key, value in query.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                query_dict.setlist(key, [str(item) for item in value])
            elif isinstance(value, bool):
                query_dict[key] = "true" if value else "false"
            else:
                query_dict[key] = str(value)
        request.GET = query_dict  # type: ignore[assignment]
        request.META["QUERY_STRING"] = query_dict.urlencode()

    def prepare_tool_request(
        self,
        request: Request,
        original_request: HttpRequest,
        tool: MCPTool,
        params: Dict[str, Any],
    ) -> None:
        """Hook: attach extra context to the per-tool ``request`` before it is authorised and run.

        Called by :meth:`execute_tool` after the request is built but before the ViewSet's permission
        classes run, so overrides can attach state that routing/middleware would normally provide (e.g.
        multi-tenant context resolved from ``params``). Raising here aborts the call (reported as a tool
        error). Default: no-op.
        """
        return None

    def on_viewset_permission_denied(
        self, tool: MCPTool, exc: exceptions.PermissionDenied, request: Request
    ) -> None:
        """Hook: called when the ViewSet's permission classes deny a ``tools/call``.

        Default re-raises the ``PermissionDenied`` so it surfaces as an HTTP 403 (or a 200 body under
        ``RETURN_200_FOR_ERRORS``), consistent with the MCP endpoint's own permission handling. Override
        to turn a per-call authorization failure into a tool-level error instead (raise a plain
        exception, which :meth:`handle_tools_call` reports with ``isError``) -- useful when a 403 would
        wrongly prompt an MCP client to re-authenticate the entire connection.
        """
        raise exc

    def execute_tool(
        self, tool: MCPTool, params: Dict[str, Any], original_request: HttpRequest
    ) -> Any:
        """Execute a tool using the structured kwargs+body parameter format.

        This manually replicates the parts of `rest_framework.views.APIView.dispatch` that apply to an
        MCP (RPC-based) call: build an equivalent DRF request, authenticate/authorize it per the
        settings, then call the action method directly.
        """
        bypass_viewset_auth = mcp_settings.BYPASS_VIEWSET_AUTHENTICATION
        bypass_viewset_permissions = mcp_settings.BYPASS_VIEWSET_PERMISSIONS

        # Extract structured parameters
        method_kwargs = dict(params.get("kwargs", {}))
        body = params.get("body", {})
        # Filtering/ordering/pagination via `query` is only meaningful on the list action.
        query = params.get("query") if tool.action == "list" else None

        drf_request = self.build_tool_request(
            tool, original_request, body=body, query=query
        )
        # Hook for subclasses to attach per-call context (e.g. tenancy) before authorization runs.
        self.prepare_tool_request(drf_request, original_request, tool, params)
        viewset = self.build_viewset(tool, drf_request, method_kwargs=method_kwargs)

        # The `query` object is the list tool's filtering/ordering interface. When it is omitted, don't
        # run the ViewSet's filter backends at all -- return the default (paginated) queryset. This
        # mirrors "the caller didn't ask to filter" and avoids surprising results from filtersets that
        # filter by default, require a field, or treat empty input as "match nothing". Pass a `query` to
        # opt into filtering/ordering. (Pagination is applied by the action itself, not a filter backend,
        # so it still works either way.)
        if tool.action == "list" and not query:
            viewset.filter_backends = []

        if not hasattr(viewset, tool.action):
            raise ValueError(f"ViewSet does not support action: {tool.action}")

        # Perform authentication and permissions based on settings, authorizing the action the same way
        # the web app would.
        try:
            if not bypass_viewset_auth:
                viewset.perform_authentication(drf_request)
            if not bypass_viewset_permissions:
                viewset.check_permissions(drf_request)
        except (exceptions.AuthenticationFailed, exceptions.NotAuthenticated) as exc:
            # Set WWW-Authenticate header for auth-related errors
            authenticators = viewset.get_authenticators()
            if authenticators:
                exc.auth_header = authenticators[0].authenticate_header(drf_request)  # type: ignore[union-attr]
            raise
        except exceptions.PermissionDenied as exc:
            # Authenticated but not authorized for this action. By default re-raised (HTTP 403);
            # subclasses may convert it into a tool-level error via on_viewset_permission_denied.
            self.on_viewset_permission_denied(tool, exc, drf_request)
            raise  # fail closed if a hook neither raised nor returned

        # Check throttles
        viewset.check_throttles(drf_request)

        # Handle versioning
        version, scheme = viewset.determine_version(
            drf_request, *viewset.args, **viewset.kwargs
        )
        drf_request.version, drf_request.versioning_scheme = version, scheme

        # Get and call the action method directly
        action_method = getattr(viewset, tool.action)
        response = action_method(drf_request, **method_kwargs)
        return self._handle_tool_response(response)

    @staticmethod
    def _handle_tool_response(response: Any) -> Any:
        """Convert a DRF Response into the tool result, raising on error responses."""
        if hasattr(response, "data"):
            # Handle DRF error responses
            if response.status_code >= HTTPStatus.BAD_REQUEST.value:
                raise ValueError(f"ViewSet returned error: {response.data}")

            # Handle successful responses
            if response.data is not None:
                return response.data
            # For responses like 204 No Content (destroy), return a success message
            return {"message": "Operation completed successfully"}

        return response
