import asyncio
import httpx
from mcp.server.fastmcp import FastMCP
import logging
import os
import base64
from typing import Optional, Dict, Union, Any, List
from enum import IntEnum, Enum
import re
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("freshdesk_mcp")

# Initialize FastMCP server
mcp = FastMCP("freshdesk-mcp")

FRESHDESK_API_KEY = (os.getenv("FRESHDESK_API_KEY") or "").strip()
FRESHDESK_DOMAIN = (os.getenv("FRESHDESK_DOMAIN") or "").strip().rstrip("/")

if not FRESHDESK_API_KEY:
    raise ValueError("FRESHDESK_API_KEY environment variable is required")
if not FRESHDESK_DOMAIN:
    raise ValueError("FRESHDESK_DOMAIN environment variable is required")

AUTH_HEADER = f"Basic {base64.b64encode(f'{FRESHDESK_API_KEY}:X'.encode()).decode()}"

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

MAX_RETRIES = 3
DEFAULT_RETRY_AFTER = 2


async def _request_with_retry(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> httpx.Response:
    """Make an HTTP request with automatic retry on 429 (rate limit) responses."""
    http_fn = getattr(client, method)
    for attempt in range(MAX_RETRIES + 1):
        response = await http_fn(url, **kwargs)
        if response.status_code == 429 and attempt < MAX_RETRIES:
            retry_after = int(response.headers.get("Retry-After", DEFAULT_RETRY_AFTER))
            logger.warning("Rate limited (429). Retry in %ds (%d/%d)", retry_after, attempt + 1, MAX_RETRIES)
            await asyncio.sleep(retry_after)
            continue
        return response
    return response


def parse_link_header(link_header: str) -> Dict[str, Optional[int]]:
    """Parse the Link header to extract pagination information.

    Args:
        link_header: The Link header string from the response

    Returns:
        Dictionary containing next and prev page numbers
    """
    pagination = {
        "next": None,
        "prev": None
    }

    if not link_header:
        return pagination

    # Split multiple links if present
    links = link_header.split(',')

    for link in links:
        # Extract URL and rel
        match = re.search(r'<(.+?)>;\s*rel="(.+?)"', link)
        if match:
            url, rel = match.groups()
            # Extract page number from URL
            page_match = re.search(r'page=(\d+)', url)
            if page_match:
                page_num = int(page_match.group(1))
                pagination[rel] = page_num

    return pagination

# enums of ticket properties
class TicketSource(IntEnum):
    EMAIL = 1
    PORTAL = 2
    PHONE = 3
    CHAT = 7
    FEEDBACK_WIDGET = 9
    OUTBOUND_EMAIL = 10

class TicketStatus(IntEnum):
    OPEN = 2
    PENDING = 3
    RESOLVED = 4
    CLOSED = 5

class TicketPriority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class AgentTicketScope(IntEnum):
    GLOBAL_ACCESS = 1
    GROUP_ACCESS = 2
    RESTRICTED_ACCESS = 3

class UnassignedForOptions(str, Enum):
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    TWO_HOURS = "2h"
    FOUR_HOURS = "4h"
    EIGHT_HOURS = "8h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"
    TWO_DAYS = "2d"
    THREE_DAYS = "3d"

class GroupCreate(BaseModel):
    name: str = Field(..., description="Name of the group")
    description: Optional[str] = Field(None, description="Description of the group")
    agent_ids: Optional[List[int]] = Field(
        default=None,
        description="Array of agent user ids"
    )
    auto_ticket_assign: Optional[int] = Field(
        default=0,
        ge=0,
        le=1,
        description="Automatic ticket assignment type (0 or 1)"
    )
    escalate_to: Optional[int] = Field(
        None,
        description="User ID to whom escalation email is sent if ticket is unassigned"
    )
    unassigned_for: Optional[UnassignedForOptions] = Field(
        default=UnassignedForOptions.THIRTY_MIN,
        description="Time after which escalation email will be sent"
    )

class ContactFieldCreate(BaseModel):
    label: str = Field(..., description="Display name for the field (as seen by agents)")
    label_for_customers: str = Field(..., description="Display name for the field (as seen by customers)")
    type: str = Field(
        ...,
        description="Type of the field",
        pattern="^(custom_text|custom_paragraph|custom_checkbox|custom_number|custom_dropdown|custom_phone_number|custom_url|custom_date)$"
    )
    editable_in_signup: bool = Field(
        default=False,
        description="Set to true if the field can be updated by customers during signup"
    )
    position: int = Field(
        default=1,
        description="Position of the company field"
    )
    required_for_agents: bool = Field(
        default=False,
        description="Set to true if the field is mandatory for agents"
    )
    customers_can_edit: bool = Field(
        default=False,
        description="Set to true if the customer can edit the fields in the customer portal"
    )
    required_for_customers: bool = Field(
        default=False,
        description="Set to true if the field is mandatory in the customer portal"
    )
    displayed_for_customers: bool = Field(
        default=False,
        description="Set to true if the customers can see the field in the customer portal"
    )
    choices: Optional[List[Dict[str, Union[str, int]]]] = Field(
        default=None,
        description="Array of objects in format {'value': 'Choice text', 'position': 1} for dropdown choices"
    )

class CannedResponseCreate(BaseModel):
    title: str = Field(..., description="Title of the canned response")
    content_html: str = Field(..., description="HTML version of the canned response content")
    folder_id: int = Field(..., description="Folder where the canned response gets added")
    visibility: int = Field(
        ...,
        description="Visibility of the canned response (0=all agents, 1=personal, 2=select groups)",
        ge=0,
        le=2
    )
    group_ids: Optional[List[int]] = Field(
        None,
        description="Groups for which the canned response is visible. Required if visibility=2"
    )

@mcp.tool()
async def get_ticket_fields() -> Dict[str, Any]:
    """Get ticket fields from Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/ticket_fields"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return {"fields": response.json()}
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
async def get_tickets(
    page: Optional[int] = 1,
    per_page: Optional[int] = 10,
    filter: Optional[str] = None,
    requester_id: Optional[int] = None,
    email: Optional[str] = None,
    name: Optional[str] = None,
    company_id: Optional[int] = None,
    updated_since: Optional[str] = None,
    order_by: Optional[str] = None,
    order_type: Optional[str] = None,
    include: Optional[str] = None,
) -> Dict[str, Any]:
    """Get tickets from Freshdesk with filtering, sorting, and pagination.

    Use this tool when filtering by requester name (e.g. "tickets from John Smith") — pass name="John Smith".
    It will automatically resolve the name to a contact ID and then fetch their tickets.

    Args:
        page: Page number (default 1).
        per_page: Results per page, 1-100 (default 30).
        filter: Predefined filter — one of "new_and_my_open", "watching", "spam", "deleted".
        requester_id: Filter by requester (contact) ID.
        email: Filter by requester email address.
        name: Filter by requester name — e.g. "John Smith". Automatically looks up the contact
              by name and returns their tickets. Use this for queries like "show tickets from [person]".
        company_id: Filter by company ID.
        updated_since: Return tickets updated after this datetime (ISO 8601, e.g. "2026-03-17T00:00:00Z").
        order_by: Sort field — one of "created_at", "due_by", "updated_at", "status" (default "created_at").
        order_type: Sort direction — "asc" or "desc" (default "desc").
        include: Extra data to embed — comma-separated: "stats", "requester", "description", "company".
    """
    # Validate input parameters
    if page < 1:
        return {"error": "Page number must be greater than 0"}

    if per_page < 1 or per_page > 100:
        return {"error": "Page size must be between 1 and 100"}

    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }

    # If name is provided, resolve it to a requester_id via contacts autocomplete
    if name and not requester_id:
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                ac_url = f"https://{FRESHDESK_DOMAIN}/api/v2/contacts/autocomplete"
                ac_response = await _request_with_retry(client, "get", ac_url, headers=headers, params={"term": name})
                ac_response.raise_for_status()
                contacts = ac_response.json()
                if contacts:
                    requester_id = contacts[0].get("id")
                else:
                    return {"error": f"No contact found matching name '{name}'", "tickets": [], "pagination": {}}
        except Exception as e:
            return {"error": f"Failed to resolve name '{name}' to a contact: {str(e)}"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets"

    params: Dict[str, Any] = {
        "page": page,
        "per_page": per_page
    }

    if filter:
        params["filter"] = filter
    if requester_id:
        params["requester_id"] = requester_id
    if email:
        params["email"] = email
    if company_id:
        params["company_id"] = company_id
    if updated_since:
        params["updated_since"] = updated_since
    if order_by:
        params["order_by"] = order_by
    if order_type:
        params["order_type"] = order_type
    if include:
        params["include"] = include

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers, params=params)
            response.raise_for_status()

            # Parse pagination from Link header
            link_header = response.headers.get('Link', '')
            pagination_info = parse_link_header(link_header)

            tickets = response.json()

            # Return compact ticket summaries to avoid exceeding token limits
            SUMMARY_FIELDS = [
                "id", "subject", "status", "priority", "type",
                "requester_id", "responder_id", "group_id", "company_id",
                "email_config_id", "source", "tags", "due_by",
                "created_at", "updated_at",
            ]
            compact_tickets = [
                {k: t[k] for k in SUMMARY_FIELDS if k in t}
                for t in tickets
            ]

            return {
                "tickets": compact_tickets,
                "pagination": {
                    "current_page": page,
                    "next_page": pagination_info.get("next"),
                    "prev_page": pagination_info.get("prev"),
                    "per_page": per_page
                },
                "note": "Use get_ticket(ticket_id) for full ticket details including description."
            }

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch tickets: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def create_ticket(
    subject: str,
    description: str,
    source: Union[int, str],
    priority: Union[int, str],
    status: Union[int, str],
    email: Optional[str] = None,
    requester_id: Optional[int] = None,
    custom_fields: Optional[Dict[str, Any]] = None,
    additional_fields: Optional[Dict[str, Any]] = None
) -> str:
    """Create a ticket in Freshdesk"""
    # Validate requester information
    if not email and not requester_id:
        return "Error: Either email or requester_id must be provided"

    # Convert string inputs to integers if necessary
    try:
        source_val = int(source)
        priority_val = int(priority)
        status_val = int(status)
    except ValueError:
        return "Error: Invalid value for source, priority, or status"

    # Validate enum values
    if (source_val not in [e.value for e in TicketSource] or
        priority_val not in [e.value for e in TicketPriority] or
        status_val not in [e.value for e in TicketStatus]):
        return "Error: Invalid value for source, priority, or status"

    # Prepare the request data
    data = {
        "subject": subject,
        "description": description,
        "source": source_val,
        "priority": priority_val,
        "status": status_val
    }

    # Add requester information
    if email:
        data["email"] = email
    if requester_id:
        data["requester_id"] = requester_id

    # Add custom fields if provided
    if custom_fields:
        data["custom_fields"] = custom_fields

     # Add any other top-level fields
    if additional_fields:
        data.update(additional_fields)

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets"
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "post", url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"Failed to create ticket: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def update_ticket(ticket_id: int, ticket_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a ticket in Freshdesk."""
    if not ticket_fields:
        return {"error": "No fields provided for update"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }

    # Separate custom fields from standard fields
    custom_fields = ticket_fields.pop('custom_fields', {})

    # Prepare the update data
    update_data = {}

    # Add standard fields if they are provided
    for field, value in ticket_fields.items():
        update_data[field] = value

    # Add custom fields if they exist
    if custom_fields:
        update_data['custom_fields'] = custom_fields

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "put", url, headers=headers, json=update_data)
            response.raise_for_status()

            return {
                "success": True,
                "message": "Ticket updated successfully",
                "ticket": response.json()
            }

        except httpx.HTTPStatusError as e:
            error_message = f"Failed to update ticket: {str(e)}"
            try:
                error_details = e.response.json()
                if "errors" in error_details:
                    error_message = f"Validation errors: {error_details['errors']}"
            except Exception:
                pass
            return {
                "success": False,
                "error": error_message
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}"
            }

@mcp.tool()
async def delete_ticket(ticket_id: int) -> Dict[str, Any]:
    """Delete a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "delete", url, headers=headers)
            if response.status_code == 204:
                return {"success": True, "message": "Ticket deleted successfully"}
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"Failed to delete ticket: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_ticket(ticket_id: int) -> Dict[str, Any]:
    """Get a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def search_tickets(query: str) -> Dict[str, Any]:
    """Search for tickets in Freshdesk using raw Freshdesk query syntax.

    IMPORTANT: Do NOT use this tool to search by requester name — Freshdesk does not support
    name-based ticket search. Use get_tickets(name="...") for name lookups instead.

    Valid Freshdesk query syntax examples:
        "priority:3 AND status:2"
        "requester_email:'xyz@gmail.com'"
        "created_at:>'2026-03-01' AND status:4"

    The tool auto-wraps the query in double quotes if needed.
    Prefer filter_tickets for common structured searches.
    """
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/search/tickets"
    headers = {
        "Authorization": AUTH_HEADER
    }
    # Freshdesk search API requires the query value wrapped in double quotes
    if not query.startswith('"'):
        query = f'"{query}"'
    params = {"query": query}
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}


def _build_search_query(fields: Dict[str, Any]) -> str:
    """Build a Freshdesk search query string from structured parameters.

    Supports exact match fields (e.g. priority:3), string fields (e.g. requester_email:'x@y.com'),
    and date range fields (e.g. created_at:>'2026-03-17').

    Returns a double-quoted query string ready for the Freshdesk search API.
    """
    clauses = []

    # Exact numeric/enum fields
    exact_fields = {
        "priority": "priority",
        "status": "status",
        "agent_id": "agent_id",
        "group_id": "group_id",
        "requester_id": "requester_id",
        "company_id": "company_id",
    }
    for param, fd_field in exact_fields.items():
        if fields.get(param) is not None:
            clauses.append(f"{fd_field}:{fields[param]}")

    # String fields (wrapped in single quotes)
    string_fields = {
        "requester_email": "requester_email",
        "type": "type",
        "tag": "tag",
    }
    for param, fd_field in string_fields.items():
        if fields.get(param):
            clauses.append(f"{fd_field}:'{fields[param]}'")

    # Date range fields
    date_ranges = {
        "created_after": ("created_at", ">"),
        "created_before": ("created_at", "<"),
        "updated_after": ("updated_at", ">"),
        "updated_before": ("updated_at", "<"),
        "due_by_before": ("due_by", "<"),
    }
    for param, (fd_field, op) in date_ranges.items():
        if fields.get(param):
            clauses.append(f"{fd_field}:{op}'{fields[param]}'")

    query_str = " AND ".join(clauses)
    return f'"{query_str}"'


@mcp.tool()
async def filter_tickets(
    requester_email: Optional[str] = None,
    priority: Optional[int] = None,
    status: Optional[int] = None,
    agent_id: Optional[int] = None,
    group_id: Optional[int] = None,
    requester_id: Optional[int] = None,
    company_id: Optional[int] = None,
    type: Optional[str] = None,
    tag: Optional[str] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    updated_after: Optional[str] = None,
    updated_before: Optional[str] = None,
    due_by_before: Optional[str] = None,
) -> Dict[str, Any]:
    """Search tickets using structured filters — no need to write raw query syntax.

    IMPORTANT: This tool does NOT support filtering by requester name.
    - To find tickets by a person's name (e.g. "tickets from John Smith"), use get_tickets(name="John Smith") instead.
    - To find tickets by email, use requester_email here or get_tickets(email="...").

    Args:
        requester_email: Filter by requester email (e.g. "xyz@gmail.com").
        priority: Filter by priority (1=Low, 2=Medium, 3=High, 4=Urgent).
        status: Filter by status (2=Open, 3=Pending, 4=Resolved, 5=Closed).
        agent_id: Filter by assigned agent ID.
        group_id: Filter by assigned group ID.
        requester_id: Filter by requester contact ID (numeric ID, not name).
        company_id: Filter by company ID.
        type: Filter by ticket type value.
        tag: Filter by tag.
        created_after: Tickets created after this date (ISO 8601, e.g. "2026-03-17").
        created_before: Tickets created before this date (ISO 8601).
        updated_after: Tickets updated after this date (ISO 8601).
        updated_before: Tickets updated before this date (ISO 8601).
        due_by_before: Tickets due before this date (ISO 8601).

    At least one filter parameter must be provided.
    """
    fields = {
        "requester_email": requester_email,
        "priority": priority,
        "status": status,
        "agent_id": agent_id,
        "group_id": group_id,
        "requester_id": requester_id,
        "company_id": company_id,
        "type": type,
        "tag": tag,
        "created_after": created_after,
        "created_before": created_before,
        "updated_after": updated_after,
        "updated_before": updated_before,
        "due_by_before": due_by_before,
    }

    # Check at least one filter is provided
    if not any(v is not None for v in fields.values()):
        return {"error": "At least one filter parameter must be provided"}

    query = _build_search_query(fields)

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/search/tickets"
    headers = {
        "Authorization": AUTH_HEADER
    }
    params = {"query": query}

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            # Freshdesk Search API doesn't support sorting, so sort client-side
            # by created_at descending (newest first) to match Freshdesk UI order
            if "results" in data and isinstance(data["results"], list):
                data["results"] = sorted(
                    data["results"],
                    key=lambda t: t.get("created_at", ""),
                    reverse=True,
                )
                # Return compact summaries to avoid exceeding token limits
                SUMMARY_FIELDS = [
                    "id", "subject", "status", "priority", "type",
                    "requester_id", "responder_id", "group_id", "company_id",
                    "source", "tags", "due_by", "created_at", "updated_at",
                ]
                data["results"] = [
                    {k: t[k] for k in SUMMARY_FIELDS if k in t}
                    for t in data["results"]
                ]
                data["note"] = "Use get_ticket(ticket_id) for full ticket details including description."
            return data
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
async def get_ticket_conversation(ticket_id: int) -> Dict[str, Any]:
    """Get a ticket conversation in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/conversations"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def create_ticket_reply(ticket_id: int, body: str) -> Dict[str, Any]:
    """Create a reply to a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/reply"
    headers = {
        "Authorization": AUTH_HEADER
    }
    data = {
        "body": body
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "post", url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def create_ticket_note(ticket_id: int, body: str) -> Dict[str, Any]:
    """Create a note for a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/notes"
    headers = {
        "Authorization": AUTH_HEADER
    }
    data = {
        "body": body
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "post", url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def update_ticket_conversation(conversation_id: int, body: str) -> Dict[str, Any]:
    """Update a conversation for a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/conversations/{conversation_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    data = {
        "body": body
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "put", url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"Failed to update conversation: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_agents(page: Optional[int] = 1, per_page: Optional[int] = 30) -> list[Dict[str, Any]]:
    """Get all agents in Freshdesk with pagination support."""
    # Validate input parameters
    if page < 1:
        return {"error": "Page number must be greater than 0"}

    if per_page < 1 or per_page > 100:
        return {"error": "Page size must be between 1 and 100"}
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/agents"
    headers = {
        "Authorization": AUTH_HEADER
    }
    params = {
        "page": page,
        "per_page": per_page
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def list_contacts(page: Optional[int] = 1, per_page: Optional[int] = 30) -> list[Dict[str, Any]]:
    """List all contacts in Freshdesk with pagination support."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contacts"
    headers = {
        "Authorization": AUTH_HEADER
    }
    params = {
        "page": page,
        "per_page": per_page
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_contact(contact_id: int) -> Dict[str, Any]:
    """Get a contact in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contacts/{contact_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def search_contacts(query: str) -> list[Dict[str, Any]]:
    """Search for contacts by name in Freshdesk using fuzzy/partial matching.

    This is the best tool for finding a contact by name — it uses autocomplete
    and will match partial names (e.g. "Petr" finds "Petr Halik").
    Use this when looking up a person by name. Use filter_contacts for exact
    field matching (email, phone, company_id).
    """
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contacts/autocomplete"
    headers = {
        "Authorization": AUTH_HEADER
    }
    params = {"term": query}
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
async def filter_contacts(
    email: Optional[str] = None,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    company_id: Optional[int] = None,
    updated_since: Optional[str] = None,
) -> Dict[str, Any]:
    """Search contacts using exact field filters via the Freshdesk search API.

    IMPORTANT: For looking up a contact by name, prefer search_contacts instead —
    it uses fuzzy/partial matching. This tool requires exact name matches.
    Use this tool when filtering by email, phone, company_id, or date.

    Args:
        email: Filter by exact email address.
        name: Filter by exact contact name (use search_contacts for partial/fuzzy name matching).
        phone: Filter by phone number.
        company_id: Filter by company ID.
        updated_since: Contacts updated after this date (ISO 8601, e.g. "2026-03-17").

    At least one filter parameter must be provided.
    """
    clauses = []
    if email:
        clauses.append(f"email:'{email}'")
    if name:
        clauses.append(f"name:'{name}'")
    if phone:
        clauses.append(f"phone:'{phone}'")
    if company_id is not None:
        clauses.append(f"company_id:{company_id}")
    if updated_since:
        clauses.append(f"updated_at:>'{updated_since}'")

    if not clauses:
        return {"error": "At least one filter parameter must be provided"}

    query = f'"{" AND ".join(clauses)}"'

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/search/contacts"
    headers = {
        "Authorization": AUTH_HEADER
    }
    params = {"query": query}

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
async def update_contact(contact_id: int, contact_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a contact in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contacts/{contact_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    data = {}
    for field, value in contact_fields.items():
        data[field] = value
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "put", url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def list_canned_responses(folder_id: int) -> list[Dict[str, Any]]:
    """List all canned responses in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_response_folders/{folder_id}/responses"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def list_canned_response_folders() -> list[Dict[str, Any]]:
    """List all canned response folders in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_response_folders"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def view_canned_response(canned_response_id: int) -> Dict[str, Any]:
    """View a canned response in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_responses/{canned_response_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def create_canned_response(canned_response_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create a canned response in Freshdesk."""
    # Validate input using Pydantic model
    try:
        validated_fields = CannedResponseCreate(**canned_response_fields)
        # Convert to dict for API request
        canned_response_data = validated_fields.model_dump(exclude_none=True)
    except Exception as e:
        return {"error": f"Validation error: {str(e)}"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_responses"
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "post", url, headers=headers, json=canned_response_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def update_canned_response(canned_response_id: int, canned_response_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a canned response in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_responses/{canned_response_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "put", url, headers=headers, json=canned_response_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def create_canned_response_folder(name: str) -> Dict[str, Any]:
    """Create a canned response folder in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_response_folders"
    headers = {
        "Authorization": AUTH_HEADER
    }
    data = {
        "name": name
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "post", url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def update_canned_response_folder(folder_id: int, name: str) -> Dict[str, Any]:
    """Update a canned response folder in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/canned_response_folders/{folder_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    data = {
        "name": name
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "put", url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def list_solution_articles(folder_id: int) -> list[Dict[str, Any]]:
    """List all solution articles in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/folders/{folder_id}/articles"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def list_solution_folders(category_id: int) -> list[Dict[str, Any]]:
    if not category_id:
        return {"error": "Category ID is required"}
    """List all solution folders in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/categories/{category_id}/folders"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def list_solution_categories() -> list[Dict[str, Any]]:
    """List all solution categories in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/categories"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def view_solution_category(category_id: int) -> Dict[str, Any]:
    """View a solution category in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/categories/{category_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def create_solution_category(category_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create a solution category in Freshdesk."""
    if not category_fields.get("name"):
        return {"error": "Name is required"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/categories"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "post", url, headers=headers, json=category_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def update_solution_category(category_id: int, category_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a solution category in Freshdesk."""
    if not category_fields.get("name"):
        return {"error": "Name is required"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/categories/{category_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "put", url, headers=headers, json=category_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def create_solution_category_folder(category_id: int, folder_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create a solution category folder in Freshdesk."""
    if not folder_fields.get("name"):
        return {"error": "Name is required"}
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/categories/{category_id}/folders"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "post", url, headers=headers, json=folder_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def view_solution_category_folder(folder_id: int) -> Dict[str, Any]:
    """View a solution category folder in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/folders/{folder_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def update_solution_category_folder(folder_id: int, folder_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a solution category folder in Freshdesk."""
    if not folder_fields.get("name"):
        return {"error": "Name is required"}
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/folders/{folder_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "put", url, headers=headers, json=folder_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
async def create_solution_article(folder_id: int, article_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create a solution article in Freshdesk."""
    if not article_fields.get("title") or not article_fields.get("status") or not article_fields.get("description"):
        return {"error": "Title, status and description are required"}
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/folders/{folder_id}/articles"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "post", url, headers=headers, json=article_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def view_solution_article(article_id: int) -> Dict[str, Any]:
    """View a solution article in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/articles/{article_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def update_solution_article(article_id: int, article_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a solution article in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/solutions/articles/{article_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "put", url, headers=headers, json=article_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def view_agent(agent_id: int) -> Dict[str, Any]:
    """View an agent in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/agents/{agent_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def create_agent(agent_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create an agent in Freshdesk."""
    # Validate mandatory fields
    if not agent_fields.get("email") or not agent_fields.get("ticket_scope"):
        return {
            "error": "Missing mandatory fields. Both 'email' and 'ticket_scope' are required."
        }
    if agent_fields.get("ticket_scope") not in [e.value for e in AgentTicketScope]:
        return {
            "error": "Invalid value for ticket_scope. Must be one of: " + ", ".join([e.name for e in AgentTicketScope])
        }

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/agents"
    headers = {
        "Authorization": AUTH_HEADER
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "post", url, headers=headers, json=agent_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"Failed to create agent: {str(e)}",
                "details": e.response.json() if e.response else None
            }

@mcp.tool()
async def update_agent(agent_id: int, agent_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update an agent in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/agents/{agent_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "put", url, headers=headers, json=agent_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def search_agents(query: str) -> list[Dict[str, Any]]:
    """Search for agents in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/agents/autocomplete?term={query}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def list_groups(page: Optional[int] = 1, per_page: Optional[int] = 30) -> list[Dict[str, Any]]:
    """List all groups in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/groups"
    headers = {
        "Authorization": AUTH_HEADER
    }
    params = {
        "page": page,
        "per_page": per_page
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def create_group(group_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create a group in Freshdesk."""
    # Validate input using Pydantic model
    try:
        validated_fields = GroupCreate(**group_fields)
        # Convert to dict for API request
        group_data = validated_fields.model_dump(exclude_none=True)
    except Exception as e:
        return {"error": f"Validation error: {str(e)}"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/groups"
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "post", url, headers=headers, json=group_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"Failed to create group: {str(e)}",
                "details": e.response.json() if e.response else None
            }

@mcp.tool()
async def view_group(group_id: int) -> Dict[str, Any]:
    """View a group in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/groups/{group_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def create_ticket_field(ticket_field_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create a ticket field in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/admin/ticket_fields"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "post", url, headers=headers, json=ticket_field_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def view_ticket_field(ticket_field_id: int) -> Dict[str, Any]:
    """View a ticket field in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/admin/ticket_fields/{ticket_field_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def update_ticket_field(ticket_field_id: int, ticket_field_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a ticket field in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/admin/ticket_fields/{ticket_field_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "put", url, headers=headers, json=ticket_field_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def update_group(group_id: int, group_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a group in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/groups/{group_id}"
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "put", url, headers=headers, json=group_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"Failed to update group: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def list_contact_fields() -> list[Dict[str, Any]]:
    """List all contact fields in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contact_fields"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def view_contact_field(contact_field_id: int) -> Dict[str, Any]:
    """View a contact field in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contact_fields/{contact_field_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def create_contact_field(contact_field_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create a contact field in Freshdesk."""
    # Validate input using Pydantic model
    try:
        validated_fields = ContactFieldCreate(**contact_field_fields)
        # Convert to dict for API request
        contact_field_data = validated_fields.model_dump(exclude_none=True)
    except Exception as e:
        return {"error": f"Validation error: {str(e)}"}
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contact_fields"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "post", url, headers=headers, json=contact_field_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def update_contact_field(contact_field_id: int, contact_field_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a contact field in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/contact_fields/{contact_field_id}"
    headers = {
        "Authorization": AUTH_HEADER
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "put", url, headers=headers, json=contact_field_fields)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_field_properties(field_name: str):
    """Get properties of a specific field by name."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/ticket_fields"
    headers = {
        "Authorization": AUTH_HEADER
    }
    actual_field_name = field_name
    if field_name == "type":
        actual_field_name = "ticket_type"
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            fields = response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %s for %s", e.response.status_code, url)
            error_detail = None
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {"error": f"API request failed: {str(e)}", "details": error_detail}
        except Exception as e:
            logger.error("Unexpected error for %s: %s", url, e)
            return {"error": f"An unexpected error occurred: {str(e)}"}
    # Filter the field by name
    matched_field = next((field for field in fields if field["name"] == actual_field_name), None)

    return matched_field

@mcp.prompt()
def create_ticket_prompt(
    subject: str,
    description: str,
    source: str,
    priority: str,
    status: str,
    email: str
) -> str:
    """Create a ticket in Freshdesk"""
    payload = {
        "subject": subject,
        "description": description,
        "source": source,
        "priority": priority,
        "status": status,
        "email": email,
    }
    return f"""
Kindly create a ticket in Freshdesk using the following payload:

{payload}

If you need to retrieve information about any fields (such as allowed values or internal keys), please use the `get_field_properties()` function.

Notes:
- The "type" field is **not** a custom field; it is a standard system field.
- The "type" field is required but should be passed as a top-level parameter, not within custom_fields.
Make sure to reference the correct keys from `get_field_properties()` when constructing the payload.
"""

@mcp.prompt()
def create_reply(
    ticket_id: int,
    reply_message: str,
) -> str:
    """Create a reply in Freshdesk"""
    payload = {
        "body": reply_message,
    }
    return f"""
Kindly create a ticket reply in Freshdesk for ticket ID {ticket_id} using the following payload:

{payload}

Notes:
- The "body" field must be in **HTML format** and should be **brief yet contextually complete**.
- When composing the "body", please **review the previous conversation** in the ticket.
- Ensure the tone and style **match the prior replies**, and that the message provides **full context** so the recipient can understand the issue without needing to re-read earlier messages.
"""

@mcp.tool()
async def list_companies(page: Optional[int] = 1, per_page: Optional[int] = 30) -> Dict[str, Any]:
    """List all companies in Freshdesk with pagination support."""
    # Validate input parameters
    if page < 1:
        return {"error": "Page number must be greater than 0"}

    if per_page < 1 or per_page > 100:
        return {"error": "Page size must be between 1 and 100"}

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/companies"

    params = {
        "page": page,
        "per_page": per_page
    }

    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers, params=params)
            response.raise_for_status()

            # Parse pagination from Link header
            link_header = response.headers.get('Link', '')
            pagination_info = parse_link_header(link_header)

            companies = response.json()

            return {
                "companies": companies,
                "pagination": {
                    "current_page": page,
                    "next_page": pagination_info.get("next"),
                    "prev_page": pagination_info.get("prev"),
                    "per_page": per_page
                }
            }

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch companies: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def view_company(company_id: int) -> Dict[str, Any]:
    """Get a company in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/companies/{company_id}"
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch company: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def search_companies(query: str) -> Dict[str, Any]:
    """Search for companies in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/companies/autocomplete"
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }
    # Use the name parameter as specified in the API
    params = {"name": query}

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to search companies: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def find_company_by_name(name: str) -> Dict[str, Any]:
    """Find a company by name in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/companies/autocomplete"
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }
    params = {"name": name}

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to find company: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
async def filter_companies(
    name: Optional[str] = None,
    domain: Optional[str] = None,
    custom_field: Optional[str] = None,
    custom_value: Optional[str] = None,
) -> Dict[str, Any]:
    """Search companies using structured filters via the Freshdesk search API.

    Args:
        name: Filter by company name.
        domain: Filter by domain name (e.g. "tucows.com") — useful for registrar lookups.
        custom_field: Custom field name to filter by (use with custom_value).
        custom_value: Value for the custom field filter.

    At least one filter parameter must be provided.
    """
    clauses = []
    if name:
        clauses.append(f"name:'{name}'")
    if domain:
        clauses.append(f"domain:'{domain}'")
    if custom_field and custom_value:
        clauses.append(f"cf_{custom_field}:'{custom_value}'")

    if not clauses:
        return {"error": "At least one filter parameter must be provided"}

    query = f'"{" AND ".join(clauses)}"'

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/search/companies"
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }
    params = {"query": query}

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to search companies: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
async def list_company_fields() -> List[Dict[str, Any]]:
    """List all company fields in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/company_fields"
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch company fields: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def view_ticket_summary(ticket_id: int) -> Dict[str, Any]:
    """Get the summary of a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/summary"
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "get", url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"message": "No summary has been created for this ticket"}
            return {"error": f"Failed to fetch ticket summary: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def update_ticket_summary(ticket_id: int, body: str) -> Dict[str, Any]:
    """Update the summary of a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/summary"
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }
    data = {
        "body": body
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "put", url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to update ticket summary: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def delete_ticket_summary(ticket_id: int) -> Dict[str, Any]:
    """Delete the summary of a ticket in Freshdesk."""
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/summary"
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await _request_with_retry(client, "delete", url, headers=headers)
            if response.status_code == 204:
                return {"success": True, "message": "Ticket summary deleted successfully"}

            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to delete ticket summary: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

def main():
    logging.info("Starting Freshdesk MCP server")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
