"""
Comprehensive unit tests for all Freshdesk MCP server tools.

Mocks httpx.AsyncClient so no real API calls are made.
Run with: .venv/bin/python -m pytest tests/test_all_tools.py -v
"""
import pytest
import pytest_asyncio
import json
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

# Patch env vars BEFORE importing server (they are read at module load)
import os
os.environ["FRESHDESK_API_KEY"] = "test_api_key"
os.environ["FRESHDESK_DOMAIN"] = "testdomain.freshdesk.com"

import freshdesk_mcp.server as _srv

from freshdesk_mcp.server import (
    # Utility
    parse_link_header,
    get_field_properties,
    # Tickets
    get_ticket_fields,
    get_tickets,
    # NOTE: create_ticket is imported separately below — the module-level name
    # is overwritten by the @mcp.prompt() with the same name (line 1005).
    update_ticket,
    delete_ticket,
    get_ticket,
    search_tickets,
    get_ticket_conversation,
    create_ticket_reply,
    create_ticket_note,
    update_ticket_conversation,
    # Ticket summaries
    view_ticket_summary,
    update_ticket_summary,
    delete_ticket_summary,
    # Ticket fields (admin)
    create_ticket_field,
    view_ticket_field,
    update_ticket_field,
    # Agents
    get_agents,
    view_agent,
    create_agent,
    update_agent,
    search_agents,
    # Contacts
    list_contacts,
    get_contact,
    search_contacts,
    update_contact,
    # Contact fields
    list_contact_fields,
    view_contact_field,
    create_contact_field,
    update_contact_field,
    # Companies
    list_companies,
    view_company,
    search_companies,
    find_company_by_name,
    list_company_fields,
    # Groups
    list_groups,
    create_group,
    view_group,
    update_group,
    # Canned responses
    list_canned_responses,
    list_canned_response_folders,
    view_canned_response,
    create_canned_response,
    update_canned_response,
    create_canned_response_folder,
    update_canned_response_folder,
    # Solutions
    list_solution_articles,
    list_solution_folders,
    list_solution_categories,
    view_solution_category,
    create_solution_category,
    update_solution_category,
    create_solution_category_folder,
    view_solution_category_folder,
    update_solution_category_folder,
    create_solution_article,
    view_solution_article,
    update_solution_article,
    # Pydantic models (for direct validation tests)
    GroupCreate,
    ContactFieldCreate,
    CannedResponseCreate,
    # Enums
    TicketSource,
    TicketStatus,
    TicketPriority,
    AgentTicketScope,
)

# Now that the name collision is fixed, import both directly.
from freshdesk_mcp.server import create_ticket as create_ticket_tool
from freshdesk_mcp.server import create_ticket_prompt


# ---------------------------------------------------------------------------
# Helpers to build mock httpx responses
# ---------------------------------------------------------------------------

def _mock_response(json_data=None, status_code=200, headers=None):
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.headers = headers or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    return resp


def _patch_client(response):
    """Return a patch context manager that makes httpx.AsyncClient return *response* for any HTTP method."""
    client_instance = AsyncMock()
    for method in ("get", "post", "put", "delete"):
        getattr(client_instance, method).return_value = response

    ctx_manager = AsyncMock()
    ctx_manager.__aenter__ = AsyncMock(return_value=client_instance)
    ctx_manager.__aexit__ = AsyncMock(return_value=False)

    return patch("freshdesk_mcp.server.httpx.AsyncClient", return_value=ctx_manager), client_instance


# ========================== parse_link_header ==============================

class TestParseLinkHeader:
    def test_both_next_and_prev(self):
        header = '<https://x.freshdesk.com/api/v2/tickets?page=3>; rel="next", <https://x.freshdesk.com/api/v2/tickets?page=1>; rel="prev"'
        result = parse_link_header(header)
        assert result["next"] == 3
        assert result["prev"] == 1

    def test_next_only(self):
        header = '<https://x.freshdesk.com/api/v2/tickets?page=2>; rel="next"'
        result = parse_link_header(header)
        assert result["next"] == 2
        assert result["prev"] is None

    def test_empty_string(self):
        assert parse_link_header("") == {"next": None, "prev": None}

    def test_none_value(self):
        assert parse_link_header(None) == {"next": None, "prev": None}

    def test_garbage_input(self):
        assert parse_link_header("not a link header") == {"next": None, "prev": None}


# ========================== Enum sanity ====================================

class TestEnums:
    def test_ticket_source_values(self):
        assert TicketSource.EMAIL == 1
        assert TicketSource.CHAT == 7

    def test_ticket_status_values(self):
        assert TicketStatus.OPEN == 2
        assert TicketStatus.CLOSED == 5

    def test_ticket_priority_values(self):
        assert TicketPriority.LOW == 1
        assert TicketPriority.URGENT == 4

    def test_agent_ticket_scope_values(self):
        assert AgentTicketScope.GLOBAL_ACCESS == 1
        assert AgentTicketScope.RESTRICTED_ACCESS == 3


# ========================== Pydantic model validation ======================

class TestPydanticModels:
    def test_group_create_valid(self):
        g = GroupCreate(name="Support Team")
        assert g.name == "Support Team"
        assert g.auto_ticket_assign == 0

    def test_group_create_missing_name(self):
        with pytest.raises(Exception):
            GroupCreate()

    def test_contact_field_create_valid(self):
        cf = ContactFieldCreate(
            label="Test", label_for_customers="Test", type="custom_text"
        )
        assert cf.label == "Test"

    def test_contact_field_create_invalid_type(self):
        with pytest.raises(Exception):
            ContactFieldCreate(
                label="Test", label_for_customers="Test", type="invalid_type"
            )

    def test_canned_response_create_valid(self):
        cr = CannedResponseCreate(
            title="Hello", content_html="<p>Hi</p>", folder_id=1, visibility=0
        )
        assert cr.visibility == 0

    def test_canned_response_create_invalid_visibility(self):
        with pytest.raises(Exception):
            CannedResponseCreate(
                title="Hello", content_html="<p>Hi</p>", folder_id=1, visibility=5
            )


# ========================== TICKET TOOLS ===================================

@pytest.mark.asyncio
class TestGetTicketFields:
    async def test_success(self):
        data = [{"id": 1, "name": "subject", "type": "default_subject"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await get_ticket_fields()
        assert result == {"fields": data}

@pytest.mark.asyncio
class TestGetTickets:
    async def test_success_with_pagination(self):
        tickets = [{"id": 1, "subject": "Test"}]
        resp = _mock_response(tickets, headers={"Link": '<https://x.freshdesk.com/api/v2/tickets?page=2>; rel="next"'})
        patcher, client = _patch_client(resp)
        with patcher:
            result = await get_tickets(page=1, per_page=10)
        assert result["tickets"] == tickets
        assert result["pagination"]["next_page"] == 2

    async def test_invalid_page(self):
        result = await get_tickets(page=0)
        assert "error" in result

    async def test_invalid_per_page(self):
        result = await get_tickets(page=1, per_page=200)
        assert "error" in result

@pytest.mark.asyncio
class TestCreateTicket:
    async def test_success(self):
        resp = _mock_response({"id": 100}, status_code=201)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await create_ticket_tool(
                subject="Test", description="Desc", source=1, priority=1,
                status=2, email="a@b.com"
            )
        assert "successfully" in result.lower()

    async def test_missing_email_and_requester(self):
        result = await create_ticket_tool(
            subject="Test", description="Desc", source=1, priority=1, status=2
        )
        assert "error" in result.lower()

    async def test_invalid_source(self):
        result = await create_ticket_tool(
            subject="Test", description="Desc", source=999, priority=1,
            status=2, email="a@b.com"
        )
        assert "error" in result.lower() or "invalid" in result.lower()

    async def test_string_enum_values(self):
        resp = _mock_response({"id": 101}, status_code=201)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await create_ticket_tool(
                subject="Test", description="Desc", source="1", priority="2",
                status="2", email="a@b.com"
            )
        assert "successfully" in result.lower()

    async def test_non_numeric_string(self):
        result = await create_ticket_tool(
            subject="Test", description="Desc", source="abc", priority="1",
            status="2", email="a@b.com"
        )
        assert "error" in result.lower()

    async def test_with_custom_fields(self):
        resp = _mock_response({"id": 102}, status_code=201)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await create_ticket_tool(
                subject="Test", description="Desc", source=1, priority=1,
                status=2, email="a@b.com",
                custom_fields={"cf_category": "billing"}
            )
        assert "successfully" in result.lower()

    async def test_with_additional_fields(self):
        resp = _mock_response({"id": 103}, status_code=201)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await create_ticket_tool(
                subject="Test", description="Desc", source=1, priority=1,
                status=2, email="a@b.com",
                additional_fields={"type": "Question"}
            )
        assert "successfully" in result.lower()

@pytest.mark.asyncio
class TestUpdateTicket:
    async def test_success(self):
        resp = _mock_response({"id": 1, "status": 4})
        patcher, client = _patch_client(resp)
        with patcher:
            result = await update_ticket(1, {"status": 4})
        assert result["success"] is True

    async def test_empty_fields(self):
        result = await update_ticket(1, {})
        assert result.get("error")

    async def test_with_custom_fields(self):
        resp = _mock_response({"id": 1})
        patcher, client = _patch_client(resp)
        with patcher:
            result = await update_ticket(1, {"custom_fields": {"cf_type": "Bug"}, "priority": 3})
        assert result["success"] is True

@pytest.mark.asyncio
class TestDeleteTicket:
    async def test_success_204(self):
        resp = _mock_response(None, status_code=204)
        resp.raise_for_status = MagicMock()
        patcher, client = _patch_client(resp)
        with patcher:
            result = await delete_ticket(1)
        assert result["success"] is True
        assert "deleted" in result["message"].lower()

@pytest.mark.asyncio
class TestGetTicket:
    async def test_success(self):
        data = {"id": 42, "subject": "Hello"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await get_ticket(42)
        assert result["id"] == 42

@pytest.mark.asyncio
class TestSearchTickets:
    async def test_success(self):
        data = {"total": 1, "results": [{"id": 1}]}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await search_tickets("priority:3")
        assert result["total"] == 1

@pytest.mark.asyncio
class TestGetTicketConversation:
    async def test_success(self):
        data = [{"id": 100, "body": "Hello"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await get_ticket_conversation(1)
        assert len(result) == 1

@pytest.mark.asyncio
class TestCreateTicketReply:
    async def test_success(self):
        data = {"id": 200, "body": "<p>Reply</p>"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await create_ticket_reply(1, "<p>Reply</p>")
        assert result["id"] == 200

@pytest.mark.asyncio
class TestCreateTicketNote:
    async def test_success(self):
        data = {"id": 300, "body": "Note"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await create_ticket_note(1, "Note")
        assert result["id"] == 300

@pytest.mark.asyncio
class TestUpdateTicketConversation:
    async def test_success(self):
        data = {"id": 400, "body": "Updated"}
        patcher, client = _patch_client(_mock_response(data, status_code=200))
        with patcher:
            result = await update_ticket_conversation(400, "Updated")
        assert result["id"] == 400

    async def test_failure(self):
        resp = _mock_response({"error": "not found"}, status_code=404)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await update_ticket_conversation(999, "test")
        assert "error" in result


# ========================== TICKET SUMMARY TOOLS ===========================

@pytest.mark.asyncio
class TestViewTicketSummary:
    async def test_success(self):
        data = {"ticket_id": 1, "body": "Summary text"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await view_ticket_summary(1)
        assert result["body"] == "Summary text"

@pytest.mark.asyncio
class TestUpdateTicketSummary:
    async def test_success(self):
        data = {"ticket_id": 1, "body": "New summary"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await update_ticket_summary(1, "New summary")
        assert result["body"] == "New summary"

@pytest.mark.asyncio
class TestDeleteTicketSummary:
    async def test_success_204(self):
        resp = _mock_response(None, status_code=204)
        # 204 should not call raise_for_status — the code checks before
        resp.raise_for_status = MagicMock()
        patcher, client = _patch_client(resp)
        with patcher:
            result = await delete_ticket_summary(1)
        assert result["success"] is True


# ========================== TICKET FIELD ADMIN TOOLS =======================

@pytest.mark.asyncio
class TestCreateTicketField:
    async def test_success(self):
        data = {"id": 10, "name": "cf_test"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await create_ticket_field({"label": "Test", "type": "custom_text"})
        assert result["id"] == 10

@pytest.mark.asyncio
class TestViewTicketField:
    async def test_success(self):
        data = {"id": 10, "name": "cf_test", "label": "Test"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await view_ticket_field(10)
        assert result["label"] == "Test"

@pytest.mark.asyncio
class TestUpdateTicketField:
    async def test_success(self):
        data = {"id": 10, "label": "Updated"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await update_ticket_field(10, {"label": "Updated"})
        assert result["label"] == "Updated"


# ========================== AGENT TOOLS ====================================

@pytest.mark.asyncio
class TestGetAgents:
    async def test_success(self):
        data = [{"id": 1, "contact": {"name": "Agent Smith"}}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await get_agents(page=1, per_page=10)
        assert len(result) == 1

    async def test_invalid_page(self):
        result = await get_agents(page=0)
        assert "error" in result

    async def test_invalid_per_page(self):
        result = await get_agents(per_page=200)
        assert "error" in result

@pytest.mark.asyncio
class TestViewAgent:
    async def test_success(self):
        data = {"id": 1, "contact": {"name": "Agent Smith"}}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await view_agent(1)
        assert result["id"] == 1

@pytest.mark.asyncio
class TestCreateAgent:
    async def test_success(self):
        data = {"id": 2, "contact": {"email": "new@test.com"}}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await create_agent({"email": "new@test.com", "ticket_scope": 1})
        assert result["id"] == 2

    async def test_missing_email(self):
        result = await create_agent({"ticket_scope": 1})
        assert "error" in result

    async def test_missing_ticket_scope(self):
        result = await create_agent({"email": "a@b.com"})
        assert "error" in result

    async def test_invalid_ticket_scope(self):
        result = await create_agent({"email": "a@b.com", "ticket_scope": 99})
        assert "error" in result

@pytest.mark.asyncio
class TestUpdateAgent:
    async def test_success(self):
        data = {"id": 1, "contact": {"name": "Updated"}}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await update_agent(1, {"contact": {"name": "Updated"}})
        assert result["id"] == 1

@pytest.mark.asyncio
class TestSearchAgents:
    async def test_success(self):
        data = [{"id": 1, "contact": {"name": "Agent Smith"}}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await search_agents("Smith")
        assert len(result) == 1


# ========================== CONTACT TOOLS ==================================

@pytest.mark.asyncio
class TestListContacts:
    async def test_success(self):
        data = [{"id": 1, "name": "John"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await list_contacts(page=1, per_page=10)
        assert len(result) == 1

@pytest.mark.asyncio
class TestGetContact:
    async def test_success(self):
        data = {"id": 1, "name": "John", "email": "john@test.com"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await get_contact(1)
        assert result["email"] == "john@test.com"

@pytest.mark.asyncio
class TestSearchContacts:
    async def test_success(self):
        data = [{"id": 1, "name": "John"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await search_contacts("John")
        assert len(result) == 1

@pytest.mark.asyncio
class TestUpdateContact:
    async def test_success(self):
        data = {"id": 1, "name": "Jane"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await update_contact(1, {"name": "Jane"})
        assert result["name"] == "Jane"


# ========================== CONTACT FIELD TOOLS ============================

@pytest.mark.asyncio
class TestListContactFields:
    async def test_success(self):
        data = [{"id": 1, "name": "email", "label": "Email"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await list_contact_fields()
        assert len(result) == 1

@pytest.mark.asyncio
class TestViewContactField:
    async def test_success(self):
        data = {"id": 1, "name": "email", "label": "Email"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await view_contact_field(1)
        assert result["label"] == "Email"

@pytest.mark.asyncio
class TestCreateContactField:
    async def test_success(self):
        data = {"id": 5, "name": "cf_test", "label": "Test"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await create_contact_field({
                "label": "Test",
                "label_for_customers": "Test",
                "type": "custom_text"
            })
        assert result["id"] == 5

    async def test_validation_error(self):
        result = await create_contact_field({"label": "Test"})
        assert "error" in result

@pytest.mark.asyncio
class TestUpdateContactField:
    async def test_success(self):
        data = {"id": 5, "label": "Updated"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await update_contact_field(5, {"label": "Updated"})
        assert result["label"] == "Updated"


# ========================== COMPANY TOOLS ==================================

@pytest.mark.asyncio
class TestListCompanies:
    async def test_success_with_pagination(self):
        companies = [{"id": 1, "name": "Acme"}]
        resp = _mock_response(companies, headers={"Link": '<https://x.freshdesk.com/api/v2/companies?page=2>; rel="next"'})
        patcher, client = _patch_client(resp)
        with patcher:
            result = await list_companies(page=1, per_page=10)
        assert result["companies"] == companies
        assert result["pagination"]["next_page"] == 2

    async def test_invalid_page(self):
        result = await list_companies(page=0)
        assert "error" in result

    async def test_invalid_per_page(self):
        result = await list_companies(per_page=101)
        assert "error" in result

@pytest.mark.asyncio
class TestViewCompany:
    async def test_success(self):
        data = {"id": 1, "name": "Acme"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await view_company(1)
        assert result["name"] == "Acme"

@pytest.mark.asyncio
class TestSearchCompanies:
    async def test_success(self):
        data = [{"id": 1, "name": "Acme"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await search_companies("Acme")
        assert len(result) == 1

@pytest.mark.asyncio
class TestFindCompanyByName:
    async def test_success(self):
        data = [{"id": 1, "name": "Acme Corp"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await find_company_by_name("Acme Corp")
        assert result[0]["name"] == "Acme Corp"

@pytest.mark.asyncio
class TestListCompanyFields:
    async def test_success(self):
        data = [{"id": 1, "name": "name", "label": "Company Name"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await list_company_fields()
        assert result[0]["label"] == "Company Name"


# ========================== GROUP TOOLS ====================================

@pytest.mark.asyncio
class TestListGroups:
    async def test_success(self):
        data = [{"id": 1, "name": "Support"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await list_groups(page=1, per_page=10)
        assert len(result) == 1

@pytest.mark.asyncio
class TestCreateGroup:
    async def test_success(self):
        data = {"id": 1, "name": "New Group"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await create_group({"name": "New Group"})
        assert result["id"] == 1

    async def test_validation_error_missing_name(self):
        result = await create_group({})
        assert "error" in result

@pytest.mark.asyncio
class TestViewGroup:
    async def test_success(self):
        data = {"id": 1, "name": "Support"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await view_group(1)
        assert result["name"] == "Support"

@pytest.mark.asyncio
class TestUpdateGroup:
    async def test_success(self):
        data = {"id": 1, "name": "Updated Group"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await update_group(1, {"name": "Updated Group"})
        assert result["id"] == 1

    async def test_validation_error(self):
        result = await update_group(1, {})
        assert "error" in result


# ========================== CANNED RESPONSE TOOLS ==========================

@pytest.mark.asyncio
class TestListCannedResponses:
    async def test_success(self):
        data = [{"id": 1, "title": "Greeting"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await list_canned_responses(100)
        assert len(result) == 1
        assert result[0]["title"] == "Greeting"

@pytest.mark.asyncio
class TestListCannedResponseFolders:
    async def test_success(self):
        data = [{"id": 100, "name": "Default"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await list_canned_response_folders()
        assert result[0]["name"] == "Default"

@pytest.mark.asyncio
class TestViewCannedResponse:
    async def test_success(self):
        data = {"id": 1, "title": "Greeting", "content_html": "<p>Hi</p>"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await view_canned_response(1)
        assert result["title"] == "Greeting"

@pytest.mark.asyncio
class TestCreateCannedResponse:
    async def test_success(self):
        data = {"id": 2, "title": "New"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await create_canned_response({
                "title": "New", "content_html": "<p>Body</p>",
                "folder_id": 100, "visibility": 0
            })
        assert result["id"] == 2

    async def test_validation_error(self):
        result = await create_canned_response({"title": "Missing fields"})
        assert "error" in result

@pytest.mark.asyncio
class TestUpdateCannedResponse:
    async def test_success(self):
        data = {"id": 1, "title": "Updated"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await update_canned_response(1, {"title": "Updated"})
        assert result["title"] == "Updated"

@pytest.mark.asyncio
class TestCreateCannedResponseFolder:
    async def test_success(self):
        data = {"id": 200, "name": "New Folder"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await create_canned_response_folder("New Folder")
        assert result["name"] == "New Folder"

@pytest.mark.asyncio
class TestUpdateCannedResponseFolder:
    async def test_success(self):
        data = {"id": 200, "name": "Renamed"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await update_canned_response_folder(200, "Renamed")
        assert result["name"] == "Renamed"


# ========================== SOLUTION TOOLS =================================

@pytest.mark.asyncio
class TestListSolutionCategories:
    async def test_success(self):
        data = [{"id": 1, "name": "General"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await list_solution_categories()
        assert result[0]["name"] == "General"

@pytest.mark.asyncio
class TestViewSolutionCategory:
    async def test_success(self):
        data = {"id": 1, "name": "General"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await view_solution_category(1)
        assert result["id"] == 1

@pytest.mark.asyncio
class TestCreateSolutionCategory:
    async def test_success(self):
        data = {"id": 2, "name": "New Category"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await create_solution_category({"name": "New Category"})
        assert result["id"] == 2

    async def test_missing_name(self):
        result = await create_solution_category({"description": "No name"})
        assert "error" in result

@pytest.mark.asyncio
class TestUpdateSolutionCategory:
    async def test_success(self):
        data = {"id": 1, "name": "Updated"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await update_solution_category(1, {"name": "Updated"})
        assert result["name"] == "Updated"

    async def test_missing_name(self):
        result = await update_solution_category(1, {"description": "No name"})
        assert "error" in result

@pytest.mark.asyncio
class TestListSolutionFolders:
    async def test_success(self):
        data = [{"id": 10, "name": "FAQ"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await list_solution_folders(1)
        assert result[0]["name"] == "FAQ"

@pytest.mark.asyncio
class TestViewSolutionCategoryFolder:
    async def test_success(self):
        data = {"id": 10, "name": "FAQ"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await view_solution_category_folder(10)
        assert result["id"] == 10

@pytest.mark.asyncio
class TestCreateSolutionCategoryFolder:
    async def test_success(self):
        data = {"id": 11, "name": "New Folder"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await create_solution_category_folder(1, {"name": "New Folder"})
        assert result["id"] == 11

    async def test_missing_name(self):
        result = await create_solution_category_folder(1, {"visibility": 1})
        assert "error" in result

@pytest.mark.asyncio
class TestUpdateSolutionCategoryFolder:
    async def test_success(self):
        data = {"id": 10, "name": "Renamed"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await update_solution_category_folder(10, {"name": "Renamed"})
        assert result["name"] == "Renamed"

    async def test_missing_name(self):
        result = await update_solution_category_folder(10, {"visibility": 1})
        assert "error" in result

@pytest.mark.asyncio
class TestListSolutionArticles:
    async def test_success(self):
        data = [{"id": 100, "title": "How to reset password"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await list_solution_articles(10)
        assert result[0]["title"] == "How to reset password"

@pytest.mark.asyncio
class TestViewSolutionArticle:
    async def test_success(self):
        data = {"id": 100, "title": "How to reset password"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await view_solution_article(100)
        assert result["id"] == 100

@pytest.mark.asyncio
class TestCreateSolutionArticle:
    async def test_success(self):
        data = {"id": 101, "title": "New Article"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await create_solution_article(10, {
                "title": "New Article", "description": "Body", "status": 1
            })
        assert result["id"] == 101

    async def test_missing_required_fields(self):
        result = await create_solution_article(10, {"title": "Only title"})
        assert "error" in result

@pytest.mark.asyncio
class TestUpdateSolutionArticle:
    async def test_success(self):
        data = {"id": 100, "title": "Updated Article"}
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await update_solution_article(100, {"title": "Updated Article"})
        assert result["title"] == "Updated Article"


# ========================== UTILITY TOOLS ==================================

@pytest.mark.asyncio
class TestGetFieldProperties:
    async def test_success(self):
        data = [
            {"name": "subject", "label": "Subject"},
            {"name": "priority", "label": "Priority"},
        ]
        resp = _mock_response(data)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await get_field_properties("priority")
        assert result["name"] == "priority"

    async def test_field_not_found(self):
        data = [{"name": "subject", "label": "Subject"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await get_field_properties("nonexistent")
        assert result is None

    async def test_type_field_maps_to_ticket_type(self):
        data = [{"name": "ticket_type", "label": "Type"}]
        patcher, client = _patch_client(_mock_response(data))
        with patcher:
            result = await get_field_properties("type")
        assert result["name"] == "ticket_type"


# ========================== HTTP ERROR HANDLING ============================

@pytest.mark.asyncio
class TestHTTPErrorHandling:
    """Test tools that DO have error handling return proper error dicts on failure."""

    async def test_get_tickets_http_error(self):
        resp = _mock_response({"errors": ["bad"]}, status_code=500)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await get_tickets(page=1, per_page=10)
        assert "error" in result

    async def test_create_ticket_400_validation(self):
        resp = _mock_response({"errors": [{"field": "email", "message": "required"}]}, status_code=400)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="400 Bad Request", request=MagicMock(), response=resp
        )
        patcher, client = _patch_client(resp)
        with patcher:
            result = await create_ticket_tool(
                subject="T", description="D", source=1, priority=1,
                status=2, email="a@b.com"
            )
        assert "error" in result.lower() or "validation" in result.lower()

    async def test_update_ticket_http_error(self):
        resp = _mock_response({"errors": ["forbidden"]}, status_code=403)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await update_ticket(1, {"status": 5})
        assert result["success"] is False

    async def test_list_companies_http_error(self):
        resp = _mock_response(None, status_code=500)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await list_companies(page=1, per_page=10)
        assert "error" in result

    async def test_view_company_http_error(self):
        resp = _mock_response(None, status_code=404)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await view_company(99999)
        assert "error" in result

    async def test_create_agent_http_error(self):
        resp = _mock_response({"errors": ["conflict"]}, status_code=409)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await create_agent({"email": "a@b.com", "ticket_scope": 1})
        assert "error" in result

    async def test_create_group_http_error(self):
        resp = _mock_response({"errors": ["bad"]}, status_code=400)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await create_group({"name": "Test"})
        assert "error" in result

    async def test_update_group_http_error(self):
        resp = _mock_response({"errors": ["bad"]}, status_code=400)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await update_group(1, {"name": "Test"})
        assert "error" in result

    async def test_view_ticket_summary_404_no_summary(self):
        """Bug #17: 404 means no summary exists, not a real error."""
        resp = _mock_response(None, status_code=404)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await view_ticket_summary(99999)
        assert "message" in result
        assert "no summary" in result["message"].lower()

    async def test_view_ticket_summary_http_error(self):
        """Non-404 errors should still return a generic error."""
        resp = _mock_response(None, status_code=500)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await view_ticket_summary(99999)
        assert "error" in result

    async def test_update_ticket_summary_http_error(self):
        resp = _mock_response(None, status_code=403)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await update_ticket_summary(1, "text")
        assert "error" in result

    async def test_delete_ticket_summary_http_error(self):
        resp = _mock_response(None, status_code=500)
        patcher, client = _patch_client(resp)
        with patcher:
            result = await delete_ticket_summary(1)
        assert "error" in result


# ========================== MCP PROMPTS ====================================

class TestMCPPrompts:
    """Test the @mcp.prompt() functions (synchronous, return template strings)."""

    def test_create_ticket_prompt_returns_string(self):
        result = create_ticket_prompt(
            subject="Test", description="Desc", source="1",
            priority="2", status="2", email="a@b.com"
        )
        assert isinstance(result, str)
        assert "create a ticket" in result.lower()
        assert "get_field_properties" in result

    def test_create_reply_prompt_returns_string(self):
        from freshdesk_mcp.server import create_reply
        result = create_reply(ticket_id=42, reply_message="Thanks!")
        assert isinstance(result, str)
        assert "42" in result
        assert "HTML format" in result


# ========================== NAME COLLISION FIX ==============================

class TestNameCollisionFixed:
    """Verify the name collision between prompt and tool is resolved.
    The prompt is now create_ticket_prompt, the tool remains create_ticket."""

    def test_create_ticket_tool_is_async(self):
        import inspect
        assert inspect.iscoroutinefunction(create_ticket_tool)

    def test_create_ticket_prompt_is_sync(self):
        import inspect
        assert not inspect.iscoroutinefunction(create_ticket_prompt)

    def test_no_name_collision(self):
        # Both should be importable with distinct names
        assert create_ticket_tool is not create_ticket_prompt


# ========================== RATE LIMIT RETRY ==================================

@pytest.mark.asyncio
class TestRateLimitRetry:
    """Bug #4: Verify _request_with_retry handles 429 responses."""

    async def test_retries_on_429_then_succeeds(self):
        from freshdesk_mcp.server import _request_with_retry

        rate_limited_resp = _mock_response(None, status_code=429, headers={"Retry-After": "0"})
        success_resp = _mock_response({"id": 1}, status_code=200)

        client = AsyncMock()
        client.get = AsyncMock(side_effect=[rate_limited_resp, success_resp])

        result = await _request_with_retry(client, "get", "https://example.com", headers={})
        assert result.status_code == 200
        assert client.get.call_count == 2

    async def test_gives_up_after_max_retries(self):
        from freshdesk_mcp.server import _request_with_retry, MAX_RETRIES

        rate_limited_resp = _mock_response(None, status_code=429, headers={"Retry-After": "0"})
        client = AsyncMock()
        client.get = AsyncMock(return_value=rate_limited_resp)

        result = await _request_with_retry(client, "get", "https://example.com", headers={})
        assert result.status_code == 429
        assert client.get.call_count == MAX_RETRIES + 1

    async def test_no_retry_on_non_429(self):
        from freshdesk_mcp.server import _request_with_retry

        error_resp = _mock_response({"error": "bad"}, status_code=500)
        client = AsyncMock()
        client.get = AsyncMock(return_value=error_resp)

        result = await _request_with_retry(client, "get", "https://example.com", headers={})
        assert result.status_code == 500
        assert client.get.call_count == 1
