"""
Manual test runner for all Freshdesk MCP tools against a live API.

Usage:
    .venv/bin/python tests/manual_test.py

Set env vars before running:
    export FRESHDESK_API_KEY=<your_key>
    export FRESHDESK_DOMAIN=<your_domain>
"""
import asyncio
import os
import sys
import json
import time
import traceback

os.environ.setdefault("FRESHDESK_API_KEY", "")
os.environ.setdefault("FRESHDESK_DOMAIN", "")

from freshdesk_mcp.server import (
    get_ticket_fields,
    get_tickets,
    get_ticket,
    search_tickets,
    get_ticket_conversation,
    update_ticket,
    delete_ticket,
    create_ticket_reply,
    create_ticket_note,
    update_ticket_conversation,
    view_ticket_summary,
    update_ticket_summary,
    delete_ticket_summary,
    get_agents,
    view_agent,
    search_agents,
    create_agent,
    update_agent,
    list_contacts,
    get_contact,
    search_contacts,
    update_contact,
    list_contact_fields,
    view_contact_field,
    create_contact_field,
    update_contact_field,
    list_companies,
    view_company,
    search_companies,
    find_company_by_name,
    list_company_fields,
    list_groups,
    view_group,
    create_group,
    update_group,
    list_canned_responses,
    list_canned_response_folders,
    view_canned_response,
    create_canned_response,
    update_canned_response,
    create_canned_response_folder,
    update_canned_response_folder,
    list_solution_categories,
    view_solution_category,
    create_solution_category,
    update_solution_category,
    list_solution_folders,
    view_solution_category_folder,
    create_solution_category_folder,
    update_solution_category_folder,
    list_solution_articles,
    view_solution_article,
    create_solution_article,
    update_solution_article,
    create_ticket_field,
    view_ticket_field,
    update_ticket_field,
    get_field_properties,
)

# Also get the real create_ticket tool (prompt overwrites it in module namespace)
import freshdesk_mcp.server as _srv
create_ticket_tool = _srv.mcp._tool_manager._tools["create_ticket"].fn

PASS = 0
FAIL = 0
SKIP = 0
RESULTS = []

def pretty(data, max_len=300):
    """Truncated JSON for display."""
    s = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return s[:max_len] + "..." if len(s) > max_len else s


async def run_test(name, coro, expect_error=False):
    """Run a single test and print result."""
    global PASS, FAIL, SKIP
    try:
        result = await coro
        is_error = (
            isinstance(result, dict) and "error" in result
        ) or (
            isinstance(result, str) and "error" in result.lower()
        )

        if expect_error and is_error:
            PASS += 1
            status = "PASS (expected error)"
        elif not expect_error and not is_error:
            PASS += 1
            status = "PASS"
        elif expect_error and not is_error:
            FAIL += 1
            status = "FAIL (expected error, got success)"
        else:
            FAIL += 1
            status = "FAIL (unexpected error)"

        RESULTS.append((name, status))
        print(f"  {status:40s} {name}")
        if is_error or status.startswith("FAIL"):
            print(f"    Response: {pretty(result)}")
        return result
    except Exception as e:
        FAIL += 1
        status = "FAIL (exception)"
        RESULTS.append((name, status))
        print(f"  {status:40s} {name}")
        print(f"    {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        return None


async def main():
    global SKIP

    write_mode = "--write" in sys.argv

    print("=" * 70)
    print("FRESHDESK MCP — MANUAL TOOL TEST")
    print(f"Domain: {os.environ.get('FRESHDESK_DOMAIN')}")
    print(f"API Key: ...{os.environ.get('FRESHDESK_API_KEY', '')[-4:]}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # READ-ONLY TESTS (safe to run, no side effects)
    # ------------------------------------------------------------------
    print("\n--- READ-ONLY: Tickets ---")
    await run_test("get_ticket_fields", get_ticket_fields())
    tickets_result = await run_test("get_tickets(page=1, per_page=5)", get_tickets(page=1, per_page=5))
    await run_test("get_tickets(page=0) [expect error]", get_tickets(page=0), expect_error=True)
    await run_test("get_tickets(per_page=200) [expect error]", get_tickets(page=1, per_page=200), expect_error=True)

    # Get a real ticket ID for further tests
    ticket_id = None
    if tickets_result and isinstance(tickets_result, dict) and "tickets" in tickets_result:
        tickets = tickets_result["tickets"]
        if tickets:
            ticket_id = tickets[0]["id"]
            print(f"    (Using ticket_id={ticket_id} for further tests)")

    if ticket_id:
        await run_test(f"get_ticket({ticket_id})", get_ticket(ticket_id))
        await run_test(f"get_ticket_conversation({ticket_id})", get_ticket_conversation(ticket_id))
        await run_test(f"view_ticket_summary({ticket_id})", view_ticket_summary(ticket_id))
    else:
        SKIP += 3
        print("  SKIP — no ticket found for get_ticket / get_ticket_conversation / view_ticket_summary")

    await run_test("search_tickets('priority:>1')", search_tickets('"priority:>1"'))

    print("\n--- READ-ONLY: Ticket Fields (admin) ---")
    fields_result = await run_test("get_ticket_fields", get_ticket_fields())
    field_id = None
    if isinstance(fields_result, list) and fields_result:
        field_id = fields_result[0].get("id")
    if field_id:
        await run_test(f"view_ticket_field({field_id})", view_ticket_field(field_id))
    else:
        SKIP += 1
        print("  SKIP — no ticket field found")

    await run_test("get_field_properties('priority')", get_field_properties("priority"))
    await run_test("get_field_properties('type')", get_field_properties("type"))
    await run_test("get_field_properties('nonexistent')", get_field_properties("nonexistent_field_xyz"))

    print("\n--- READ-ONLY: Agents ---")
    agents_result = await run_test("get_agents(page=1, per_page=5)", get_agents(page=1, per_page=5))
    await run_test("get_agents(page=0) [expect error]", get_agents(page=0), expect_error=True)

    agent_id = None
    if isinstance(agents_result, list) and agents_result:
        agent_id = agents_result[0].get("id")
    if agent_id:
        await run_test(f"view_agent({agent_id})", view_agent(agent_id))
    else:
        SKIP += 1
        print("  SKIP — no agent found")

    await run_test("search_agents('a')", search_agents("a"))

    print("\n--- READ-ONLY: Contacts ---")
    contacts_result = await run_test("list_contacts(page=1, per_page=5)", list_contacts(page=1, per_page=5))

    contact_id = None
    if isinstance(contacts_result, list) and contacts_result:
        contact_id = contacts_result[0].get("id")
    if contact_id:
        await run_test(f"get_contact({contact_id})", get_contact(contact_id))
    else:
        SKIP += 1
        print("  SKIP — no contact found")

    await run_test("search_contacts('a')", search_contacts("a"))

    print("\n--- READ-ONLY: Contact Fields ---")
    cf_result = await run_test("list_contact_fields", list_contact_fields())
    cf_id = None
    if isinstance(cf_result, list) and cf_result:
        cf_id = cf_result[0].get("id")
    if cf_id:
        await run_test(f"view_contact_field({cf_id})", view_contact_field(cf_id))
    else:
        SKIP += 1
        print("  SKIP — no contact field found")

    print("\n--- READ-ONLY: Companies ---")
    companies_result = await run_test("list_companies(page=1, per_page=5)", list_companies(page=1, per_page=5))
    await run_test("list_companies(page=0) [expect error]", list_companies(page=0), expect_error=True)

    company_id = None
    if isinstance(companies_result, dict) and "companies" in companies_result:
        comps = companies_result["companies"]
        if comps:
            company_id = comps[0]["id"]
    if company_id:
        await run_test(f"view_company({company_id})", view_company(company_id))
    else:
        SKIP += 1
        print("  SKIP — no company found")

    await run_test("search_companies('a')", search_companies("a"))
    await run_test("find_company_by_name('a')", find_company_by_name("a"))
    await run_test("list_company_fields", list_company_fields())

    print("\n--- READ-ONLY: Groups ---")
    groups_result = await run_test("list_groups(page=1, per_page=5)", list_groups(page=1, per_page=5))

    group_id = None
    if isinstance(groups_result, list) and groups_result:
        group_id = groups_result[0].get("id")
    if group_id:
        await run_test(f"view_group({group_id})", view_group(group_id))
    else:
        SKIP += 1
        print("  SKIP — no group found")

    print("\n--- READ-ONLY: Canned Responses ---")
    folders_result = await run_test("list_canned_response_folders", list_canned_response_folders())

    folder_id = None
    if isinstance(folders_result, list) and folders_result:
        folder_id = folders_result[0].get("id")
    if folder_id:
        cr_result = await run_test(f"list_canned_responses({folder_id})", list_canned_responses(folder_id))
        cr_id = None
        if isinstance(cr_result, list) and cr_result:
            cr_id = cr_result[0].get("id")
        if cr_id:
            await run_test(f"view_canned_response({cr_id})", view_canned_response(cr_id))
        else:
            SKIP += 1
            print("  SKIP — no canned response found in folder")
    else:
        SKIP += 2
        print("  SKIP — no canned response folder found")

    print("\n--- READ-ONLY: Solutions (Knowledge Base) ---")
    cats_result = await run_test("list_solution_categories", list_solution_categories())

    cat_id = None
    if isinstance(cats_result, list) and cats_result:
        cat_id = cats_result[0].get("id")
    if cat_id:
        await run_test(f"view_solution_category({cat_id})", view_solution_category(cat_id))
        sol_folders = await run_test(f"list_solution_folders({cat_id})", list_solution_folders(cat_id))

        sol_folder_id = None
        if isinstance(sol_folders, list) and sol_folders:
            sol_folder_id = sol_folders[0].get("id")
        if sol_folder_id:
            await run_test(f"view_solution_category_folder({sol_folder_id})", view_solution_category_folder(sol_folder_id))
            articles = await run_test(f"list_solution_articles({sol_folder_id})", list_solution_articles(sol_folder_id))

            article_id = None
            if isinstance(articles, list) and articles:
                article_id = articles[0].get("id")
            if article_id:
                await run_test(f"view_solution_article({article_id})", view_solution_article(article_id))
            else:
                SKIP += 1
                print("  SKIP — no solution article found")
        else:
            SKIP += 3
            print("  SKIP — no solution folder found")
    else:
        SKIP += 5
        print("  SKIP — no solution categories found")

    # ------------------------------------------------------------------
    # WRITE TESTS (only when --write flag is passed)
    # ------------------------------------------------------------------
    if not write_mode:
        print("\nNOTE: Write tests were skipped. Run with --write to include them.")
        print("  Example: .venv/bin/python tests/manual_test.py --write")
    else:
        print("\n--- WRITE: Ticket Lifecycle (fully cleaned up) ---")

        created_ticket = await run_test(
            "create_ticket",
            create_ticket_tool(
                subject="[TEST] MCP Manual Test — auto-cleanup",
                description="<p>This ticket was created by manual_test.py and will be deleted.</p>",
                source=1,
                priority=1,
                status=2,
                email="mcp-test@example.com"
            )
        )

        created_ticket_id = None
        if created_ticket and isinstance(created_ticket, dict) and "id" in created_ticket:
            created_ticket_id = created_ticket["id"]
            print(f"    (Created ticket_id={created_ticket_id})")

        if created_ticket_id:
            await run_test(
                f"update_ticket({created_ticket_id}, priority=2)",
                update_ticket(created_ticket_id, {"priority": 2})
            )
            await run_test(
                f"create_ticket_reply({created_ticket_id})",
                create_ticket_reply(created_ticket_id, "<p>Test reply from MCP manual test.</p>")
            )
            await run_test(
                f"create_ticket_note({created_ticket_id})",
                create_ticket_note(created_ticket_id, "<p>Test note from MCP manual test.</p>")
            )

            conv_result = await run_test(
                f"get_ticket_conversation({created_ticket_id})",
                get_ticket_conversation(created_ticket_id)
            )

            conv_id = None
            if isinstance(conv_result, list) and conv_result:
                # Only notes can be updated (not replies) — find the note we just created
                for conv in reversed(conv_result):
                    if conv.get("private") or conv.get("source") == 2:
                        conv_id = conv.get("id")
                        break
            if conv_id:
                await run_test(
                    f"update_ticket_conversation({conv_id})",
                    update_ticket_conversation(conv_id, "<p>Updated reply from MCP manual test.</p>")
                )
            else:
                SKIP += 1
                print("  SKIP — no conversation found to update")

            await run_test(
                f"update_ticket_summary({created_ticket_id})",
                update_ticket_summary(created_ticket_id, "<p>Test summary from MCP manual test.</p>")
            )
            await run_test(
                f"view_ticket_summary({created_ticket_id})",
                view_ticket_summary(created_ticket_id)
            )
            await run_test(
                f"delete_ticket_summary({created_ticket_id})",
                delete_ticket_summary(created_ticket_id)
            )
            await run_test(
                f"delete_ticket({created_ticket_id}) [cleanup]",
                delete_ticket(created_ticket_id)
            )
        else:
            SKIP += 8
            print("  SKIP — ticket creation failed, skipping remaining ticket write tests")

        print("\n--- WRITE: Group (no delete tool — test group will remain) ---")
        ts = int(time.time())

        created_group = await run_test(
            "create_group",
            create_group({"name": f"MCP Test Group {ts}", "description": "Created by manual_test.py"})
        )
        created_group_id = None
        if created_group and isinstance(created_group, dict) and "id" in created_group:
            created_group_id = created_group["id"]
        if created_group_id:
            await run_test(
                f"update_group({created_group_id})",
                update_group(created_group_id, {"description": "Updated by manual_test.py"})
            )
        else:
            SKIP += 1
            print("  SKIP — group creation failed")

        print("\n--- WRITE: Canned Response (no delete tool — test data will remain) ---")

        cr_folder = await run_test(
            "create_canned_response_folder",
            create_canned_response_folder(f"MCP Test Folder {ts}")
        )
        cr_folder_id = None
        if cr_folder and isinstance(cr_folder, dict) and "id" in cr_folder:
            cr_folder_id = cr_folder["id"]

        if cr_folder_id:
            await run_test(
                f"update_canned_response_folder({cr_folder_id})",
                update_canned_response_folder(cr_folder_id, f"MCP Test Folder {ts} (updated)")
            )
            cr = await run_test(
                "create_canned_response",
                create_canned_response({
                    "title": f"MCP Test Response {ts}",
                    "content_html": "<p>Test canned response content.</p>",
                    "folder_id": cr_folder_id,
                    "visibility": 0
                })
            )
            cr_id = None
            if cr and isinstance(cr, dict) and "id" in cr:
                cr_id = cr["id"]
            if cr_id:
                await run_test(
                    f"update_canned_response({cr_id})",
                    update_canned_response(cr_id, {"title": "MCP Test Response (updated)"})
                )
            else:
                SKIP += 1
                print("  SKIP — canned response creation failed")
        else:
            SKIP += 3
            print("  SKIP — canned response folder creation failed")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print("=" * 70)

    if FAIL > 0:
        print("\nFailed tests:")
        for name, status in RESULTS:
            if "FAIL" in status:
                print(f"  - {name}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
