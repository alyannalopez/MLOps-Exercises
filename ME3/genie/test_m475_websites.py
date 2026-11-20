#!/usr/bin/env python3
"""M4.75: Website Service and GitHub Service Tests.

Tests cover:
  1. Project creation and validation
  2. File CRUD operations
  3. Path security (traversal, symlinks, URL encoding, null bytes, cross-project)
  4. Preview server start/stop/port conflict
  5. GitHub configuration checks (without real credentials)
  6. GitHub token masking
  7. Publishing without confirmation
  8. Integration with M4.5 task manager

Run:
  python genie/test_m475_websites.py
"""
from __future__ import annotations

import asyncio
import http.client
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from genie.website_service import (
    WebsiteService,
    validate_project_name,
    resolve_and_validate,
    sanitize_filename,
    PreviewServer,
    WORKSPACE_ROOT,
)
from genie.github_service import (
    GitHubService,
    GitHubConfig,
)

PASS = 0
FAIL = 0
TOTAL = 0

SYNC_TESTS = []
ASYNC_TESTS = []


def test(name: str):
    """Decorator to register sync tests."""
    def decorator(fn):
        SYNC_TESTS.append((name, fn))
        return fn
    return decorator


def async_test(name: str):
    """Decorator to register async tests."""
    def decorator(fn):
        ASYNC_TESTS.append((name, fn))
        return fn
    return decorator


# =========================================================================
# Helpers
# =========================================================================

def _tmp_workspace() -> Path:
    """Create a temporary workspace directory."""
    d = tempfile.mkdtemp(prefix="genie_web_test_")
    return Path(d)


def _free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# =========================================================================
# Section 1: Project Name Validation
# =========================================================================

@test("Valid project names")
def test_valid_names():
    assert validate_project_name("my-site") == "my-site"
    assert validate_project_name("Portfolio2024") == "portfolio2024"
    assert validate_project_name("my_project") == "my_project"
    assert validate_project_name("A") == "a"
    assert validate_project_name("  My Site  ") == "my-site"
    assert validate_project_name("test123") == "test123"


@test("Invalid project names")
def test_invalid_names():
    assert validate_project_name("") is None
    assert validate_project_name("   ") is None
    assert validate_project_name("../etc/passwd") is None
    assert validate_project_name("a/b") is None
    assert validate_project_name("a\\b") is None
    assert validate_project_name("a\x00b") is None
    assert validate_project_name("a" * 200) is None  # Too long
    assert validate_project_name("-start") is None  # Must start with alnum
    assert validate_project_name("_start") is None  # Must start with alnum


# =========================================================================
# Section 2: Path Security
# =========================================================================

@test("Path traversal blocked")
def test_path_traversal():
    ws = _tmp_workspace()
    try:
        proj = ws / "test-proj"
        proj.mkdir()

        # Direct traversal
        assert resolve_and_validate(proj, "../other") is None
        assert resolve_and_validate(proj, "../../etc/passwd") is None
        assert resolve_and_validate(proj, "sub/../../escape") is None
        assert resolve_and_validate(proj, "./../../../etc/hosts") is None

        # Valid paths
        assert resolve_and_validate(proj, "index.html") is not None
        assert resolve_and_validate(proj, "css/style.css") is not None
        assert resolve_and_validate(proj, "deep/nested/file.js") is not None
    finally:
        shutil.rmtree(ws, ignore_errors=True)


@test("URL-encoded traversal blocked")
def test_url_encoded_traversal():
    ws = _tmp_workspace()
    try:
        proj = ws / "test-proj"
        proj.mkdir()

        # Single encoding
        assert resolve_and_validate(proj, "%2e%2e/etc/passwd") is None
        assert resolve_and_validate(proj, "css/%2e%2e/%2e%2e/etc/passwd") is None

        # Mixed encoding
        assert resolve_and_validate(proj, "..%2f..%2fetc%2fpasswd") is None
    finally:
        shutil.rmtree(ws, ignore_errors=True)


@test("Null byte injection blocked")
def test_null_bytes():
    ws = _tmp_workspace()
    try:
        proj = ws / "test-proj"
        proj.mkdir()

        assert resolve_and_validate(proj, "index.html\x00.jpg") is None
        assert resolve_and_validate(proj, "\x00") is None
        assert resolve_and_validate(proj, "") is None
    finally:
        shutil.rmtree(ws, ignore_errors=True)


@test("Symlink escape blocked")
def test_symlink_escape():
    ws = _tmp_workspace()
    try:
        proj = ws / "test-proj"
        proj.mkdir()

        # Create a secret file outside the project
        secret = ws / "secret.txt"
        secret.write_text("TOP SECRET")

        # Create a symlink inside the project pointing outside
        leaky_link = proj / "leaky.txt"
        try:
            leaky_link.symlink_to(secret)
        except OSError:
            # Some systems don't support symlinks — skip
            print("    (symlinks not supported on this system, skipping)")
            return

        # The symlink itself resolves to a path outside the project
        resolved = resolve_and_validate(proj, "leaky.txt")
        assert resolved is None, f"Symlink escape should be blocked, got {resolved}"

        # But a regular file inside the project is fine
        regular = proj / "regular.txt"
        regular.write_text("safe content")
        assert resolve_and_validate(proj, "regular.txt") is not None
    finally:
        shutil.rmtree(ws, ignore_errors=True)


@test("Cross-project access blocked")
def test_cross_project_access():
    ws = _tmp_workspace()
    try:
        proj_a = ws / "project-a"
        proj_a.mkdir()
        proj_b = ws / "project-b"
        proj_b.mkdir()

        secret = proj_b / "secret.html"
        secret.write_text("<h1>Project B secret</h1>")

        # Trying to access project-b's files from project-a
        assert resolve_and_validate(proj_a, "../project-b/secret.html") is None
    finally:
        shutil.rmtree(ws, ignore_errors=True)


@test("Absolute path blocked")
def test_absolute_path():
    ws = _tmp_workspace()
    try:
        proj = ws / "test-proj"
        proj.mkdir()

        assert resolve_and_validate(proj, "/etc/passwd") is None
        assert resolve_and_validate(proj, "/tmp/anything") is None
    finally:
        shutil.rmtree(ws, ignore_errors=True)


@test("Filename sanitization")
def test_sanitize_filename():
    assert sanitize_filename("normal.html") == "normal.html"
    assert sanitize_filename("file\x00.html") == "file.html"
    assert sanitize_filename("") == ""
    # Sanitization removes null bytes and strips
    assert "\x00" not in sanitize_filename("evil\x00file.html")


# =========================================================================
# Section 3: Project Operations
# =========================================================================

@test("Create project")
def test_create_project():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)

    result = svc.create_project("my-portfolio")
    assert result.success, result.message
    assert (ws / "my-portfolio" / "index.html").exists()
    assert "Welcome to My Portfolio" in (ws / "my-portfolio" / "index.html").read_text()


@test("Create duplicate project rejected")
def test_duplicate_project():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)

    svc.create_project("dup-test")
    result = svc.create_project("dup-test")
    assert not result.success
    assert "already exists" in result.message


@test("Overwrite existing project")
def test_overwrite_project():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)

    svc.create_project("overwrite-me")
    (ws / "overwrite-me" / "extra.txt").write_text("old content")

    result = svc.create_project("overwrite-me", overwrite=True)
    assert result.success
    assert not (ws / "overwrite-me" / "extra.txt").exists()
    assert (ws / "overwrite-me" / "index.html").exists()


@test("List projects")
def test_list_projects():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)

    svc.create_project("alpha")
    svc.create_project("beta")

    result = svc.list_projects()
    assert result.success
    assert "alpha" in result.message
    assert "beta" in result.message


@test("Get project status")
def test_project_status():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)

    svc.create_project("status-test")
    (ws / "status-test" / "page2.html").write_text("<h2>Page 2</h2>")

    result = svc.get_project_status("status-test")
    assert result.success
    assert "2" in result.message  # 2 files
    assert "index.html" in result.message


# =========================================================================
# Section 4: File Operations
# =========================================================================

@test("Create file in project")
def test_create_file():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)
    svc.create_project("file-test")

    result = svc.create_file("file-test", "css/style.css", "body { margin: 0; }")
    assert result.success
    assert (ws / "file-test" / "css" / "style.css").read_text() == "body { margin: 0; }"


@test("Create file with traversal path rejected")
def test_create_file_traversal():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)
    svc.create_project("secure-test")

    result = svc.create_file("secure-test", "../../../etc/evil.txt", "malicious")
    assert not result.success


@test("Create duplicate file rejected")
def test_create_duplicate_file():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)
    svc.create_project("dup-file-test")

    svc.create_file("dup-file-test", "test.txt", "first")
    result = svc.create_file("dup-file-test", "test.txt", "second")
    assert not result.success
    assert "already exists" in result.message


@test("Read file")
def test_read_file():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)
    svc.create_project("read-test")
    svc.create_file("read-test", "hello.txt", "Hello, World!")

    result = svc.read_file("read-test", "hello.txt")
    assert result.success
    assert result.content == "Hello, World!"


@test("Read nonexistent file")
def test_read_nonexistent():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)
    svc.create_project("read-test")

    result = svc.read_file("read-test", "nope.txt")
    assert not result.success


@test("Edit file")
def test_edit_file():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)
    svc.create_project("edit-test")
    svc.create_file("edit-test", "app.js", "console.log('v1');")

    result = svc.edit_file("edit-test", "app.js", "console.log('v2');", confirm_overwrite=True)
    assert result.success
    assert (ws / "edit-test" / "app.js").read_text() == "console.log('v2');"


@test("Edit without confirmation rejected")
def test_edit_no_confirm():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)
    svc.create_project("edit-test")
    svc.create_file("edit-test", "app.js", "v1")

    result = svc.edit_file("edit-test", "app.js", "v2", confirm_overwrite=False)
    assert not result.success
    assert "not confirmed" in result.message


@test("List files")
def test_list_files():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)
    svc.create_project("list-test")
    svc.create_file("list-test", "css/style.css", "body{}")
    svc.create_file("list-test", "js/app.js", "// app")

    result = svc.list_files("list-test")
    assert result.success
    assert "index.html" in result.message
    assert "css/style.css" in result.message
    assert "js/app.js" in result.message


# =========================================================================
# Section 5: Preview Server
# =========================================================================

@async_test("Preview server starts and serves files")
async def test_preview_server():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)
    svc.create_project("preview-test")
    (ws / "preview-test" / "index.html").write_text("<h1>Hello</h1>")

    result = await svc.start_preview("preview-test", preferred_port=_free_port())
    assert result.success
    assert "http://127.0.0.1:" in result.message

    # Extract port from URL
    url = result.file_path
    port = int(url.split(":")[-1])

    # Give the server a moment to start
    await asyncio.sleep(0.3)

    # Test HTTP access
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/index.html")
    resp = conn.getresponse()
    assert resp.status == 200
    body = resp.read().decode()
    assert "<h1>Hello</h1>" in body
    conn.close()

    # Clean up
    await svc.stop_preview(port)


@async_test("Preview server blocks traversal requests")
async def test_preview_traversal():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)
    svc.create_project("secure-preview")
    (ws / "secure-preview" / "index.html").write_text("<h1>Safe</h1>")

    result = await svc.start_preview("secure-preview", preferred_port=_free_port())
    assert result.success
    port = int(result.file_path.split(":")[-1])
    await asyncio.sleep(0.3)

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)

    # Try traversal
    conn.request("GET", "/../etc/passwd")
    resp = conn.getresponse()
    assert resp.status in (400, 403, 404), f"Traversal should be blocked, got {resp.status}"
    resp.read()

    # Try URL-encoded traversal
    conn.request("GET", "/%2e%2e/etc/passwd")
    resp = conn.getresponse()
    assert resp.status in (400, 403, 404), f"Encoded traversal should be blocked, got {resp.status}"
    resp.read()

    # Valid request should work
    conn.request("GET", "/index.html")
    resp = conn.getresponse()
    assert resp.status == 200
    resp.read()

    conn.close()
    await svc.stop_preview(port)


@async_test("Preview server port conflict handling")
async def test_preview_port_conflict():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)
    svc.create_project("port-test")

    # Occupy a port
    occupied_port = _free_port()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", occupied_port))
    listener.listen(1)

    try:
        result = await svc.start_preview("port-test", preferred_port=occupied_port)
        assert result.success
        # Should have auto-incremented to a different port
        used_port = int(result.file_path.split(":")[-1])
        assert used_port != occupied_port, "Should have auto-incremented past occupied port"
        await svc.stop_preview(used_port)
    finally:
        listener.close()


@async_test("Stop preview server")
async def test_stop_preview():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)
    svc.create_project("stop-test")

    result = await svc.start_preview("stop-test", preferred_port=_free_port())
    port = int(result.file_path.split(":")[-1])

    stop_result = await svc.stop_preview(port)
    assert stop_result.success

    # Stopping again should fail
    stop_result2 = await svc.stop_preview(port)
    assert not stop_result2.success


# =========================================================================
# Section 6: GitHub Configuration (no real credentials)
# =========================================================================

@test("GitHub config not configured")
def test_github_not_configured():
    cfg = GitHubConfig()
    svc = GitHubService(cfg)
    info = svc.check_configuration()
    assert not info["configured"]
    assert "GITHUB_TOKEN" in info["missing"]
    assert "GITHUB_USERNAME" in info["missing"]
    assert "GITHUB_PAGES_REPOSITORY" in info["missing"]


@test("GitHub config partially configured")
def test_github_partial_config():
    cfg = GitHubConfig(token="fake_token_12345")
    svc = GitHubService(cfg)
    info = svc.check_configuration()
    assert not info["configured"]
    assert "GITHUB_USERNAME" in info["missing"]


@test("GitHub config fully configured")
def test_github_full_config():
    cfg = GitHubConfig(token="fake_token_abc123", username="testuser", pages_repository="test-pages")
    svc = GitHubService(cfg)
    info = svc.check_configuration()
    assert info["configured"]
    assert info["username"] == "testuser"
    assert info["repository"] == "test-pages"


@test("GitHub token masking")
def test_token_masking():
    cfg = GitHubConfig(token="ghp_1234567890abcdef", username="user", pages_repository="repo")
    svc = GitHubService(cfg)

    # Mask token in text
    masked = svc._mask_token("Error: auth failed with ghp_1234567890abcdef")
    assert "ghp_1234567890abcdef" not in masked
    assert "***" in masked

    # Masked token property
    assert cfg.masked_token != cfg.token
    assert "..." in cfg.masked_token


@async_test("Publish without confirmation rejected")
async def test_publish_no_confirm():
    cfg = GitHubConfig(token="fake", username="user", pages_repository="repo")
    svc = GitHubService(cfg)

    result = await svc.publish_website(
        project_root=Path("/tmp/nonexistent-project"),
        confirm_publish=False,
    )
    assert not result.success
    assert "confirmation" in result.message.lower()


@async_test("Publish without configuration rejected")
async def test_publish_no_config():
    cfg = GitHubConfig()
    svc = GitHubService(cfg)

    result = await svc.publish_website(
        project_root=Path("/tmp/something"),
        confirm_publish=True,
    )
    assert not result.success
    assert "not configured" in result.message.lower()


@async_test("Check repo without configuration returns None")
async def test_check_repo_no_config():
    cfg = GitHubConfig()
    svc = GitHubService(cfg)
    result = await svc.check_repository()
    assert result is None


@async_test("Check deployment status without configuration")
async def test_deployment_status_no_config():
    cfg = GitHubConfig()
    svc = GitHubService(cfg)
    result = await svc.get_deployment_status()
    assert not result["success"]


# =========================================================================
# Section 7: Integration with M4.5 Task Manager
# =========================================================================

@async_test("Website creation as background task")
async def test_website_as_background_task():
    """Verify that website operations can run as background tasks."""
    from genie.task_manager import get_task_manager, TaskManager

    # Reset task manager for clean test
    TaskManager._instance = None
    tm = get_task_manager()

    ws = _tmp_workspace()

    async def website_task():
        """Simulate website generation as a background task."""
        svc = WebsiteService(workspace_root=ws)
        result = svc.create_project("bg-website")
        if not result.success:
            return f"Failed to create project: {result.message}"

        await asyncio.sleep(0.1)  # Simulate work
        svc.create_file("bg-website", "css/style.css", "body { background: #fff; }")
        await asyncio.sleep(0.1)
        svc.create_file("bg-website", "js/app.js", "console.log('loaded');")
        return f"Website 'bg-website' created at {ws}/bg-website/"

    task_id = await tm.create_task("Create bg-website", website_task())

    # Wait for completion
    for _ in range(30):
        await asyncio.sleep(0.2)
        task = tm.get_task(task_id)
        if task and task.state.name in ("COMPLETED", "FAILED", "CANCELLED"):
            break

    task = tm.get_task(task_id)
    assert task is not None
    assert task.state.name == "COMPLETED", f"Expected COMPLETED, got {task.state.name} ({task.state.value})"
    assert task.result is not None
    assert "bg-website" in task.result

    # Verify files were created
    assert (ws / "bg-website" / "index.html").exists()
    assert (ws / "bg-website" / "css" / "style.css").exists()
    assert (ws / "bg-website" / "js" / "app.js").exists()


@async_test("Multiple concurrent website tasks")
async def test_concurrent_website_tasks():
    """Two website projects created concurrently should not interfere."""
    from genie.task_manager import get_task_manager, TaskManager

    TaskManager._instance = None
    tm = get_task_manager()

    ws = _tmp_workspace()

    async def make_site(name: str):
        svc = WebsiteService(workspace_root=ws)
        svc.create_project(name)
        svc.create_file(name, "info.txt", f"This is {name}")
        return f"Created {name}"

    id1 = await tm.create_task("Create site-alpha", make_site("site-alpha"))
    id2 = await tm.create_task("Create site-beta", make_site("site-beta"))

    # Wait for both
    for _ in range(30):
        await asyncio.sleep(0.2)
        t1 = tm.get_task(id1)
        t2 = tm.get_task(id2)
        if t1 and t2 and t1.state.name == "COMPLETED" and t2.state.name == "COMPLETED":
            break

    assert tm.get_task(id1).state.name == "COMPLETED"
    assert tm.get_task(id2).state.name == "COMPLETED"

    # Verify results are not mixed up
    assert "site-alpha" in tm.get_task(id1).result
    assert "site-beta" in tm.get_task(id2).result


# =========================================================================
# Section 8: Edge Cases
# =========================================================================

@test("Nonexistent project operations")
def test_nonexistent_project_ops():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)

    assert not svc.read_file("nope", "x.html").success
    assert not svc.edit_file("nope", "x.html", "y", confirm_overwrite=True).success
    assert not svc.list_files("nope").success
    assert not svc.get_project_status("nope").success


@test("File size limit")
def test_file_size_limit():
    ws = _tmp_workspace()
    svc = WebsiteService(workspace_root=ws)
    svc.create_project("big-test")

    # Create a file larger than 5MB limit
    big_content = "x" * (5 * 1024 * 1024 + 1)
    result = svc.create_file("big-test", "big.txt", big_content)
    assert not result.success
    assert "exceeds" in result.message.lower() or "maximum" in result.message.lower()


@test("Repeated URL encoding")
def test_repeated_encoding():
    ws = _tmp_workspace()
    try:
        proj = ws / "enc-test"
        proj.mkdir()

        # After URL decoding, the path should still be validated
        # %252e decodes to %2e (literal percent-e), not ..
        # So this is actually a valid filename attempt that resolves safely
        result = resolve_and_validate(proj, "%252e%252e/etc/passwd")
        # After decoding: %2e%2e/etc/passwd → this is within the project as a weird filename
        # But it should still be within the project
        if result is not None:
            # If it resolved, it must be within the project
            assert str(result).startswith(str(proj))
    finally:
        shutil.rmtree(ws, ignore_errors=True)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("M4.75: Website & GitHub Service Tests")
    print("=" * 50)
    print()

    def run_test(name, fn):
        global PASS, FAIL, TOTAL
        TOTAL += 1
        try:
            fn()
            PASS += 1
            print(f"  Test {TOTAL}: {name} ............ PASSED")
        except Exception as e:
            FAIL += 1
            print(f"  Test {TOTAL}: {name} ............ FAILED: {e}")

    async def run_async_test(name, fn):
        global PASS, FAIL, TOTAL
        TOTAL += 1
        try:
            await fn()
            PASS += 1
            print(f"  Test {TOTAL}: {name} ............ PASSED")
        except Exception as e:
            FAIL += 1
            print(f"  Test {TOTAL}: {name} ............ FAILED: {e}")

    # Run sync tests
    for name, fn in SYNC_TESTS:
        run_test(name, fn)

    # Run async tests
    async def _run_all_async():
        for name, fn in ASYNC_TESTS:
            await run_async_test(name, fn)

    asyncio.run(_run_all_async())

    print()
    print("=" * 50)
    print(f"Results: {PASS}/{TOTAL} passed, {FAIL} failed")
    print("=" * 50)

    sys.exit(1 if FAIL > 0 else 0)
