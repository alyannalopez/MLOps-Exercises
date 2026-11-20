"""M4.75: Website Service — controlled project creation, file management, and local preview.

Provides:
  * WebsiteService — manages website projects within ~/GenieWorkspace/websites/
  * Strict path validation preventing directory traversal, symlink escapes, cross-project access
  * Local HTTP preview server with automatic port selection
  * Project creation, file CRUD, listing, and status reporting

Security principles:
  - All file operations are confined to the designated project directory.
  - Paths are resolved and validated against the project root using Path.is_relative_to().
  - Symlinks are followed but the resolved path must still be within the project.
  - URL-encoded traversal sequences are decoded before validation.
  - Project names are validated against a whitelist of safe characters.
  - Destructive operations (overwriting existing files, deleting projects) require explicit confirmation.
  - Preview server serves only the selected project's files.
  - No arbitrary shell commands are executed.
"""
from __future__ import annotations

import asyncio
import html
import http.server
import logging
import os
import re
import socketserver
import threading
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = Path.home() / "GenieWorkspace" / "websites"

PROJECT_NAME_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$')

ALLOWED_EXTENSIONS = {
    '.html', '.htm', '.css', '.js', '.json',
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.ico',
    '.txt', '.md', '.xml', '.woff', '.woff2', '.ttf', '.eot',
}

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB per file
MAX_PROJECT_FILES = 100


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def validate_project_name(name: str) -> str | None:
    """Validate a project name. Returns cleaned name or None if invalid."""
    if not name or len(name) < 1:
        return None
    name = name.strip().lower().replace(' ', '-')
    if not PROJECT_NAME_RE.match(name):
        return None
    return name


def resolve_and_validate(project_root: Path, file_path: str) -> Path | None:
    """Resolve a file path and validate it stays within the project directory.

    Handles:
      - Absolute paths
      - Relative paths with ../ traversal
      - URL-encoded traversal (%2e%2e%2f)
      - Null bytes
      - Symlink escapes (resolves symlinks and checks containment)

    Returns the resolved Path if valid, None otherwise.
    """
    if not file_path:
        return None

    # Reject null bytes
    if '\x00' in file_path:
        return None

    # Decode URL encoding to catch encoded traversal attempts
    decoded = urllib.parse.unquote(file_path)
    if '\x00' in decoded:
        return None

    # Build the candidate path
    candidate = project_root / decoded

    # Resolve to absolute path (follows symlinks)
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, ValueError):
        return None

    # Also resolve the project root
    try:
        resolved_root = project_root.resolve(strict=False)
    except (OSError, ValueError):
        return None

    # Check containment using is_relative_to
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None

    # Additional check: ensure no component is '..' after normalization
    parts = resolved.parts
    # The resolved path should start with the resolved root's parts
    if not parts[:len(resolved_root.parts)] == resolved_root.parts:
        return None

    return resolved


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename, removing dangerous characters."""
    if not filename:
        return ""
    # Remove null bytes
    filename = filename.replace('\x00', '')
    # Decode URL encoding
    filename = urllib.parse.unquote(filename)
    # Remove path separators (prevent directory creation via filename)
    filename = filename.replace('/', '').replace('\\', '')
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)
    # Limit length
    filename = filename[:255]
    return filename.strip()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WebsiteProject:
    """Represents a website project."""
    name: str
    root: Path
    created_at: float = 0.0
    updated_at: float = 0.0
    pages: list[str] = field(default_factory=list)
    file_count: int = 0
    total_size: int = 0


@dataclass
class FileOperationResult:
    """Result of a file operation."""
    success: bool
    message: str
    file_path: str = ""
    content: str = ""
    size: int = 0


# ---------------------------------------------------------------------------
# Preview server
# ---------------------------------------------------------------------------

class _PreviewHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that serves only files within the project directory."""

    project_root: Path = Path(".")  # Set by PreviewServer
    server_name = "Genie-Preview/1.0"

    def translate_path(self, path: str) -> str:
        """Override to enforce project directory containment."""
        # Decode URL encoding
        decoded = urllib.parse.unquote(path)

        # Reject null bytes
        if '\x00' in decoded:
            self.send_error(400, "Bad Request")
            return ""

        # Resolve within project root
        candidate = self.project_root / decoded.lstrip('/')

        try:
            resolved = candidate.resolve(strict=False)
            resolved_root = self.project_root.resolve(strict=False)
        except (OSError, ValueError):
            self.send_error(400, "Bad Request")
            return ""

        # Containment check
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            self.send_error(403, "Forbidden")
            return ""

        return str(resolved)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging to stderr."""
        pass


class PreviewServer:
    """Manages local HTTP preview servers for website projects."""

    def __init__(self):
        self._servers: dict[int, tuple[socketserver.TCPServer, threading.Thread]] = {}
        self._lock = asyncio.Lock()

    async def start(self, project_root: Path, preferred_port: int = 8800) -> int | None:
        """Start a preview server for the given project.

        Returns the port number or None if failed.
        """
        async with self._lock:
            # Find an available port
            port = preferred_port
            for _ in range(20):
                if port not in self._servers:
                    try:
                        with socketserver.TCPServer(("127.0.0.1", port), _PreviewHandler) as test:
                            test.server_close()
                        break  # Port is available
                    except OSError:
                        port += 1
                        continue
                else:
                    port += 1
            else:
                return None  # Could not find available port

            # Create and start the server
            resolved_root = project_root.resolve()

            class _HandlerWithRoot(_PreviewHandler):
                project_root = resolved_root

            server = socketserver.TCPServer(
                ("127.0.0.1", port),
                _HandlerWithRoot,
                bind_and_activate=True
            )
            server.allow_reuse_address = True

            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            self._servers[port] = (server, thread)
            return port

    async def stop(self, port: int) -> bool:
        """Stop a preview server by port."""
        async with self._lock:
            if port in self._servers:
                server, thread = self._servers.pop(port)
                server.shutdown()
                server.server_close()
                thread.join(timeout=3.0)
                return True
            return False

    async def stop_all(self) -> None:
        """Stop all preview servers."""
        async with self._lock:
            ports = list(self._servers.keys())
            for port in ports:
                await self.stop(port)

    async def get_active_ports(self) -> list[int]:
        """Get list of active preview server ports."""
        async with self._lock:
            return list(self._servers.keys())


# ---------------------------------------------------------------------------
# Website Service
# ---------------------------------------------------------------------------

class WebsiteService:
    """Manages website projects within the designated workspace."""

    def __init__(self, workspace_root: Path | None = None):
        self.workspace_root = (workspace_root or WORKSPACE_ROOT).resolve()
        self.preview_server = PreviewServer()
        self._ensure_workspace()

    def _ensure_workspace(self) -> None:
        """Create the workspace directory if it doesn't exist."""
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _get_project_root(self, project_name: str) -> Path | None:
        """Get the project root path, validating containment."""
        project_dir = self.workspace_root / project_name
        try:
            resolved = project_dir.resolve(strict=False)
            resolved_ws = self.workspace_root.resolve(strict=False)
            resolved.relative_to(resolved_ws)
            return resolved
        except (ValueError, OSError):
            return None

    def _scan_project(self, project_root: Path) -> WebsiteProject:
        """Scan a project directory and return its metadata."""
        import time
        pages = []
        file_count = 0
        total_size = 0

        if project_root.exists():
            for item in sorted(project_root.rglob('*')):
                if item.is_file():
                    file_count += 1
                    try:
                        total_size += item.stat().st_size
                    except OSError:
                        pass
                    if item.suffix.lower() in ('.html', '.htm'):
                        rel = item.relative_to(project_root)
                        pages.append(str(rel))

        created = 0.0
        updated = 0.0
        if project_root.exists():
            try:
                stat = project_root.stat()
                created = stat.st_birthtime if hasattr(stat, 'st_birthtime') else stat.st_ctime
                updated = stat.st_mtime
            except OSError:
                pass

        return WebsiteProject(
            name=project_root.name,
            root=project_root,
            created_at=created,
            updated_at=updated,
            pages=pages,
            file_count=file_count,
            total_size=total_size,
        )

    # ---- Project operations ----

    def create_project(self, name: str, overwrite: bool = False) -> FileOperationResult:
        """Create a new website project.

        Args:
            name: Project name (alphanumeric, hyphens, underscores).
            overwrite: If True, overwrite existing project. Requires explicit confirmation.

        Returns:
            FileOperationResult with success status and message.
        """
        clean_name = validate_project_name(name)
        if not clean_name:
            return FileOperationResult(
                success=False,
                message=f"Invalid project name '{name}'. Use only letters, numbers, hyphens, and underscores. Max 128 characters."
            )

        project_root = self._get_project_root(clean_name)
        if not project_root:
            return FileOperationResult(
                success=False,
                message="Could not resolve project path."
            )

        if project_root.exists():
            if not overwrite:
                return FileOperationResult(
                    success=False,
                    message=f"Project '{clean_name}' already exists. Use overwrite=True to replace it."
                )
            # Overwrite: remove existing project
            import shutil
            try:
                shutil.rmtree(project_root)
            except OSError as e:
                return FileOperationResult(
                    success=False,
                    message=f"Failed to remove existing project: {e}"
                )

        try:
            project_root.mkdir(parents=True, exist_ok=True)
            # Create a default index.html
            default_html = self._generate_default_html(clean_name)
            index_path = project_root / "index.html"
            index_path.write_text(default_html, encoding='utf-8')

            return FileOperationResult(
                success=True,
                message=f"Project '{clean_name}' created. It contains a default index.html. Use edit_website_file to customize it, or create_website_file for additional files like css/style.css.",
                file_path=str(project_root),
            )
        except OSError as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to create project: {e}"
            )

    def _generate_default_html(self, project_name: str) -> str:
        """Generate a default index.html for a new project."""
        display_name = project_name.replace('-', ' ').title()
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(display_name)}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; }}
        h1 {{ color: #333; }}
        p {{ color: #666; }}
    </style>
</head>
<body>
    <h1>Welcome to {html.escape(display_name)}</h1>
    <p>This website was generated by Genie.</p>
</body>
</html>"""

    def get_project_status(self, name: str) -> FileOperationResult:
        """Get the status of a website project."""
        clean_name = validate_project_name(name)
        if not clean_name:
            return FileOperationResult(success=False, message=f"Invalid project name '{name}'.")

        project_root = self._get_project_root(clean_name)
        if not project_root or not project_root.exists():
            return FileOperationResult(success=False, message=f"Project '{clean_name}' not found.")

        project = self._scan_project(project_root)

        # Check for active preview
        active_ports = []
        try:
            # Synchronous check — acceptable for status queries
            loop = asyncio.get_event_loop()
            if loop.is_running():
                active_ports = asyncio.run_coroutine_threadsafe(
                    self.preview_server.get_active_ports(), loop
                ).result(timeout=2.0)
            else:
                active_ports = []
        except Exception:
            active_ports = []

        info = (
            f"Project: {project.name}\n"
            f"Location: {project.root}\n"
            f"Files: {project.file_count}\n"
            f"Size: {project.total_size:,} bytes\n"
            f"Pages: {', '.join(project.pages) if project.pages else 'none'}"
        )

        return FileOperationResult(
            success=True,
            message=info,
            file_path=str(project.root),
        )

    def list_projects(self) -> FileOperationResult:
        """List all website projects in the workspace."""
        if not self.workspace_root.exists():
            return FileOperationResult(
                success=True,
                message="No projects found. Create one with create_website_project."
            )

        projects = []
        for item in sorted(self.workspace_root.iterdir()):
            if item.is_dir():
                proj = self._scan_project(item)
                projects.append(
                    f"  - {proj.name} ({proj.file_count} files, {proj.total_size:,} bytes)"
                )

        if not projects:
            return FileOperationResult(
                success=True,
                message="No projects found. Create one with create_website_project."
            )

        return FileOperationResult(
            success=True,
            message=f"Projects in {self.workspace_root}:\n" + "\n".join(projects),
        )

    # ---- File operations ----

    def create_file(self, project_name: str, file_path: str, content: str) -> FileOperationResult:
        """Create a file within a website project.

        Args:
            project_name: The project to create the file in.
            file_path: Path relative to project root (e.g., "css/style.css").
            content: File content.

        Returns:
            FileOperationResult with success status.
        """
        clean_name = validate_project_name(project_name)
        if not clean_name:
            return FileOperationResult(success=False, message=f"Invalid project name '{project_name}'.")

        project_root = self._get_project_root(clean_name)
        if not project_root or not project_root.exists():
            return FileOperationResult(success=False, message=f"Project '{clean_name}' not found.")

        # Validate the file path stays within the project
        resolved = resolve_and_validate(project_root, file_path)
        if not resolved:
            return FileOperationResult(
                success=False,
                message=f"Invalid file path '{file_path}'. Path must be within the project directory."
            )

        # Check file size
        if len(content.encode('utf-8')) > MAX_FILE_SIZE:
            return FileOperationResult(
                success=False,
                message=f"File content exceeds maximum size of {MAX_FILE_SIZE / 1024 / 1024}MB."
            )

        # Check if file exists
        if resolved.exists():
            return FileOperationResult(
                success=False,
                message=f"File '{file_path}' already exists. Use edit_website_file to modify it."
            )

        # Create parent directories
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding='utf-8')

            # Verify file count limit
            project = self._scan_project(project_root)
            if project.file_count > MAX_PROJECT_FILES:
                resolved.unlink()
                return FileOperationResult(
                    success=False,
                    message=f"Project exceeds maximum file count of {MAX_PROJECT_FILES}."
                )

            return FileOperationResult(
                success=True,
                message=f"Created file '{file_path}' ({len(content)} chars) in project '{clean_name}'.",
                file_path=str(resolved),
                size=len(content),
            )
        except OSError as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to create file: {e}"
            )

    def read_file(self, project_name: str, file_path: str) -> FileOperationResult:
        """Read a file from a website project."""
        clean_name = validate_project_name(project_name)
        if not clean_name:
            return FileOperationResult(success=False, message=f"Invalid project name '{project_name}'.")

        project_root = self._get_project_root(clean_name)
        if not project_root or not project_root.exists():
            return FileOperationResult(success=False, message=f"Project '{clean_name}' not found.")

        resolved = resolve_and_validate(project_root, file_path)
        if not resolved:
            return FileOperationResult(
                success=False,
                message=f"Invalid file path '{file_path}'."
            )

        if not resolved.exists():
            return FileOperationResult(
                success=False,
                message=f"File '{file_path}' not found in project '{clean_name}'."
            )

        if not resolved.is_file():
            return FileOperationResult(
                success=False,
                message=f"'{file_path}' is not a file."
            )

        try:
            content = resolved.read_text(encoding='utf-8')
            return FileOperationResult(
                success=True,
                message=f"Read file '{file_path}' ({len(content)} chars).",
                file_path=str(resolved),
                content=content,
                size=len(content),
            )
        except (OSError, UnicodeDecodeError) as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to read file: {e}"
            )

    def edit_file(self, project_name: str, file_path: str, new_content: str,
                  confirm_overwrite: bool = False) -> FileOperationResult:
        """Edit an existing file in a website project.

        Args:
            project_name: The project containing the file.
            file_path: Path relative to project root.
            new_content: New file content (replaces entire file).
            confirm_overwrite: Must be True to overwrite existing content.

        Returns:
            FileOperationResult with success status.
        """
        clean_name = validate_project_name(project_name)
        if not clean_name:
            return FileOperationResult(success=False, message=f"Invalid project name '{project_name}'.")

        project_root = self._get_project_root(clean_name)
        if not project_root or not project_root.exists():
            return FileOperationResult(success=False, message=f"Project '{clean_name}' not found.")

        resolved = resolve_and_validate(project_root, file_path)
        if not resolved:
            return FileOperationResult(
                success=False,
                message=f"Invalid file path '{file_path}'."
            )

        if not resolved.exists():
            return FileOperationResult(
                success=False,
                message=f"File '{file_path}' not found. Use create_website_file to create it."
            )

        if not confirm_overwrite:
            return FileOperationResult(
                success=False,
                message=f"Overwrite not confirmed. Set confirm_overwrite=True to edit '{file_path}'."
            )

        if len(new_content.encode('utf-8')) > MAX_FILE_SIZE:
            return FileOperationResult(
                success=False,
                message=f"Content exceeds maximum size of {MAX_FILE_SIZE / 1024 / 1024}MB."
            )

        try:
            resolved.write_text(new_content, encoding='utf-8')
            return FileOperationResult(
                success=True,
                message=f"Updated file '{file_path}' ({len(new_content)} chars) in project '{clean_name}'.",
                file_path=str(resolved),
                size=len(new_content),
            )
        except OSError as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to edit file: {e}"
            )

    def list_files(self, project_name: str) -> FileOperationResult:
        """List all files in a website project."""
        clean_name = validate_project_name(project_name)
        if not clean_name:
            return FileOperationResult(success=False, message=f"Invalid project name '{project_name}'.")

        project_root = self._get_project_root(clean_name)
        if not project_root or not project_root.exists():
            return FileOperationResult(success=False, message=f"Project '{clean_name}' not found.")

        files = []
        for item in sorted(project_root.rglob('*')):
            if item.is_file():
                rel = item.relative_to(project_root)
                try:
                    size = item.stat().st_size
                    files.append(f"  {rel} ({size:,} bytes)")
                except OSError:
                    files.append(f"  {rel}")

        if not files:
            return FileOperationResult(
                success=True,
                message=f"Project '{clean_name}' is empty."
            )

        return FileOperationResult(
            success=True,
            message=f"Files in project '{clean_name}':\n" + "\n".join(files),
        )

    # ---- Preview ----

    async def start_preview(self, project_name: str, preferred_port: int = 8800) -> FileOperationResult:
        """Start a local preview server for a website project.

        Args:
            project_name: The project to preview.
            preferred_port: Preferred port (will auto-increment if busy).

        Returns:
            FileOperationResult with the preview URL or error message.
        """
        clean_name = validate_project_name(project_name)
        if not clean_name:
            return FileOperationResult(success=False, message=f"Invalid project name '{project_name}'.")

        project_root = self._get_project_root(clean_name)
        if not project_root or not project_root.exists():
            return FileOperationResult(success=False, message=f"Project '{clean_name}' not found.")

        port = await self.preview_server.start(project_root, preferred_port)
        if port is None:
            return FileOperationResult(
                success=False,
                message="Could not start preview server — no available ports."
            )

        url = f"http://127.0.0.1:{port}"
        return FileOperationResult(
            success=True,
            message=(
                f"Preview server started for project '{clean_name}'.\n"
                f"URL: {url}\n"
                f"(Local preview only — not publicly accessible.)"
            ),
            file_path=url,
        )

    async def stop_preview(self, port: int) -> FileOperationResult:
        """Stop a preview server."""
        stopped = await self.preview_server.stop(port)
        if stopped:
            return FileOperationResult(
                success=True,
                message=f"Preview server on port {port} stopped."
            )
        return FileOperationResult(
            success=False,
            message=f"No preview server running on port {port}."
        )

    async def stop_all_previews(self) -> FileOperationResult:
        """Stop all preview servers."""
        await self.preview_server.stop_all()
        return FileOperationResult(
            success=True,
            message="All preview servers stopped."
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_website_service: WebsiteService | None = None


def get_website_service() -> WebsiteService:
    """Get or create the WebsiteService singleton."""
    global _website_service
    if _website_service is None:
        _website_service = WebsiteService()
    return _website_service
