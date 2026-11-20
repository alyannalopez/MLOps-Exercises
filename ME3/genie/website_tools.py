"""M4.75: MCP Website and GitHub Publishing Tools.

These tools are registered with the MCP server alongside existing tools.
They allow the LLM to manage website projects and publish to GitHub Pages:

Website tools:
  - create_website_project: Create a new website project
  - read_website_file: Read a file from a project
  - edit_website_file: Edit an existing file
  - list_website_files: List files in a project
  - preview_website: Start a local preview server
  - get_website_project_status: Get project information

GitHub tools:
  - check_github_configuration: Check if GitHub publishing is configured
  - check_github_repository: Check repository status
  - publish_website_to_github: Publish a project to GitHub Pages
  - get_github_pages_status: Check deployment status
"""
from __future__ import annotations

from genie.website_service import get_website_service
from genie.github_service import get_github_service, GitHubConfig
from genie.config import Config


# ---------------------------------------------------------------------------
# Website MCP Tools
# ---------------------------------------------------------------------------

async def create_website_project(name: str, description: str = "") -> str:
    """Create a new website project.

    Args:
        name: Project name (letters, numbers, hyphens, underscores).
        description: Optional description for context.

    Returns:
        Result message with project location.
    """
    service = get_website_service()
    result = service.create_project(name, overwrite=False)
    return result.message


async def read_website_file(project_name: str, file_path: str) -> str:
    """Read a file from a website project.

    Args:
        project_name: The project containing the file.
        file_path: Path relative to project root (e.g., "index.html").

    Returns:
        File content or error message.
    """
    service = get_website_service()
    result = service.read_file(project_name, file_path)
    if not result.success:
        return result.message
    # Return message + content (content may be large)
    output = result.message + "\n\n--- Content ---\n"
    if result.content and len(result.content) > 5000:
        output += result.content[:5000] + f"\n...(truncated, {len(result.content)} total chars)"
    elif result.content:
        output += result.content
    return output


async def edit_website_file(project_name: str, file_path: str, new_content: str) -> str:
    """Edit an existing file in a website project.

    IMPORTANT: This replaces the entire file content. The user must have
    explicitly confirmed this overwrite operation.

    Args:
        project_name: The project containing the file.
        file_path: Path relative to project root.
        new_content: Complete new file content.

    Returns:
        Result message.
    """
    service = get_website_service()
    result = service.edit_file(project_name, file_path, new_content, confirm_overwrite=True)
    return result.message


async def list_website_files(project_name: str) -> str:
    """List all files in a website project.

    Args:
        project_name: The project to list.

    Returns:
        File listing or error message.
    """
    service = get_website_service()
    result = service.list_files(project_name)
    return result.message


async def preview_website(project_name: str, port: int = 8800) -> str:
    """Start a local preview server for a website project.

    Args:
        project_name: The project to preview.
        port: Preferred port (auto-increments if busy).

    Returns:
        Preview URL or error message.
    """
    service = get_website_service()
    result = await service.start_preview(project_name, port)
    return result.message


async def get_website_project_status(project_name: str) -> str:
    """Get the status of a website project.

    Args:
        project_name: The project to check.

    Returns:
        Project information or error message.
    """
    service = get_website_service()
    result = service.get_project_status(project_name)
    return result.message


# ---------------------------------------------------------------------------
# GitHub Publishing MCP Tools
# ---------------------------------------------------------------------------

async def check_github_configuration() -> str:
    """Check if GitHub Pages publishing is configured.

    Returns:
        Configuration status. Never exposes the actual token.
    """
    cfg = Config.load()
    gh_config = GitHubConfig(
        token=cfg.github_token,
        username=cfg.github_username,
        pages_repository=cfg.github_pages_repository,
    )
    service = get_github_service(gh_config)
    info = service.check_configuration()

    if info["configured"]:
        return (
            f"GitHub Pages publishing is configured.\n"
            f"Username: {info['username']}\n"
            f"Repository: {info['repository']}\n"
            f"Token: present (not displayed for security)"
        )
    else:
        missing = info.get("missing", [])
        return (
            "GitHub Pages publishing is NOT configured.\n\n"
            "Add these to your .env file:\n"
            + "\n".join(f"  {m}=your_value_here" for m in missing) + "\n\n"
            "The GITHUB_TOKEN needs 'repo' scope for private repositories."
        )


async def check_github_repository() -> str:
    """Check the status of the target GitHub repository.

    Returns:
        Repository information or error message.
    """
    cfg = Config.load()
    gh_config = GitHubConfig(
        token=cfg.github_token,
        username=cfg.github_username,
        pages_repository=cfg.github_pages_repository,
    )
    service = get_github_service(gh_config)

    if not gh_config.is_configured:
        return "GitHub is not configured. Run check_github_configuration for details."

    repo_info = await service.check_repository()
    if repo_info is None:
        return "Failed to check repository. Check your GitHub token and network connection."

    if not repo_info.full_name:
        return (
            f"Repository '{gh_config.pages_repository}' does not exist yet.\n"
            f"It will be created automatically when you publish."
        )

    return (
        f"Repository: {repo_info.full_name}\n"
        f"URL: {repo_info.html_url}\n"
        f"Private: {'Yes' if repo_info.private else 'No'}\n"
        f"Default branch: {repo_info.default_branch}\n"
        f"GitHub Pages: {'Configured' if repo_info.has_pages else 'Not configured'}\n"
        + (f"Pages URL: {repo_info.pages_url}\n" if repo_info.pages_url else "")
    )


async def publish_website_to_github(
    project_name: str,
    confirm_publish: bool = False,
) -> str:
    """Publish a website project to GitHub Pages.

    IMPORTANT: This requires explicit user confirmation. The LLM must set
    confirm_publish=True ONLY after the user has explicitly confirmed.

    Args:
        project_name: The project to publish.
        confirm_publish: Must be True — enforces explicit confirmation.

    Returns:
        Deployment result or error message.
    """
    cfg = Config.load()
    gh_config = GitHubConfig(
        token=cfg.github_token,
        username=cfg.github_username,
        pages_repository=cfg.github_pages_repository,
    )
    github_svc = get_github_service(gh_config)
    web_svc = get_website_service()

    # Get project root
    from genie.website_service import validate_project_name
    clean_name = validate_project_name(project_name)
    if not clean_name:
        return f"Invalid project name '{project_name}'."

    project_root = web_svc._get_project_root(clean_name)
    if not project_root or not project_root.exists():
        return f"Project '{clean_name}' not found."

    result = await github_svc.publish_website(
        project_root=project_root,
        confirm_publish=confirm_publish,
        replace_existing=False,
    )
    return result.message


async def get_github_pages_status() -> str:
    """Check the current GitHub Pages deployment status.

    Returns:
        Deployment status information.
    """
    cfg = Config.load()
    gh_config = GitHubConfig(
        token=cfg.github_token,
        username=cfg.github_username,
        pages_repository=cfg.github_pages_repository,
    )
    service = get_github_service(gh_config)

    if not gh_config.is_configured:
        return "GitHub is not configured. Run check_github_configuration for details."

    status = await service.get_deployment_status()
    return status.get("message", "Unable to determine deployment status.")


# ---------------------------------------------------------------------------
# create_website_file — exposed for the LLM to create files during generation
# ---------------------------------------------------------------------------

async def create_website_file(project_name: str, file_path: str, content: str) -> str:
    """Create a new file in a website project. Only use for files that DON'T exist yet.

    Args:
        project_name: The project to create the file in.
        file_path: Path relative to project root (e.g., "css/style.css").
        content: Complete file content. Keep it concise — under 2000 characters for reliability.

    Returns:
        Result message.
    """
    service = get_website_service()
    result = service.create_file(project_name, file_path, content)
    return result.message


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_website_tools(server):
    """Register website and GitHub publishing tools with an MCP server.

    Args:
        server: The MCPServer instance from mcp_server.py.
    """
    server.tool(
        name="create_website_project",
        description=(
            "Create a new website project in the Genie workspace. "
            "Generates a default index.html. Project names use letters, numbers, "
            "hyphens, and underscores only."
        ),
    )(create_website_project)

    server.tool(
        name="create_website_file",
        description=(
            "Create a NEW file (HTML, CSS, JS, etc.) in an existing website project. "
            "Only use for files that don't exist yet. For existing files, use edit_website_file instead. "
            "Keep content concise — under 2000 characters. "
            "The file_path is relative to the project root (e.g., 'css/style.css')."
        ),
    )(create_website_file)

    server.tool(
        name="read_website_file",
        description="Read the content of a file from a website project.",
    )(read_website_file)

    server.tool(
        name="edit_website_file",
        description=(
            "Edit an existing file in a website project. Replaces the entire file content. "
            "Use this for revisions and modifications."
        ),
    )(edit_website_file)

    server.tool(
        name="list_website_files",
        description="List all files in a website project.",
    )(list_website_files)

    server.tool(
        name="preview_website",
        description=(
            "Start a local HTTP preview server for a website project. "
            "Returns a local URL (127.0.0.1:port) for browser viewing. "
            "This is a local preview only — not publicly accessible."
        ),
    )(preview_website)

    server.tool(
        name="get_website_project_status",
        description="Get information about a website project (files, size, pages).",
    )(get_website_project_status)

    # GitHub publishing tools
    server.tool(
        name="check_github_configuration",
        description="Check if GitHub Pages publishing is configured.",
    )(check_github_configuration)

    server.tool(
        name="check_github_repository",
        description="Check the status of the target GitHub repository.",
    )(check_github_repository)

    server.tool(
        name="publish_website_to_github",
        description=(
            "Publish a website project to GitHub Pages. "
            "REQUIRES explicit user confirmation — set confirm_publish=True only "
            "after the user has explicitly agreed to publish."
        ),
    )(publish_website_to_github)

    server.tool(
        name="get_github_pages_status",
        description="Check the current GitHub Pages deployment status.",
    )(get_github_pages_status)
