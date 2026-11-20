"""M4.75: GitHub Service — GitHub Pages publishing for website projects.

Provides:
  * GitHubService — manages GitHub Pages publishing using the GitHub REST API
  * Secure credential handling (never logs or exposes tokens)
  * Repository creation, configuration, and deployment
  * Deployment status checking
  * Confirmation enforcement before publishing

Security principles:
  - GitHub token is loaded from .env, never hardcoded or logged.
  - Token is masked in all error messages and responses.
  - Publishing requires explicit user confirmation.
  - Existing repository content is preserved unless explicitly authorized for replacement.
  - Rate limits and API errors are handled gracefully.
  - No arbitrary shell commands or unrestricted API access.
"""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"
TOKEN_MASK = "***"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GitHubConfig:
    """GitHub configuration loaded from .env."""
    token: str = ""
    username: str = ""
    pages_repository: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.username and self.pages_repository)

    @property
    def masked_token(self) -> str:
        if not self.token:
            return ""
        return self.token[:4] + "..." + self.token[-4:] if len(self.token) > 8 else TOKEN_MASK


@dataclass
class GitHubRepoInfo:
    """Information about a GitHub repository."""
    full_name: str = ""
    private: bool = False
    default_branch: str = ""
    has_pages: bool = False
    pages_url: str = ""
    html_url: str = ""


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    success: bool
    message: str
    repo_url: str = ""
    pages_url: str = ""
    deployment_status: str = ""  # "queued", "built", "ready", "error"


# ---------------------------------------------------------------------------
# GitHub Service
# ---------------------------------------------------------------------------

class GitHubService:
    """Manages GitHub Pages publishing for website projects."""

    def __init__(self, config: GitHubConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client with GitHub auth headers."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=GITHUB_API_BASE,
                headers={
                    "Authorization": f"Bearer {self.config.token}",
                    "Accept": "application/vnd.github.v3+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=httpx.Timeout(30.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _mask_token(self, text: str) -> str:
        """Mask the GitHub token in any text to prevent leakage."""
        if self.config.token and len(self.config.token) > 4:
            text = text.replace(self.config.token, TOKEN_MASK)
            text = text.replace(self.config.token[:8], "****")
        return text

    # ---- Configuration checks ----

    def check_configuration(self) -> dict[str, Any]:
        """Check if GitHub publishing is configured.

        Returns a dict with configuration status. Never exposes the actual token.
        """
        result = {
            "configured": self.config.is_configured,
            "username": self.config.username or "(not set)",
            "repository": self.config.pages_repository or "(not set)",
            "token_present": bool(self.config.token),
        }

        if not self.config.is_configured:
            missing = []
            if not self.config.token:
                missing.append("GITHUB_TOKEN")
            if not self.config.username:
                missing.append("GITHUB_USERNAME")
            if not self.config.pages_repository:
                missing.append("GITHUB_PAGES_REPOSITORY")
            result["missing"] = missing

        return result

    # ---- Repository operations ----

    async def check_repository(self) -> GitHubRepoInfo | None:
        """Check if the target repository exists and get its info.

        Returns GitHubRepoInfo or None on error.
        """
        if not self.config.is_configured:
            return None

        try:
            client = await self._get_client()
            repo_name = self.config.pages_repository.split("/")[-1] if "/" in self.config.pages_repository else self.config.pages_repository
            resp = await client.get(f"/repos/{self.config.username}/{repo_name}")

            if resp.status_code == 404:
                return GitHubRepoInfo()  # Empty = doesn't exist
            elif resp.status_code != 200:
                log.error("GitHub API error checking repo: %d %s", resp.status_code, self._mask_token(resp.text[:200]))
                return None

            data = resp.json()
            # Check for Pages configuration
            pages_url = ""
            has_pages = False
            try:
                pages_resp = await client.get(f"/repos/{self.config.username}/{repo_name}/pages")
                if pages_resp.status_code == 200:
                    has_pages = True
                    pages_data = pages_resp.json()
                    pages_url = pages_data.get("html_url", "")
            except Exception:
                pass

            return GitHubRepoInfo(
                full_name=data.get("full_name", ""),
                private=data.get("private", False),
                default_branch=data.get("default_branch", "main"),
                has_pages=has_pages,
                pages_url=pages_url,
                html_url=data.get("html_url", ""),
            )
        except Exception as e:
            log.error("Error checking repository: %s", self._mask_token(str(e)))
            return None

    async def create_repository(self, name: str | None = None, private: bool = True) -> dict[str, Any]:
        """Create a new repository for GitHub Pages.

        Args:
            name: Repository name (defaults to config.pages_repository).
            private: Whether the repo should be private.

        Returns:
            Dict with success status and repository info.
        """
        if not self.config.is_configured:
            return {"success": False, "message": "GitHub is not configured. Set GITHUB_TOKEN, GITHUB_USERNAME, and GITHUB_PAGES_REPOSITORY in .env."}

        try:
            client = await self._get_client()
            repo_name = name or self.config.pages_repository.split("/")[-1] if "/" in (name or self.config.pages_repository) else (name or self.config.pages_repository)

            resp = await client.post("/user/repos", json={
                "name": repo_name,
                "private": private,
                "auto_init": True,
                "description": f"Website project managed by Genie",
            })

            if resp.status_code == 422:
                return {"success": False, "message": f"Repository '{repo_name}' already exists."}
            elif resp.status_code != 201:
                return {"success": False, "message": f"Failed to create repository: {resp.status_code}"}

            data = resp.json()
            return {
                "success": True,
                "message": f"Repository '{data.get('full_name', repo_name)}' created.",
                "html_url": data.get("html_url", ""),
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to create repository: {self._mask_token(str(e))}"}

    async def configure_pages(self, branch: str = "main", path: str = "/") -> dict[str, Any]:
        """Configure GitHub Pages for the repository.

        Args:
            branch: The branch to serve Pages from.
            path: The directory within the branch ("/" for root, "/docs" for docs folder).

        Returns:
            Dict with success status and Pages URL.
        """
        if not self.config.is_configured:
            return {"success": False, "message": "GitHub is not configured."}

        try:
            client = await self._get_client()
            repo_name = self.config.pages_repository.split("/")[-1] if "/" in self.config.pages_repository else self.config.pages_repository

            resp = await client.put(
                f"/repos/{self.config.username}/{repo_name}/pages",
                json={"build_type": "legacy", "source": {"branch": branch, "path": path}}
            )

            if resp.status_code == 404:
                return {"success": False, "message": f"Repository '{repo_name}' not found."}
            elif resp.status_code == 403:
                return {"success": False, "message": "Permission denied. Check your GitHub token has 'repo' scope."}
            elif resp.status_code != 201 and resp.status_code != 200:
                return {"success": False, "message": f"Failed to configure Pages: {resp.status_code}"}

            data = resp.json()
            pages_url = data.get("html_url", "")
            return {
                "success": True,
                "message": f"GitHub Pages configured. URL: {pages_url}",
                "pages_url": pages_url,
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to configure Pages: {self._mask_token(str(e))}"}

    # ---- Publishing ----

    async def publish_website(
        self,
        project_root: Path,
        branch: str = "main",
        confirm_publish: bool = False,
        replace_existing: bool = False,
    ) -> DeploymentResult:
        """Publish a website project to GitHub Pages.

        Args:
            project_root: Path to the project directory.
            branch: Target branch (default: main).
            confirm_publish: MUST be True — enforces explicit user confirmation.
            replace_existing: If True, replace existing files. If False, merge with existing.

        Returns:
            DeploymentResult with success status and URLs.
        """
        # 1. Check configuration
        if not self.config.is_configured:
            return DeploymentResult(
                success=False,
                message="GitHub publishing is not configured.\n\n"
                        "Add these to your .env file:\n"
                        "  GITHUB_TOKEN=your_personal_access_token\n"
                        "  GITHUB_USERNAME=your_github_username\n"
                        "  GITHUB_PAGES_REPOSITORY=your_pages_repo_name\n\n"
                        "The token needs 'repo' scope for private repositories."
            )

        # 2. Enforce confirmation
        if not confirm_publish:
            return DeploymentResult(
                success=False,
                message="Publishing requires explicit confirmation. Set confirm_publish=True after the user has confirmed."
            )

        # 3. Validate project
        if not project_root.exists() or not project_root.is_dir():
            return DeploymentResult(
                success=False,
                message=f"Project directory not found: {project_root}"
            )

        # 4. Collect files to upload
        files = {}
        for item in sorted(project_root.rglob('*')):
            if item.is_file():
                rel = item.relative_to(project_root)
                try:
                    content = item.read_bytes()
                    files[str(rel)] = content
                except OSError:
                    continue

        if not files:
            return DeploymentResult(
                success=False,
                message="Project has no files to publish."
            )

        repo_name = self.config.pages_repository.split("/")[-1] if "/" in self.config.pages_repository else self.config.pages_repository

        try:
            client = await self._get_client()

            # 5. Check if repo exists, create if not
            repo_info = await self.check_repository()
            if not repo_info:
                # Try to create it
                create_result = await self.create_repository(private=True)
                if not create_result.get("success"):
                    return DeploymentResult(
                        success=False,
                        message=f"Repository does not exist and could not be created: {create_result['message']}"
                    )
                repo_info = await self.check_repository()
                if not repo_info:
                    return DeploymentResult(
                        success=False,
                        message="Repository was created but could not be verified."
                    )

            # 6. Determine target branch
            target_branch = repo_info.default_branch or "main"

            # 7. Upload files via Git Trees API
            upload_result = await self._upload_files(client, repo_name, files, target_branch, replace_existing)
            if not upload_result["success"]:
                return DeploymentResult(
                    success=False,
                    message=f"Upload failed: {upload_result['message']}"
                )

            # 8. Configure Pages if not already configured
            if not repo_info.has_pages:
                pages_result = await self.configure_pages(branch=target_branch)
                if not pages_result.get("success"):
                    log.warning("Pages configuration failed: %s", pages_result.get("message", ""))
                else:
                    log.info("Pages configured: %s", pages_result.get("pages_url", ""))

            # 9. Determine Pages URL
            pages_url = repo_info.pages_url
            if not pages_url:
                if self.config.username.endswith(".github.io"):
                    pages_url = f"https://{self.config.username}"
                else:
                    pages_url = f"https://{self.config.username}.github.io/{repo_name}/"

            return DeploymentResult(
                success=True,
                message=(
                    f"Website published successfully!\n\n"
                    f"Repository: {self.config.username}/{repo_name}\n"
                    f"Files uploaded: {upload_result.get('files_uploaded', 0)}\n"
                    f"Branch: {target_branch}\n\n"
                    f"GitHub Pages URL: {pages_url}\n"
                    f"(May take 1-2 minutes to become available.)"
                ),
                repo_url=f"https://github.com/{self.config.username}/{repo_name}",
                pages_url=pages_url,
                deployment_status="queued",
            )

        except Exception as e:
            return DeploymentResult(
                success=False,
                message=f"Publishing failed: {self._mask_token(str(e))}"
            )

    async def _upload_files(
        self,
        client: httpx.AsyncClient,
        repo_name: str,
        files: dict[str, bytes],
        branch: str,
        replace_existing: bool = False,
    ) -> dict[str, Any]:
        """Upload files to a repository using the Git Trees API.

        Uses the tree + commit + update-ref approach for reliable uploads.
        """
        try:
            # Get the latest commit SHA for the branch
            ref_resp = await client.get(f"/repos/{self.config.username}/{repo_name}/git/ref/heads/{branch}")
            if ref_resp.status_code == 404:
                base_tree_sha = None  # Branch doesn't exist yet, start fresh
            elif ref_resp.status_code == 200:
                commit_sha = ref_resp.json().get("object", {}).get("sha")
                # Get the commit's tree
                commit_resp = await client.get(f"/repos/{self.config.username}/{repo_name}/git/commits/{commit_sha}")
                if commit_resp.status_code == 200:
                    base_tree_sha = commit_resp.json().get("tree", {}).get("sha")
                else:
                    base_tree_sha = None
            else:
                base_tree_sha = None

            # Build tree entries
            tree_entries = []
            for file_path, content in files.items():
                encoded = base64.b64encode(content).decode('ascii')
                tree_entries.append({
                    "path": file_path,
                    "mode": "100644",
                    "type": "blob",
                    "content": encoded,
                })

            # Create tree
            tree_payload: dict[str, Any] = {"tree": tree_entries}
            if base_tree_sha and not replace_existing:
                tree_payload["base_tree"] = base_tree_sha

            tree_resp = await client.post(
                f"/repos/{self.config.username}/{repo_name}/git/trees",
                json=tree_payload
            )

            if tree_resp.status_code != 201:
                return {"success": False, "message": f"Failed to create tree: {tree_resp.status_code}"}

            tree_sha = tree_resp.json().get("sha")

            # Create commit
            commit_msg = f"Genie: publish website ({len(files)} files)"
            parents = []
            if ref_resp.status_code == 200:
                parents.append(commit_sha)

            commit_resp = await client.post(
                f"/repos/{self.config.username}/{repo_name}/git/commits",
                json={
                    "message": commit_msg,
                    "tree": tree_sha,
                    "parents": parents,
                }
            )

            if commit_resp.status_code != 201:
                return {"success": False, "message": f"Failed to create commit: {commit_resp.status_code}"}

            commit_sha_new = commit_resp.json().get("sha")

            # Update branch ref
            resp = await client.patch(
                f"/repos/{self.config.username}/{repo_name}/git/refs/heads/{branch}",
                json={"sha": commit_sha_new, "force": False}
            )

            if resp.status_code != 200:
                return {"success": False, "message": f"Failed to update branch: {resp.status_code}"}

            return {"success": True, "files_uploaded": len(files)}

        except Exception as e:
            return {"success": False, "message": self._mask_token(str(e))}

    # ---- Deployment status ----

    async def get_deployment_status(self) -> dict[str, Any]:
        """Check the current GitHub Pages deployment status.

        Returns a dict with deployment status information.
        """
        if not self.config.is_configured:
            return {"success": False, "message": "GitHub is not configured."}

        try:
            client = await self._get_client()
            repo_name = self.config.pages_repository.split("/")[-1] if "/" in self.config.pages_repository else self.config.pages_repository

            # Check Pages info
            pages_resp = await client.get(f"/repos/{self.config.username}/{repo_name}/pages")
            if pages_resp.status_code == 404:
                return {
                    "success": True,
                    "message": "GitHub Pages is not configured for this repository.",
                    "status": "not_configured",
                }
            elif pages_resp.status_code != 200:
                return {"success": False, "message": f"Could not check Pages status: {pages_resp.status_code}"}

            pages_data = pages_resp.json()
            pages_url = pages_data.get("html_url", "")

            # Check latest deployment
            deployments_resp = await client.get(
                f"/repos/{self.config.username}/{repo_name}/deployments",
                params={"per_page": 1, "ref": "main"}
            )

            status = "unknown"
            if deployments_resp.status_code == 200:
                deployments = deployments_resp.json()
                if deployments:
                    latest = deployments[0]
                    # Get deployment status
                    deploy_id = latest.get("id")
                    statuses_resp = await client.get(
                        f"/repos/{self.config.username}/{repo_name}/deployments/{deploy_id}/statuses",
                        params={"per_page": 1}
                    )
                    if statuses_resp.status_code == 200:
                        deploy_statuses = statuses_resp.json()
                        if deploy_statuses:
                            status = deploy_statuses[0].get("state", "unknown")

            return {
                "success": True,
                "status": status,
                "pages_url": pages_url,
                "message": f"Deployment status: {status}. Pages URL: {pages_url}" if pages_url else "Pages not yet available.",
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to check deployment status: {self._mask_token(str(e))}"}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_github_service: GitHubService | None = None


def get_github_service(config: GitHubConfig | None = None) -> GitHubService:
    """Get or create the GitHubService singleton."""
    global _github_service
    if _github_service is None or config is not None:
        if config is None:
            from genie.config import Config
            cfg = Config.load()
            config = GitHubConfig(
                token=cfg.github_token,
                username=cfg.github_username,
                pages_repository=cfg.github_pages_repository,
            )
        _github_service = GitHubService(config)
    return _github_service
