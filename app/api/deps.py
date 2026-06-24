from fastapi import Header


async def get_current_user(x_user_id: str = Header(default="default")) -> str:
    """User identity for tenant isolation. No auth layer — the caller supplies
    an `X-User-Id` header; every repository still filters on this id so users
    never see each other's data. Defaults to 'default' when absent."""
    return x_user_id
