import pytest

from glen_opt.server import mcp

pytestmark = pytest.mark.anyio


async def test_tools_listed():
    names = {t.name for t in await mcp.list_tools()}
    assert names == {"select_parcels", "list_solvers"}
