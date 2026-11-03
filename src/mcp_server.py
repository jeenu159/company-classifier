from mcp.server.fastmcp import FastMCP
from tools import fda_profile as _fda_profile, web_search as _web_search

mcp = FastMCP("company-intel")


@mcp.tool()
def fda_profile(company: str) -> dict:
    """Summarise a pharma company's FDA drug labels: product types (Rx/OTC), share of generic products,
    share of labels made by OTHER companies (a repackager signal), homeopathic / sunscreen / repackaged
    signals, and sample brand names. Pass the exact manufacturer name as it appears in openFDA."""
    return _fda_profile(company)


@mcp.tool()
def web_search(company: str) -> dict:
    """Search the web for what a company's business actually is (manufacturer, distributor, retailer...).
    Returns 3 short snippets with titles and URLs."""
    return _web_search(company)


if __name__ == "__main__":
    mcp.run(transport="stdio")