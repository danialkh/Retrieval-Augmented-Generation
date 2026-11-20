# Exact case-sensitive match
uv run cli/keyword_search_cli.py search "Alien"

# Verify truncation to at most 5 results
uv run cli/keyword_search_cli.py search "The"

# Verify case sensitivity (lowercase usually returns fewer or no matches)
uv run cli/keyword_search_cli.py search "alien"

# Verify help output still works when no subcommand is passed
uv run cli/keyword_search_cli.py