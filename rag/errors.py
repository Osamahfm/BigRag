class RAGError(Exception):
    """Safe, actionable message for the UI; never include raw API errors."""
