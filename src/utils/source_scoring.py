"""
Project Argus - Source Credibility Scoring

Evaluates a given URL and assigns a static credibility score and 
source type classification based on domain authority rules.
"""

from urllib.parse import urlparse
from src.config import TRUSTED_DOMAINS

# Tier 2.5: Major News Outlets (Score 0.7)
MAJOR_NEWS_DOMAINS = {
    "reuters.com", "bloomberg.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "wsj.com",
    "cnbc.com", "nbcnews.com", "apnews.com",
    "theguardian.com", "ft.com", "economist.com",
    "sciencedaily.com", "arstechnica.com",
}

# Tier 3: Reputable Industry/Market Research (Score 0.65)
INDUSTRY_DOMAINS = {
    "marketsandmarkets.com", "grandviewresearch.com",
    "researchandmarkets.com", "mordorintelligence.com",
    "statista.com", "iea.org", "irena.org",
    "mckinsey.com", "bcg.com", "deloitte.com",
    "pwc.com", "kpmg.com",
    "yahoo.com",  # yahoo finance/news
    "cars.com", "notebookcheck.net",
}

def evaluate_source(url: str) -> dict:
    """
    Evaluates the credibility of a URL based on its domain.
    
    Returns:
        dict: {
            "score": float (0.0 to 1.0),
            "type": str (Classification of the source)
        }
    """
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower().removeprefix("www.")
    except Exception:
        hostname = ""
        
    if not hostname:
        return {"score": 0.4, "type": "Unverified/Web"}
        
    # Tier 1: Peer-reviewed / Academic (Score 1.0)
    for trusted in TRUSTED_DOMAINS:
        if hostname == trusted or hostname.endswith("." + trusted):
            return {"score": 1.0, "type": "Academic/Scientific"}
            
    if hostname.endswith(".edu"):
        return {"score": 1.0, "type": "Academic/Scientific"}

    # Tier 2: Government / Institutional (Score 0.9)
    if hostname.endswith(".gov"):
        return {"score": 0.9, "type": "Government/Institutional"}
        
    # Tier 2.5: Major News (Score 0.7)
    for news_domain in MAJOR_NEWS_DOMAINS:
        if hostname == news_domain or hostname.endswith("." + news_domain):
            return {"score": 0.7, "type": "Major News"}

    # Tier 3: Industry/Market Research (Score 0.65)
    for ind_domain in INDUSTRY_DOMAINS:
        if hostname == ind_domain or hostname.endswith("." + ind_domain):
            return {"score": 0.65, "type": "Industry/Market Research"}

    # Tier 4: Reputable Industry/News (Score 0.6)
    # We apply a slight boost to .org (non-profits) and .io / standard reputable domains
    # In a full production system, this would explicitly list Bloomberg, Reuters, etc.
    if hostname.endswith(".org"):
        return {"score": 0.6, "type": "News/Industry"}
        
    # Tier 5: Everything Else / Generic Web (Score 0.4)
    return {"score": 0.4, "type": "Unverified/Web"}
