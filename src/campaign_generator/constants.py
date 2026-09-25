"""Shared application constants for the data and evaluation workflow."""

import re

# Published Zarn split names; this order keeps generated files predictable.
SPLITS = ("train", "validation", "test")

# Reserve held-out rows first when an identical brief appears in multiple splits.
DEDUP_PRIORITY = ("test", "validation", "train")

# Flags directions about asset workflow for the data-quality report.
PROCESS_WORDS = re.compile(r"\b(asset plan|first release|package|review cycle|handoff)\b", re.I)

# Identifies a clear conflict in the candidate Brainstorming examples.
NO_PAID_INFLUENCER = re.compile(r"no paid influencers", re.I)

# Matches the influencer suggestion that violates that constraint.
INFLUENCER_SUGGESTION = re.compile(r"(micro[- ]?influencer|paid influencer)", re.I)

# Labels download requests in Hugging Face logs and troubleshooting output.
DATA_USER_AGENT = "campaign-generator-day1/0.1"

# Versioned so changing enrichment instructions creates a distinct cache key.
ENRICHMENT_PROMPT_VERSION = "zarn-to-brainstorm-v3"
