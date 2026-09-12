from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]


def combine_filing_data(
    metadata,
    facts,
    contexts,
):
    metadata = metadata.copy()
    facts = facts.copy()
    contexts = contexts.copy()

    required_metadata = {
        "filing_key",
        "symbol",
        "period_end",
    }

    required_facts = {
        "filing_key",
        "context_ref",
    }

    required_contexts = {
        "filing_key",
        "context_ref",
    }

    for required, df, name in [
        (
            required_metadata,
            metadata,
            "metadata",
        ),
        (
            required_facts,
            facts,
            "facts",
        ),
        (
            required_contexts,
            contexts,
            "contexts",
        ),
    ]:
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing {name} columns: "
                f"{sorted(missing)}"
            )

    if metadata["filing_key"].duplicated().any():
        raise ValueError(
            "Duplicate filing keys in metadata."
        )

    facts = facts.merge(
        contexts,
        on=[
            "filing_key",
            "context_ref",
        ],
        how="left",
        validate="many_to_one",
        suffixes=("", "_context"),
    )

    filing_columns = [
        "filing_key",
        "symbol",
        "company_name",
        "period_end",
        "filing_type",
        "filing_sub_type",
        "consolidated",
        "audited",
        "broadcast_date",
        "creation_date",
        "revised_date",
        "revision_remark",
    ]

    available = [
        column
        for column in filing_columns
        if column in metadata.columns
    ]

    facts = facts.merge(
        metadata[available],
        on="filing_key",
        how="left",
        validate="many_to_one",
    )

    return facts


def validate_filing_dataset(
    metadata,
    facts,
    contexts,
):
    if metadata.empty:
        raise ValueError(
            "Financial metadata is empty."
        )

    if facts.empty:
        raise ValueError(
            "Financial facts are empty."
        )

    if contexts.empty:
        raise ValueError(
            "Financial contexts are empty."
        )

    if metadata["filing_key"].isna().any():
        raise ValueError(
            "Missing filing keys in metadata."
        )

    if facts["filing_key"].isna().any():
        raise ValueError(
            "Missing filing keys in facts."
        )

    if contexts["filing_key"].isna().any():
        raise ValueError(
            "Missing filing keys in contexts."
        )

    metadata_keys = set(
        metadata["filing_key"]
    )

    fact_keys = set(
        facts["filing_key"]
    )

    context_keys = set(
        contexts["filing_key"]
    )

    if not fact_keys.issubset(metadata_keys):
        raise ValueError(
            "Facts contain unknown filing keys."
        )

    if not context_keys.issubset(metadata_keys):
        raise ValueError(
            "Contexts contain unknown filing keys."
        )

    return True
