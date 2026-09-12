from datetime import datetime, timezone
from pathlib import Path
import json
import re

import pandas as pd
import requests
from io import BytesIO
from lxml import etree


PROJECT_DIR = Path(__file__).resolve().parents[3]

RAW_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "nse"
    / "financials"
)

XBRL_RAW_DIR = RAW_DIR / "xbrl"
METADATA_DIR = RAW_DIR / "metadata"

NSE_INTEGRATED_FILINGS_API = (
    "https://www.nseindia.com/api/integrated-filing-results"
)

NSE_INTEGRATED_FILING_PAGE = (
    "https://www.nseindia.com/companies-listing/"
    "corporate-integrated-filing"
)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
}

STRUCTURAL_LOCAL_NAMES = {
    "xbrl",
    "context",
    "entity",
    "identifier",
    "period",
    "startDate",
    "endDate",
    "instant",
    "scenario",
    "explicitMember",
    "typedMember",
    "unit",
    "measure",
    "divide",
    "unitNumerator",
    "unitDenominator",
    "schemaRef",
    "roleRef",
    "arcroleRef",
}


def fetch_integrated_filings(
    symbol=None,
    page=1,
    size=20,
):
    params = {
        "type": "Integrated Filing- Financials",
        "index": "equities",
        "page": page,
        "size": size,
    }

    if symbol:
        params["symbol"] = symbol

    response = requests.get(
        NSE_INTEGRATED_FILINGS_API,
        params=params,
        headers=NSE_HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    return response.json()


def collect_filing_metadata(
    page_size=100,
    max_pages=None,
    symbol=None,
):
    records = []
    page = 1

    while True:
        payload = fetch_integrated_filings(
            symbol=symbol,
            page=page,
            size=page_size,
        )

        batch = payload.get("data", [])
        total = payload.get("totalCount", 0)

        if not batch:
            break

        records.extend(batch)

        if max_pages and page >= max_pages:
            break

        if len(records) >= total:
            break

        page += 1

    return records


def fetch_xbrl(xbrl_url):
    response = requests.get(
        xbrl_url,
        headers=NSE_HEADERS,
        timeout=60,
    )
    response.raise_for_status()

    return response.content


def parse_xbrl_facts(xbrl_content):
    root = etree.parse(
        BytesIO(xbrl_content)
    ).getroot()

    rows = []

    for element in root.iter():
        local_name = etree.QName(element).localname

        if local_name in STRUCTURAL_LOCAL_NAMES:
            continue

        context_ref = element.get("contextRef")

        if not context_ref:
            continue

        rows.append(
            {
                "tag": local_name,
                "qualified_tag": (
                    etree.QName(element).text
                ),
                "value_raw": element.text,
                "context_ref": context_ref,
                "unit_ref": element.get("unitRef"),
                "decimals": element.get("decimals"),
            }
        )

    return pd.DataFrame(rows), root


def parse_xbrl_contexts(root):
    rows = []

    for context in root.xpath(
        "//*[local-name()='context']"
    ):
        context_id = context.get("id")

        identifier = context.xpath(
            ".//*[local-name()='identifier']/text()"
        )

        start_date = context.xpath(
            ".//*[local-name()='startDate']/text()"
        )

        end_date = context.xpath(
            ".//*[local-name()='endDate']/text()"
        )

        instant = context.xpath(
            ".//*[local-name()='instant']/text()"
        )

        dimensions = []

        for member in context.xpath(
            ".//*[local-name()='explicitMember' "
            "or local-name()='typedMember']"
        ):
            dimensions.append(
                {
                    "dimension": member.get("dimension"),
                    "member": "".join(member.itertext()).strip(),
                }
            )

        rows.append(
            {
                "context_ref": context_id,
                "entity_identifier": (
                    identifier[0]
                    if identifier
                    else None
                ),
                "period_start": (
                    start_date[0]
                    if start_date
                    else None
                ),
                "period_end": (
                    end_date[0]
                    if end_date
                    else None
                ),
                "instant": (
                    instant[0]
                    if instant
                    else None
                ),
                "dimensions": dimensions,
            }
        )

    return pd.DataFrame(rows)


def safe_filing_key(record):
    key = (
        record.get("seq_Id")
        or record.get("seq_id")
        or (
            f"{record.get('symbol', 'unknown')}_"
            f"{record.get('qe_Date', 'unknown')}"
        )
    )

    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(key),
    )


def build_filing_metadata(record):
    return {
        "filing_key": safe_filing_key(record),
        "symbol": record.get("symbol"),
        "company_name": record.get("cmName"),
        "seq_id": record.get("seq_Id"),
        "period_end": record.get("qe_Date"),
        "filing_type": record.get("type"),
        "filing_sub_type": record.get("type_Sub"),
        "consolidated": record.get("consolidated"),
        "audited": record.get("audited"),
        "broadcast_date": record.get("broadcast_Date"),
        "creation_date": record.get("creation_Date"),
        "revised_date": record.get("revised_Date"),
        "revision_remark": record.get("revisionRemark"),
        "xbrl_url": record.get("xbrl"),
        "ixbrl_url": record.get("ixbrl"),
        "xbrl_file_size": record.get("xbrlFileSize"),
    }


def collect_filing_package(record):
    metadata = build_filing_metadata(record)

    xbrl_content = fetch_xbrl(
        metadata["xbrl_url"]
    )

    facts_df, root = parse_xbrl_facts(
        xbrl_content
    )

    contexts_df = parse_xbrl_contexts(root)

    facts_df["filing_key"] = metadata["filing_key"]
    facts_df["symbol"] = metadata["symbol"]
    facts_df["period_end"] = metadata["period_end"]

    contexts_df["filing_key"] = metadata["filing_key"]
    contexts_df["symbol"] = metadata["symbol"]

    return (
        metadata,
        facts_df,
        contexts_df,
        xbrl_content,
    )


def save_filing_package(
    metadata,
    facts_df,
    contexts_df,
    xbrl_content,
):
    XBRL_RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filing_key = metadata["filing_key"]

    xbrl_path = (
        XBRL_RAW_DIR /
        f"{filing_key}.xml"
    )

    xbrl_path.write_bytes(xbrl_content)

    facts_path = (
        XBRL_RAW_DIR.parent /
        "xbrl_facts.parquet"
    )

    contexts_path = (
        XBRL_RAW_DIR.parent /
        "xbrl_contexts.parquet"
    )

    facts_df.to_parquet(
        facts_path,
        index=False,
    )

    contexts_df.to_parquet(
        contexts_path,
        index=False,
    )

    return xbrl_path


def acquire_financial_filings(
    records,
):
    metadata_rows = []
    fact_frames = []
    context_frames = []
    manifest_rows = []

    for record in records:
        try:
            (
                metadata,
                facts,
                contexts,
                xbrl_content,
            ) = collect_filing_package(record)

            xbrl_path = save_filing_package(
                metadata,
                facts,
                contexts,
                xbrl_content,
            )

            metadata_rows.append(metadata)
            fact_frames.append(facts)
            context_frames.append(contexts)

            manifest_rows.append(
                {
                    "provider": "National Stock Exchange of India Limited (NSE)",
                    "dataset": "Integrated Filing XBRL",
                    "filing_key": metadata["filing_key"],
                    "symbol": metadata["symbol"],
                    "period_end": metadata["period_end"],
                    "source_url": metadata["xbrl_url"],
                    "local_file": str(xbrl_path),
                    "accessed_at_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "status": "downloaded",
                }
            )

        except Exception as exc:
            manifest_rows.append(
                {
                    "filing_key": safe_filing_key(record),
                    "symbol": record.get("symbol"),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    metadata_df = pd.DataFrame(metadata_rows)

    facts_df = (
        pd.concat(
            fact_frames,
            ignore_index=True,
        )
        if fact_frames
        else pd.DataFrame()
    )

    contexts_df = (
        pd.concat(
            context_frames,
            ignore_index=True,
        )
        if context_frames
        else pd.DataFrame()
    )

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    return (
        metadata_df,
        facts_df,
        contexts_df,
        manifest_df,
    )
