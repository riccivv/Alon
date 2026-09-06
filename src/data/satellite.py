"""Resumable NASA satellite data downloader.

Uses ``earthaccess`` to search NASA CMR for ocean-color granules over the
Philippine study region and download them into ``data/raw/<dataset_key>/``.

Design goals:
    * Resumable: already-downloaded granules (on disk OR in the manifest)
      are skipped across runs, so interrupted batches pick up where they left.
    * Resilient: downloads use explicit timeouts, bounded per-file retries
      with backoff, and per-file error collection (a flaky connection fails
      one file, not the whole batch).
    * Traceable: a manifest CSV records every granule we fetched.
    * Parallel: granules download using our own thread pool.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import logging
import random
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import earthaccess
import pandas as pd
import requests
from tqdm import tqdm

from src.config import (
    EARTHDATA_PASSWORD,
    EARTHDATA_USERNAME,
    PHILIPPINES_BBOX,
    RAW_DATA_DIR,
    SATELLITE_DATASETS,
    SATELLITE_DEFAULT_TEMPORAL,
)

logger = logging.getLogger(__name__)

MANIFEST_PATH = RAW_DATA_DIR / "manifest.csv"

# Per-file download policy.
DOWNLOAD_ATTEMPTS = 5          # retries per file before giving up
CONNECT_TIMEOUT = 15           # seconds to establish the TCP/TLS connection
READ_TIMEOUT = 120             # seconds of silence before aborting the stream
RETRY_BACKOFF_BASE = 2         # exponential backoff base (seconds)

# Batch-level retry policy: if every file fails (host looks down), wait a
# growing pause and try the remaining files again. The run only gives up after
# exhausting all passes, so a temporarily unreachable host is ridden out.
MAX_BATCH_PASSES = 6           # total download passes over remaining files
FIRST_PASS_PAUSE = 10          # seconds to wait before the 2nd pass
MAX_PASS_PAUSE = 300           # cap on the pause between passes

# Cloud-hosted (AWS) collections for the same products. The Level-3 mapped
# ocean-color data are mirrored in NASA's cloud and served from
# obdaac-tea.earthdatacloud.nasa.gov (a different, often reachable host when
# the on-prem oceandata.sci.gsfc.nasa.gov host is unreachable). SST is NOT
# cloud-hosted. Cloud granules are authenticated with a Bearer token.
CLOUD_SHORT_NAMES = {
    "chlorophyll": "MODISA_L3m_CHL",   # same short name as on-prem
    "par": "MODISA_L3m_PAR",           # same short name
    "kd490": "MODISA_L3m_KD",          # renamed in the cloud collection
}
CLOUD_BUCKET_HOST = "earthdatacloud.nasa.gov"


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds as MM:SS (or H:MM:SS if long)."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
_auth: object | None = None


def login() -> None:
    """Authenticate with NASA Earthdata using .netrc or env credentials."""
    import os

    # earthaccess >=0.19 credentials come from env vars or ~/.netrc only.
    if EARTHDATA_USERNAME and EARTHDATA_PASSWORD:
        os.environ.setdefault("EARTHDATA_USERNAME", EARTHDATA_USERNAME)
        os.environ.setdefault("EARTHDATA_PASSWORD", EARTHDATA_PASSWORD)

    auth = earthaccess.login(strategy="all", persist=True)
    global _auth
    _auth = auth
    authenticated = getattr(auth, "authenticated", None)
    if isinstance(authenticated, bool):
        ok = authenticated
    elif callable(authenticated):
        ok = authenticated()
    else:
        ok = bool(auth)
    if not ok:
        # Fall back to an interactive .netrc creation.
        raise RuntimeError(
            "earthaccess authentication failed. Register at "
            "https://urs.earthdata.nasa.gov/ then run: "
            'python -c "import earthaccess; earthaccess.login()"'
            " to create ~/.netrc, or fill EARTHDATA_USERNAME/PASSWORD in .env"
        )
    logger.info("Earthdata authentication successful.")


# --------------------------------------------------------------------------- #
# Manifest bookkeeping
# --------------------------------------------------------------------------- #
def load_manifest() -> pd.DataFrame:
    """Return the existing download manifest (empty DataFrame if none)."""
    if MANIFEST_PATH.exists():
        return pd.read_csv(MANIFEST_PATH, dtype={"granule_name": str})
    return pd.DataFrame(
        columns=[
            "short_name",
            "granule_name",
            "start_time",
            "end_time",
            "cloud_cover",
            "local_path",
            "size_bytes",
        ]
    )


def append_manifest(df: pd.DataFrame) -> None:
    """Append a DataFrame of new downloads to the manifest CSV."""
    current = load_manifest()
    merged = pd.concat([current, df], ignore_index=True).drop_duplicates(
        subset=["short_name", "granule_name"], keep="last"
    )
    merged.to_csv(MANIFEST_PATH, index=False)


# --------------------------------------------------------------------------- #
# Granule search
# --------------------------------------------------------------------------- #
def _parse_granule_time(name: str) -> dt.datetime | None:
    """Extract the first 8-digit YYYYMMDD timestamp from a granule name."""
    match = re.search(r"(\d{8})", name)
    if not match:
        return None
    try:
        return dt.datetime.strptime(match.group(1), "%Y%m%d")
    except ValueError:
        return None


def search_granules(
    short_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    bbox: tuple[float, float, float, float] = PHILIPPINES_BBOX,
    cloud_cover: tuple[int, int] | None = None,
    granule_pattern: str | None = None,
    count: int | None = None,
    cloud_hosted: bool = False,
) -> list[earthaccess.results.DataGranule]:
    """Search NASA CMR for granules of a dataset over the study region.

    ``cloud_cover`` when combined with ``bounding_box`` and ``temporal`` can
    silently return zero results on some OB.DAAC collections, so it is disabled
    by default. L3 8-day composites already average cloud-free pixels; residual
    gaps are handled during preprocessing (temporal interpolation).
    """
    start_date = start_date or SATELLITE_DEFAULT_TEMPORAL[0]
    end_date = end_date or SATELLITE_DEFAULT_TEMPORAL[1]

    kwargs = dict(
        short_name=short_name,
        temporal=(start_date, end_date),
        bounding_box=bbox,
        cloud_hosted=cloud_hosted,
    )
    if cloud_cover is not None:
        kwargs["cloud_cover"] = cloud_cover
    if granule_pattern:
        kwargs["granule_name"] = granule_pattern
    if count:
        kwargs["count"] = count

    results = earthaccess.search_data(**kwargs)
    logger.info("Found %s granules for %s (%s -> %s)", len(results), short_name, start_date, end_date)
    return results


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _granule_filename(granule) -> str:
    """Return the local filename a granule should be saved under."""
    try:
        links = granule.data_links() or []
    except Exception:  # noqa: BLE001
        try:
            links = granule.dataLinks() or []
        except Exception:  # noqa: BLE001
            links = []
    for href in links:
        name = Path(urlsplit(href).path).name
        if name and name.endswith(".nc"):
            return name
    return granule["umm"]["GranuleUR"].rsplit("/", 1)[-1]


def _granule_url(granule) -> str | None:
    """Pick a downloadable HTTPS link for an on-prem granule."""
    try:
        links = granule.data_links() or []
    except Exception:  # noqa: BLE001
        try:
            links = granule.dataLinks() or []
        except Exception:  # noqa: BLE001
            links = []
    for href in links:
        if "getfile" in href or "archive" in href or "cmr" in href:
            return href
    return links[0] if links else None


def _bearer_token() -> str:
    """Return the current earthaccess access token (or '' if unavailable)."""
    try:
        token = (_auth or earthaccess.Auth()).token
    except Exception:  # noqa: BLE001
        return ""
    return token.get("access_token", "") if isinstance(token, dict) else ""


_thread_local = threading.local()


def _thread_session() -> requests.Session:
    """A pooled connection per worker thread (keep-alive, no re-handshake)."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1, pool_maxsize=4, max_retries=0
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return session


def _download_one(args: tuple[str, Path, str]) -> tuple[Path, bool, str]:
    """Download one file with bounded retries. Returns (dest, ok, error).

    Uses a per-thread pooled :class:`requests.Session` so concurrent transfers
    reuse their TCP/TLS connection instead of re-handshaking every file.
    """
    url, dest, token = args
    headers = {"User-Agent": "OceanForecast-PH/0.1 (fisheries-yield)"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_error = ""
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with _thread_session().get(
                url, headers=headers, stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            ) as response:
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}"
                    response.close()
                else:
                    content_type = response.headers.get("Content-Type", "").lower()
                    if "text/html" in content_type:
                        # Some NASA hosts return an HTML redirect/error page with
                        # HTTP 200; never persist it as a .nc stub (we saw this
                        # with SST -- 162-byte HTML files with .nc names).
                        last_error = "Server returned HTML instead of a granule"
                        response.close()
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        tmp = dest.with_suffix(dest.suffix + ".part")
                        with open(tmp, "wb") as fh:
                            for chunk in response.iter_content(chunk_size=1 << 16):
                                if chunk:
                                    fh.write(chunk)
                        tmp.replace(dest)
                        if dest.stat().st_size == 0:
                            dest.unlink()
                            last_error = "Downloaded an empty file"
                        else:
                            return dest, True, ""
        except (requests.Timeout, requests.ConnectionError,
                requests.exceptions.ChunkedEncodingError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:100]}"
            time.sleep(RETRY_BACKOFF_BASE ** attempt + random.uniform(0, 1))
    return dest, False, last_error


def _download_granules(
    granules: list,
    local_dir: Path,
    token: str,
    threads: int = 4,
    show_progress: bool = True,
) -> tuple[list[Path], list[tuple[str, str]]]:
    """Download granules resumably; never abort the batch on one failure.

    Files that fail are retried across a few passes: if a pass makes no
    progress at all (host looks down), we wait a growing pause, then retry the
    remaining files. The run only gives up after ``MAX_BATCH_PASSES``.

    Returns ``(downloaded, failures)`` where ``failures`` is a list of
    ``(granule_name, error_message)`` for files that still need retrying.
    """
    tasks: list[tuple[str, Path, str]] = []
    for g in granules:
        fname = _granule_filename(g)
        url = _granule_url(g)
        if not url:
            logger.warning("No download link for %s; skipping.", fname)
            continue
        dest = local_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            logger.debug("On disk already: %s", fname)
            continue
        tasks.append((url, dest, token))

    if not tasks:
        return [], []

    logger.info("Queueing %s file(s) for download (%s threads)", len(tasks), threads)
    t0 = dt.datetime.now()
    downloaded: list[Path] = []
    pending = list(tasks)
    last_error: dict[Path, str] = {}
    early_abort_at = max(10, threads * 3)

    for pass_no in range(1, MAX_BATCH_PASSES + 1):
        if not pending:
            break
        this_pass = pending
        pending = []
        pass_downloaded: list[Path] = []
        consecutive = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
            futures = {pool.submit(_download_one, task): task for task in this_pass}
            iterator = concurrent.futures.as_completed(futures)
            if show_progress:
                iterator = tqdm(
                    iterator, total=len(futures),
                    desc=f"Downloading (pass {pass_no})",
                    unit="file", smoothing=0.05,
                )
            for future in iterator:
                task = futures[future]
                try:
                    path, ok, msg = future.result()
                except Exception as exc:  # noqa: BLE001
                    last_error[task[1]] = f"{type(exc).__name__}: {exc}"
                    consecutive += 1
                    continue
                if ok:
                    pass_downloaded.append(path)
                    consecutive = 0
                else:
                    last_error[task[1]] = msg
                    consecutive += 1
                    # No downloads at all across enough tries: host is down.
                    if not pass_downloaded and consecutive >= early_abort_at:
                        for f in futures:
                            f.cancel()
                        break

        downloaded.extend(pass_downloaded)
        done_names = {p.name for p in pass_downloaded}
        pending = [task for task in this_pass if task[1].name not in done_names]

        if pending:
            if pass_no == MAX_BATCH_PASSES:
                logger.error(
                    "Giving up after %s passes with %s file(s) undownloaded "
                    "(e.g. %s). Host unreachable from this network.",
                    MAX_BATCH_PASSES, len(pending), pending[0][0],
                )
                raise RuntimeError(
                    f"Satellite host unreachable after {MAX_BATCH_PASSES} passes: "
                    f"{pending[0][1].name} -> {pending[0][0]}"
                )
            pause = min(FIRST_PASS_PAUSE * (2 ** (pass_no - 1)), MAX_PASS_PAUSE)
            logger.warning(
                "Host looks down (0 files in pass %s of a %s-file batch). "
                "Waiting %ss then retrying %s remaining file(s).",
                pass_no, len(tasks), pause, len(pending),
            )
            time.sleep(pause)

    failures = [
        (task[1].name, last_error.get(task[1], "not attempted"))
        for task in pending
    ]
    logger.info(
        "Downloaded %s file(s) in %s (%s failures)",
        len(downloaded), _fmt_elapsed((dt.datetime.now() - t0).total_seconds()),
        len(failures),
    )
    return downloaded, failures


def download_dataset(
    dataset_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    bbox: tuple[float, float, float, float] = PHILIPPINES_BBOX,
    cloud_cover: tuple[int, int] | None = None,
    max_files: int | None = None,
    skip_existing: bool = True,
    threads: int = 4,
    show_progress: bool = True,
    cloud_hosted: bool = False,
) -> pd.DataFrame:
    """Download all granules of a configured dataset.

    Downloads are resumable (on-disk files are skipped even if a previous
    batch died before updating the manifest) and per-file resilient: one
    connection failure does not abort the batch; failed files are reported
    so a re-run can finish them.

    When ``cloud_hosted`` is set, granules are pulled from the AWS-hosted
    collection (``obdaac-tea.earthdatacloud.nasa.gov``) instead of the on-prem
    ``oceandata.sci.gsfc.nasa.gov`` host. File names are identical.

    Returns a DataFrame of newly downloaded granules (for the manifest).
    """
    if dataset_key not in SATELLITE_DATASETS:
        raise KeyError(
            f"Unknown dataset key {dataset_key!r}. Valid: {list(SATELLITE_DATASETS)}"
        )

    cfg = SATELLITE_DATASETS[dataset_key]
    short_name = cfg["short_name"]
    if cloud_hosted:
        cloud_short_name = CLOUD_SHORT_NAMES.get(dataset_key, short_name)
        if cloud_short_name != short_name:
            logger.info(
                "Using cloud collection for %s (short_name %s)",
                dataset_key, cloud_short_name,
            )
        short_name = cloud_short_name
    local_dir = RAW_DATA_DIR / dataset_key
    local_dir.mkdir(parents=True, exist_ok=True)

    search_results = search_granules(
        short_name=short_name,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        cloud_cover=cloud_cover,
        granule_pattern=cfg.get("granule_pattern"),
        count=max_files,
        cloud_hosted=cloud_hosted,
    )
    if not search_results:
        logger.warning("No granules found for %s", short_name)
        return pd.DataFrame()

    if max_files:
        search_results = search_results[:max_files]

    # Determine which granules we still need.
    manifest = load_manifest()
    if skip_existing and not manifest.empty:
        existing = set(
            manifest.loc[manifest["short_name"] == short_name, "granule_name"]
        )
        search_results = [g for g in search_results if g["umm"]["GranuleUR"] not in existing]

    logger.info("Downloading %s new granules to %s", len(search_results), local_dir)
    if not search_results:
        logger.info("All granules already present. Nothing to do.")
        return pd.DataFrame()

    t0 = dt.datetime.now()
    token = _bearer_token()
    try:
        _downloaded, failures = _download_granules(
            search_results, local_dir, token, threads=threads, show_progress=show_progress
        )
    except RuntimeError:
        # Host unreachable; still record any granules that landed.
        _downloaded, failures = [], []
    elapsed = (dt.datetime.now() - t0).total_seconds()

    if failures:
        logger.warning(
            "Could not download %s file(s) after %s attempts each: %s",
            len(failures), DOWNLOAD_ATTEMPTS,
            ", ".join(name for name, _ in failures[:5]) + (" ..." if len(failures) > 5 else ""),
        )
        logger.info("Re-run the script to retry the missing files (resumable).")

    # Build manifest rows from every granule whose target now exists on disk.
    granule_name = lambda g: g["umm"]["GranuleUR"]  # noqa: E731
    records = []
    for granule in search_results:
        p = local_dir / _granule_filename(granule)
        if not p.exists():
            continue
        meta = granule.get("umm", {})
        records.append(
            {
                "short_name": short_name,
                "granule_name": granule_name(granule),
                "start_time": (meta.get("TemporalExtent", {}) or {}).get("RangeDateTime", {}).get("BeginningDateTime"),
                "end_time": (meta.get("TemporalExtent", {}) or {}).get("RangeDateTime", {}).get("EndingDateTime"),
                "cloud_cover": (meta.get("DataGranule", {}) or {}).get("CloudCover"),
                "local_path": str(p),
                "size_bytes": p.stat().st_size if p.exists() else None,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        append_manifest(df)
        total_mb = df["size_bytes"].sum() / (1024**2)
        logger.info(
            "Downloaded %s granules in %s (~%.1f MB) for %s",
            len(df),
            _fmt_elapsed(elapsed),
            total_mb,
            dataset_key,
        )
    return df


def download_all_datasets(
    start_date: str | None = None,
    end_date: str | None = None,
    max_files_per_dataset: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Download every configured satellite dataset."""
    login()
    results = {}
    for key in SATELLITE_DATASETS:
        logger.info("=== Downloading dataset: %s ===", key)
        results[key] = download_dataset(
            key, start_date=start_date, end_date=end_date, max_files=max_files_per_dataset
        )
    return results


# Convenience alias used by CLI scripts.
fetch_dataset = download_dataset