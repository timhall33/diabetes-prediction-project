"""
NHANES Data Download Module

Downloads XPT files from the CDC NHANES website with retry logic,
progress tracking, and manifest generation.

Usage:
    # As a module
    from src.data.download import download_nhanes_data
    download_nhanes_data(years=["2015-2016", "2017-2018"])

    # As a script
    python -m src.data.download --years 2015-2016 2017-2018
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import yaml
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/data/download.py to project root
    return Path(__file__).parent.parent.parent


def load_yaml_config(config_path: Path) -> dict:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_file_mappings() -> dict:
    """Load the file mappings configuration."""
    config_path = get_project_root() / "config" / "file_mappings.yaml"
    return load_yaml_config(config_path)


def load_main_config() -> dict:
    """Load the main project configuration."""
    config_path = get_project_root() / "config" / "config.yaml"
    return load_yaml_config(config_path)


def get_file_url(
    file_id: str,
    year: str,
    mappings: dict,
) -> Optional[str]:
    """
    Construct the download URL for a specific file and year.

    Args:
        file_id: Logical file identifier (e.g., "DEMO", "GHB")
        year: Survey cycle (e.g., "2015-2016")
        mappings: File mappings configuration

    Returns:
        Full URL to the XPT file, or None if file not available for year
    """
    base_url = mappings["base_url"]
    first_year = mappings["year_to_first_year"].get(year)
    suffix = mappings["year_suffixes"].get(year, "")

    if first_year is None:
        logger.error(f"Unknown year: {year}")
        return None

    # Check if file is not available for this year
    unavailable = mappings.get("files_not_available", {}).get(year, [])
    if file_id in unavailable:
        logger.info(f"File {file_id} not available for {year}")
        return None

    # Check for year-specific override
    overrides = mappings.get("year_overrides", {}).get(year, {})
    if file_id in overrides:
        override_value = overrides[file_id]
        if override_value is None:
            logger.info(f"File {file_id} explicitly marked unavailable for {year}")
            return None
        filename = override_value
    else:
        # Find the file prefix from the standard definitions
        filename = None
        for category in mappings["files"].values():
            for file_def in category:
                if file_def["id"] == file_id:
                    filename = file_def["prefix"] + suffix
                    break
            if filename:
                break

        if filename is None:
            logger.error(f"Unknown file ID: {file_id}")
            return None

    url = f"{base_url}/{first_year}/DataFiles/{filename}.xpt"
    return url


def get_all_file_ids(mappings: dict) -> list[dict]:
    """Get all file IDs with their metadata from the mappings."""
    all_files = []
    for category, files in mappings["files"].items():
        for file_def in files:
            all_files.append({
                "id": file_def["id"],
                "category": category,
                "description": file_def["description"],
                "required": file_def.get("required", False),
            })
    return all_files


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(
    url: str,
    dest_path: Path,
    timeout: int = 60,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> dict:
    """
    Download a file with retry logic and exponential backoff.

    Args:
        url: URL to download from
        dest_path: Local path to save the file
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff

    Returns:
        Dictionary with download status and metadata
    """
    result = {
        "url": url,
        "path": str(dest_path),
        "success": False,
        "attempts": 0,
        "error": None,
        "size_bytes": None,
        "md5": None,
        "download_time": None,
    }

    # Create parent directory if needed
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        result["attempts"] = attempt + 1
        try:
            start_time = time.time()

            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            # Get file size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with open(dest_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=dest_path.name,
                    leave=False,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            elapsed = time.time() - start_time

            # Calculate checksum
            md5 = calculate_md5(dest_path)

            result["success"] = True
            result["size_bytes"] = dest_path.stat().st_size
            result["md5"] = md5
            result["download_time"] = round(elapsed, 2)

            logger.info(
                f"Downloaded {dest_path.name} "
                f"({result['size_bytes']:,} bytes) in {elapsed:.1f}s"
            )
            return result

        except requests.exceptions.RequestException as e:
            result["error"] = str(e)
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}"
            )

            if attempt < max_retries - 1:
                sleep_time = backoff_factor ** attempt
                logger.info(f"Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

    logger.error(f"Failed to download {url} after {max_retries} attempts")
    return result


def download_nhanes_data(
    years: list[str],
    output_dir: Optional[Path] = None,
    file_ids: Optional[list[str]] = None,
    required_only: bool = False,
    skip_existing: bool = True,
    timeout: int = 60,
    max_retries: int = 3,
) -> dict:
    """
    Download NHANES data files for specified years.

    Args:
        years: List of survey cycles to download (e.g., ["2015-2016", "2017-2018"])
        output_dir: Directory to save files (default: data/raw)
        file_ids: Specific file IDs to download (default: all files)
        required_only: If True, only download files marked as required
        skip_existing: If True, skip files that already exist with matching checksum
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts per file

    Returns:
        Dictionary with download summary and manifest
    """
    # Load configurations
    mappings = load_file_mappings()

    # Set output directory
    if output_dir is None:
        output_dir = get_project_root() / "data" / "raw"
    output_dir = Path(output_dir)

    # Get files to download
    all_files = get_all_file_ids(mappings)
    if file_ids:
        all_files = [f for f in all_files if f["id"] in file_ids]
    if required_only:
        all_files = [f for f in all_files if f["required"]]

    # Load existing manifest if available
    manifest_path = output_dir / "manifest.json"
    existing_manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            existing_manifest = json.load(f)

    # Track results
    results = {
        "started_at": datetime.now().isoformat(),
        "years": years,
        "downloads": [],
        "summary": {
            "total": 0,
            "success": 0,
            "skipped": 0,
            "failed": 0,
        },
    }

    # Download each file for each year
    for year in years:
        year_dir = output_dir / year
        year_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading data for {year}")
        logger.info(f"{'='*60}")

        for file_info in all_files:
            file_id = file_info["id"]
            results["summary"]["total"] += 1

            # Get URL
            url = get_file_url(file_id, year, mappings)
            if url is None:
                results["summary"]["skipped"] += 1
                results["downloads"].append({
                    "year": year,
                    "file_id": file_id,
                    "status": "not_available",
                    "description": file_info["description"],
                })
                continue

            # Determine filename from URL
            filename = url.split("/")[-1]
            dest_path = year_dir / filename

            # Check if already downloaded with matching checksum
            manifest_key = f"{year}/{filename}"
            if skip_existing and dest_path.exists():
                if manifest_key in existing_manifest:
                    existing_md5 = existing_manifest[manifest_key].get("md5")
                    current_md5 = calculate_md5(dest_path)
                    if existing_md5 == current_md5:
                        logger.info(f"Skipping {filename} (already exists)")
                        results["summary"]["skipped"] += 1
                        results["downloads"].append({
                            "year": year,
                            "file_id": file_id,
                            "status": "skipped",
                            "path": str(dest_path),
                            "description": file_info["description"],
                        })
                        continue

            # Download the file
            download_result = download_file(
                url=url,
                dest_path=dest_path,
                timeout=timeout,
                max_retries=max_retries,
            )

            # Record result
            download_result["year"] = year
            download_result["file_id"] = file_id
            download_result["description"] = file_info["description"]
            download_result["category"] = file_info["category"]

            if download_result["success"]:
                results["summary"]["success"] += 1
                download_result["status"] = "success"
                # Update manifest
                existing_manifest[manifest_key] = {
                    "md5": download_result["md5"],
                    "size_bytes": download_result["size_bytes"],
                    "downloaded_at": datetime.now().isoformat(),
                    "url": url,
                }
            else:
                results["summary"]["failed"] += 1
                download_result["status"] = "failed"

            results["downloads"].append(download_result)

    # Save manifest
    results["completed_at"] = datetime.now().isoformat()
    with open(manifest_path, "w") as f:
        json.dump(existing_manifest, f, indent=2)
    logger.info(f"\nManifest saved to {manifest_path}")

    # Save download log
    log_path = output_dir / "download_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Download log saved to {log_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files: {results['summary']['total']}")
    logger.info(f"  Success:   {results['summary']['success']}")
    logger.info(f"  Skipped:   {results['summary']['skipped']}")
    logger.info(f"  Failed:    {results['summary']['failed']}")

    # List any failures
    failed = [d for d in results["downloads"] if d.get("status") == "failed"]
    if failed:
        logger.warning("\nFailed downloads:")
        for f in failed:
            logger.warning(f"  - {f['year']}/{f['file_id']}: {f.get('error', 'Unknown error')}")

    return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download NHANES data files")
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2015-2016", "2017-2018"],
        help="Survey cycles to download (default: 2015-2016 2017-2018)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="Only download required files",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-download files even if they exist",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retry attempts per file (default: 3)",
    )

    args = parser.parse_args()

    results = download_nhanes_data(
        years=args.years,
        output_dir=args.output_dir,
        required_only=args.required_only,
        skip_existing=not args.no_skip,
        timeout=args.timeout,
        max_retries=args.retries,
    )

    # Exit with error if any required files failed
    failed_required = [
        d for d in results["downloads"]
        if d.get("status") == "failed" and d.get("required", False)
    ]
    if failed_required:
        exit(1)


if __name__ == "__main__":
    main()
