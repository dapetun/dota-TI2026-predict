"""Download match details in background with rate limiting."""
import sys, json, time, os, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_collection.opendota_api import OpenDotaClient

client = OpenDotaClient(rate_limit=1.1)
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def download_all():
    """Download all match details with caching."""
    all_ids = set()
    for f in glob.glob(str(RAW_DIR / "*_matches.json")):
        with open(f) as fh:
            data = json.load(fh)
            for m in data:
                all_ids.add(str(m["match_id"]))

    print(f"Total unique match IDs: {len(all_ids)}")

    # Load existing details
    details_file = RAW_DIR / "match_details.json"
    existing = {}
    if details_file.exists():
        with open(details_file) as f:
            existing = json.load(f)
        print(f"Already downloaded: {len(existing)}")

    to_download = sorted(all_ids - set(existing.keys()))
    print(f"To download: {len(to_download)}")

    if not to_download:
        print("All done!")
        return

    # Download with rate limiting
    batch_size = 55  # stay under 60/min
    errors = 0

    for i, mid in enumerate(to_download):
        if i > 0 and i % batch_size == 0:
            print(f"  Progress: {i}/{len(to_download)}, sleeping 65s...")
            time.sleep(65)

        try:
            detail = client.get_match(int(mid))
            if detail and "error" not in detail:
                existing[mid] = detail
            else:
                errors += 1
        except Exception as e:
            errors += 1
            if "429" in str(e) or "rate" in str(e).lower():
                print(f"  Rate limited at {i}, sleeping 65s...")
                time.sleep(65)
            continue

        # Save every 100
        if (i + 1) % 100 == 0:
            with open(details_file, "w") as f:
                json.dump(existing, f)
            print(f"  Saved {len(existing)} details ({errors} errors)")

    # Final save
    with open(details_file, "w") as f:
        json.dump(existing, f)
    print(f"\nDone! Total: {len(existing)} details, {errors} errors")


if __name__ == "__main__":
    download_all()
