import copy
import csv
import io
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from paths import HERE


def extract_csv_from_zips(zip_files, destination_folder):
    """
    Extract all CSV files from the given zip files into the destination folder.
    
    Args:
        zip_files: List of paths to zip files
        destination_folder: Path to destination folder
    """
    destination = Path(destination_folder)
    destination.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    
    for zip_path in zip_files:
        zip_path = Path(zip_path)
        
        if not zip_path.exists():
            print(f"Warning: {zip_path} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {zip_path.name}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get all files in the zip
                all_files = zip_ref.namelist()
                
                # Filter for CSV files
                csv_files = [f for f in all_files if f.lower().endswith('.csv')]
                
                if not csv_files:
                    print(f"  No CSV files found in {zip_path.name}")
                    continue
                
                # Extract each CSV file
                for csv_file in csv_files:
                    # Extract to destination folder
                    zip_ref.extract(csv_file, destination)
                    
                    # If the file is in a subdirectory, move it to the root of destination
                    extracted_path = destination / csv_file
                    final_path = destination / Path(csv_file).name
                    
                    if extracted_path != final_path:
                        # Move file to destination root
                        final_path.parent.mkdir(parents=True, exist_ok=True)
                        extracted_path.rename(final_path)
                        print(f"  ✓ Extracted and moved: {csv_file} -> {final_path.name}")
                        
                        # Clean up empty parent directories recursively
                        parent = extracted_path.parent
                        while parent != destination and parent.exists():
                            try:
                                # Only remove if empty
                                if not any(parent.iterdir()):
                                    parent.rmdir()
                                    parent = parent.parent
                                else:
                                    break
                            except (OSError, PermissionError):
                                break
                    else:
                        print(f"  ✓ Extracted: {csv_file}")
                    
                    extracted_count += 1
                    
        except zipfile.BadZipFile:
            print(f"Error: {zip_path.name} is not a valid zip file")
        except Exception as e:
            print(f"Error processing {zip_path.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Extraction complete! Total CSV files extracted: {extracted_count}")
    print(f"Destination folder: {destination.absolute()}")
    print(f"{'='*60}")


def _detect_csv_separator(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def _prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}{str(c)}" for c in df.columns]
    return df


def merge_rendements_archives_to_csv(zip_files, output_csv_path: Path) -> None:
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
        sample_text = raw_bytes[:65536].decode("utf-8", errors="ignore")
        sep = _detect_csv_separator(sample_text)
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), sep=sep)
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw_bytes), sep=sep, encoding="latin1")

    def _read_excel_bytes(raw_bytes: bytes) -> tuple[str, pd.DataFrame]:
        xls = pd.ExcelFile(io.BytesIO(raw_bytes))
        chosen_sheet: str | None = None
        chosen_df: pd.DataFrame | None = None
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            if df is None:
                continue
            df = df.dropna(how="all")
            if len(df) == 0:
                continue
            chosen_sheet = str(sheet)
            chosen_df = df
            break
        if chosen_sheet is None or chosen_df is None:
            chosen_sheet = str(xls.sheet_names[0]) if xls.sheet_names else "Sheet1"
            chosen_df = xls.parse(chosen_sheet)
        return chosen_sheet, chosen_df

    dataframes: list[pd.DataFrame] = []
    lengths: list[tuple[str, int]] = []

    for zip_path in zip_files:
        zip_path = Path(zip_path)
        if not zip_path.exists():
            print(f"Warning: {zip_path} does not exist, skipping...")
            continue

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                members = [m for m in zip_ref.namelist() if not m.endswith("/")]
                members = [
                    m
                    for m in members
                    if Path(m).suffix.lower() in {".csv", ".xls", ".xlsx", ".xlsm"}
                ]
                members.sort(key=lambda p: p.lower())

                if not members:
                    print(f"  No CSV/Excel files found in {zip_path.name}")
                    continue

                for member in members:
                    raw = zip_ref.read(member)
                    suffix = Path(member).suffix.lower()
                    base_prefix = f"{zip_path.stem}__{Path(member).stem}__"

                    if suffix == ".csv":
                        df = _read_csv_bytes(raw)
                        df = _prefix_columns(df, base_prefix)
                        dataframes.append(df)
                        lengths.append((f"{zip_path.name}:{member}", len(df)))
                        continue

                    try:
                        sheet_name, df = _read_excel_bytes(raw)
                    except ImportError as e:
                        raise ImportError(
                            "Reading Excel files requires an Excel engine. Install 'openpyxl' for .xlsx "
                            "and 'xlrd' for .xls."
                        ) from e
                    except Exception:
                        df = _read_csv_bytes(raw)
                        df = _prefix_columns(df, base_prefix)
                        dataframes.append(df)
                        lengths.append((f"{zip_path.name}:{member}", len(df)))
                        continue

                    sheet_prefix = f"{base_prefix}{sheet_name}__"
                    df = _prefix_columns(df, sheet_prefix)
                    dataframes.append(df)
                    lengths.append((f"{zip_path.name}:{member}:{sheet_name}", len(df)))
        except ImportError as e:
            print(f"Error processing {zip_path.name}: {e}")
            raise
        except zipfile.BadZipFile:
            print(f"Error: {zip_path.name} is not a valid zip file")
        except Exception as e:
            print(f"Error processing {zip_path.name}: {e}")

    if not dataframes:
        print("No data found to merge; randement.csv was not created.")
        return

    unique_lengths = {l for _, l in lengths}
    if len(unique_lengths) != 1:
        details = "\n".join([f"  - {name}: {l} rows" for name, l in lengths])
        raise ValueError(
            "Cannot merge rendements files horizontally because row counts differ:\n" + details
        )

    merged = pd.concat(dataframes, axis=1)
    merged.to_csv(output_csv_path, index=False)
    print(f"\nMerged rendements written to: {output_csv_path.absolute()}")


if __name__ == "__main__":

    base_dir = copy.deepcopy(HERE)
    
    zip_data_dir = base_dir / "default_data"
    
    # Find all zip files in the default_data directory
    zip_files = list(zip_data_dir.glob("*.zip"))
    
    if not zip_files:
        print(f"Warning: No zip files found in {zip_data_dir}")
        # Fallback to specific files if directory doesn't exist
        zip_files = [
            zip_data_dir / "data_1.zip",
            zip_data_dir / "data_2.zip",
            zip_data_dir / "data_3.zip",
            zip_data_dir / "data_4.zip",
            zip_data_dir / "data_5.zip",
            zip_data_dir / "test-benoit.zip",
        ]
    
    # Define destination folder (you can change this)
    destination_folder = base_dir / "data_in"
    
    print("CSV Extraction Script")
    print(f"{'='*60}")
    print(f"Base directory: {base_dir}")
    print(f"Source zip files: {len(zip_files)}")
    print(f"Destination: {destination_folder}")
    print(f"{'='*60}")
    
    # Separate test-benoit.zip from other files
    test_benoit_zip = zip_data_dir / "test-benoit.zip"
    regular_zip_files = [zf for zf in zip_files if zf.name != "test-benoit.zip"]
    
    # Extract regular zip files to data_in
    if regular_zip_files:
        extract_csv_from_zips(regular_zip_files, destination_folder)
    
    # Extract test-benoit.zip to separate folder
    if test_benoit_zip.exists():
        test_benoit_destination = base_dir / "data_in_test_benoit"
        print(f"\n{'='*60}")
        print(f"Extracting test-benoit.zip to separate folder:")
        print(f"{'='*60}")
        extract_csv_from_zips([test_benoit_zip], test_benoit_destination)

    rendements_int_src = zip_data_dir / "rendements_int.csv"
    rendements_int_dst = destination_folder / "RENDEMENTS_INT.csv"
    if rendements_int_src.exists():
        shutil.copy2(rendements_int_src, rendements_int_dst)
        print(f"Copied: {rendements_int_src.name} -> {rendements_int_dst}")
    else:
        print(f"Warning: {rendements_int_src} does not exist, skipping...")

    rendements_zips = [
        zip_data_dir / "rendements.zip",
        zip_data_dir / "rendements1.zip",
        zip_data_dir / "rendements2.zip",
        zip_data_dir / "rendements3.zip",
    ]
    merge_rendements_archives_to_csv(
        rendements_zips,
        base_dir / "data_in" / "RENDEMENTS.csv",
    )
