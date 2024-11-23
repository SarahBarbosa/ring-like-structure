import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import Angle
import warnings


# Constants
DATA_PATH_GCSI = "data/gcs1.fits"
SIMBAD_QUERY_FILE = "resources/simbad_query.txt"
STAR_NAMES_FILE = "resources/stars_names.txt"
OUTPUT_DIR = "data"
OUTPUT_STARS = os.path.join(OUTPUT_DIR, "gcs-allstars.csv")
OUTPUT_F_FNAME = os.path.join(OUTPUT_DIR, "gcs-Fstars.csv")
OUTPUT_G_FNAME = os.path.join(OUTPUT_DIR, "gcs-Gstars.csv")


def suppress_fits_warnings():
    """
    Suppress VerifyWarning from astropy.io.fits to avoid unnecessary warnings.
    """
    warnings.simplefilter('ignore', category=fits.verify.VerifyWarning)


def load_fits_data(file_path: str) -> pd.DataFrame:
    """
    Load FITS data and convert to a pandas DataFrame.
    """
    with fits.open(file_path) as hdul:
        fits_table = hdul[1].data
    df = pd.DataFrame.from_records(fits_table)
    return df


def display_initial_statistics(df: pd.DataFrame):
    """
    Display initial statistics of the Geneva-Copenhagen Survey data.
    """
    total_stars = len(df)
    total_parameters = len(df.columns)
    stars_with_vsini = len(df[df['vsini'] != 0])
    stars_with_mass = len(df[df['mass'] != 0])
    mass_min = df['mass'].min()
    mass_max = df['mass'].max()

    print("## Geneva-Copenhagen Survey Original Data:")
    print(f"- Number of stars: {total_stars}")
    print(f"- Number of parameters: {total_parameters}")
    print(f"- Stars with vsini data: {stars_with_vsini}")
    print(f"- Stars with mass data: {stars_with_mass}")
    print(f"- Mass ranges from {mass_min} to {mass_max} M⊙\n")


def convert_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert RA and DEC from separate columns to single numeric columns in degrees.
    """

    def convert_ra(row):
        """
        Convert RA from hours, minutes, seconds to degrees.
        """
        ra_str = f"{int(row['RAh'])} {int(row['RAm'])} {row['RAs']}"
        ra_deg = Angle(ra_str, unit='hourangle').degree
        return ra_deg

    def convert_dec(row):
        """
        Convert DEC from degrees, arcminutes, arcseconds to degrees.
        """
        dec_sign = row['DE_'].strip()
        dec_str = f"{dec_sign}{int(row['DEd'])} {int(row['DEm'])} {row['DEs']}"
        dec_deg = Angle(dec_str, unit='degree').degree
        return dec_deg

    df["RA"] = df.apply(convert_ra, axis=1)
    df["DEC"] = df.apply(convert_dec, axis=1)
    return df


def compute_cartesian_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Cartesian coordinates (X, Y, Z) from Galactic longitude and latitude.
    """
    # Equations (1) to (3) from our paper
    df["X"] = df["Dist"] * np.cos(np.radians(df["GLAT"])) * np.cos(np.radians(df["GLON"]))
    df["Y"] = df["Dist"] * np.cos(np.radians(df["GLAT"])) * np.sin(np.radians(df["GLON"]))
    df["Z"] = df["Dist"] * np.sin(np.radians(df["GLAT"]))
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning by removing binaries and giants, and ensuring well-defined vsini and distance.
    """
    # Flags:
    # FB: Confirmed/suspected binaries
    # FD: Spectroscopic binaries
    # COMP: Multiple system components
    # FG: Suspected giants

    # Identify and exclude binaries
    binaries_condition = (df["fb"] != "*") & (df["fd"] != "*") & (df["Comp"] == '    ')
    df_without_binaries = df[binaries_condition]
    binaries_num = len(df) - len(df_without_binaries)
    binaries_percent = (binaries_num / len(df)) * 100
    print(f"- Number of binary targets: {binaries_num} ({binaries_percent:.2f}%)")

    # Identify and exclude suspected giants
    giants_condition = df_without_binaries["fg"] != "*"
    df_without_giants = df_without_binaries[giants_condition]

    # Ensure vsini is well-defined
    df_welldefined_vsini = df_without_giants[df_without_giants["vsini"] != 0]

    # Ensure distance is non-zero
    df_clean = df_welldefined_vsini[df_welldefined_vsini["Dist"] != 0].reset_index(drop=True)

    print(f"- Number of suspected giant targets: {len(df[df['fg'] == '*'])}")
    print(f"- Number of stars after cleaning: {len(df_clean)}\n")

    return df_clean


def prepare_simbad_data(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Merge SIMBAD query results with the cleaned GCS I data.
    """
    # Write star names to a file for SIMBAD querying
    with open(STAR_NAMES_FILE, 'w') as file:
        for name in df_clean['Name']:
            file.write(f"{name.strip()}\n")

    # Avoiding unnecessary empty spaces
    df_clean["Name"] = df_clean["Name"].str.strip()

    # Check if SIMBAD_QUERY_FILE exists
    if not os.path.isfile(SIMBAD_QUERY_FILE):
        error_message = (
            "Error: SIMBAD query results not found.\n"
            f"It looks like you didn't save SIMBAD query results as '{SIMBAD_QUERY_FILE}'.\n"
            "Please perform a SIMBAD query by identifier using the star names saved in "
            f"'{STAR_NAMES_FILE}' and save the results as '{SIMBAD_QUERY_FILE}'.\n"
            "You can access SIMBAD at https://simbad.cds.unistra.fr/simbad/."
        )
        raise FileNotFoundError(error_message)

    # Read SIMBAD query results
    try:
        simbad_df = pd.read_csv(SIMBAD_QUERY_FILE, delimiter="|")
    except Exception as e:
        raise ValueError(
            f"Error reading SIMBAD query results from '{SIMBAD_QUERY_FILE}': {e}"
        )

    # Read SIMBAD query results
    simbad_df = pd.read_csv(SIMBAD_QUERY_FILE, delimiter="|")
    simbad_df["typed ident"] = simbad_df["typed ident"].str.strip()

    # Clean column names
    simbad_df.columns = simbad_df.columns.str.strip()

    # Rename columns for clarity
    simbad_df = simbad_df.rename(columns={
        "typed ident": "Name",
        "typ": "obj_type",
        "spec. type": "spec_type"
    })

    # Strip whitespace from relevant columns
    simbad_df["obj_type"] = simbad_df["obj_type"].str.strip()
    simbad_df["spec_type"] = simbad_df["spec_type"].str.strip()

    # Merge with GCS I data
    df_merged = df_clean.merge(
        simbad_df[["Name", "obj_type", "spec_type"]],
        on="Name",
        how="left"
    )

    obj_type_counts = df_merged["obj_type"].value_counts()
    print(f"- SIMBAD object type distribution:\n{obj_type_counts}\n")

    # Filter based on object types "PM*" and "*"
    valid_obj_types = ["PM*", "*"]
    df_filtered = df_merged[
        df_merged["obj_type"].isin(valid_obj_types)
    ]

    return df_filtered


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finalize the dataset by selecting necessary columns and filtering for F and G spectral types.
    """
    # Select necessary columns
    selected_columns = [
        "Name", "GLON", "GLAT", "Vmag",
        "logTe", "_Fe_H_", "Dist",
        "Age", "clAge", "chAge",
        "mass", "clmass", "chmass",
        "vsini", "X", "Y", "Z", "RA", "DEC",
        "obj_type", "spec_type"
    ]
    df_final = df[selected_columns].copy()

    # Filter for F and G spectral types
    df_final = df_final[
        df_final["spec_type"].str.startswith(("F", "G"))
    ].reset_index(drop=True)

    # Separate into F and G type stars
    df_F = df_final[df_final["spec_type"].str.startswith("F")].reset_index(drop=True)
    df_G = df_final[df_final["spec_type"].str.startswith("G")].reset_index(drop=True)

    print(f"## Final Dataset Summary:")
    print(f"- Total number of stars: {len(df_final)}")
    print(f"- Number of F-type stars: {len(df_F)}")
    print(f"- Number of G-type stars: {len(df_G)}\n")

    return df_final, df_F, df_G


def export_data(df_final: pd.DataFrame, df_F: pd.DataFrame, df_G: pd.DataFrame):
    """
    Export the final all, F and G type star datasets to CSV files.

    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_final.to_csv(OUTPUT_STARS, index=False)
    df_F.to_csv(OUTPUT_F_FNAME, index=False)
    df_G.to_csv(OUTPUT_G_FNAME, index=False)
    print(f"- Exported all type stars to {OUTPUT_STARS}")
    print(f"- Exported F-type stars to {OUTPUT_F_FNAME}")
    print(f"- Exported G-type stars to {OUTPUT_G_FNAME}")


def main():
    print("### Geneva-Copenhagen Survey (GCS I) Data Processing ###\n")

    # Suppress FITS verification warnings
    suppress_fits_warnings()

    # Load data
    gcsi_df = load_fits_data(DATA_PATH_GCSI)
    display_initial_statistics(gcsi_df)

    # Convert coordinates
    gcsi_df = convert_coordinates(gcsi_df)

    # Compute Cartesian coordinates
    gcsi_df = compute_cartesian_coordinates(gcsi_df)

    # Drop original RA and DEC components
    gcsi_df.drop(columns=["RAh", "RAm", "RAs", "DE_", "DEd", "DEm", "DEs"], inplace=True)

    # Clean data
    gcsi_clean = clean_dataframe(gcsi_df)

    # Prepare and merge SIMBAD data
    gcsi_merged = prepare_simbad_data(gcsi_clean)

    # Finalize dataset
    df_final, df_F, df_G = finalize_dataset(gcsi_merged)

    # Export final data
    export_data(df_final, df_F, df_G)

    print("\n### Data Processing Completed Successfully ###")


if __name__ == "__main__":
    main()
