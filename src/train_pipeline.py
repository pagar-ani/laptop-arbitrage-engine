import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Set basic configurations for visualization and output
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# GOOGLE DRIVE PERSISTENCE LAYER (REMOVED)
# ==========================================

def sync_to_drive():
    """Mocked function to prevent downstream NameErrors after removing Drive logic."""
    pass

# ==========================================
# NEXT CELL
# ==========================================

# Load the dataset
# Adjust the file path if your dataset is in a different directory in your cloud environment
# V2 FIX [M1]: Count raw lines to detect silent row drops during ingestion
with open('Kaggle_Master_Dataset.csv', 'r', encoding='utf-8', errors='replace') as _f:
    _raw_line_count = sum(1 for _ in _f) - 1  # subtract header
df = pd.read_csv('Kaggle_Master_Dataset.csv', engine='python', on_bad_lines='skip')
_skipped = _raw_line_count - len(df)
if _skipped > 0:
    print(f"[WARNING] {_skipped} malformed rows silently dropped during CSV ingestion.")
else:
    print(f"[INGESTION] {len(df)} rows loaded cleanly. Zero malformed lines.")

# Display the first 5 rows
print(df.head())

# ==========================================
# NEXT CELL
# ==========================================

print(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n")

# Display column names, non-null counts, and data types
df.info()


# ==========================================
# NEXT CELL
# ==========================================

# Summary statistics for numerical columns (like price, ram_size_gb, storage_size_gb)
print(df.describe())

# Summary statistics for categorical columns (like brand, city, state, cpu, gpu)
print(df.describe(include=['object']))

# ==========================================
# NEXT CELL
# ==========================================

import numpy as np

# 1. Standardize missing value placeholders to true NaN
# V2 FIX [M2]: Consolidated placeholder list from both cleaning passes into single canonical set
placeholders = ["unknown", "Unknown", "", " ", "-", "N/A", "NA", "n/a", "na", "null", "Null", "none", "None"]

# Replace these exact matches with np.nan â€” restrict to object columns only
# to preserve CPU cache locality on numeric dtype blocks
cat_cols_placeholder = df.select_dtypes(include=['object']).columns
df[cat_cols_placeholder] = df[cat_cols_placeholder].replace(placeholders, np.nan)
df[cat_cols_placeholder] = df[cat_cols_placeholder].replace(r'^\s*$', np.nan, regex=True)

# 2. Now calculate the number and percentage of true missing values
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percent
})

# Filter out columns with no missing values and sort
missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)

print("Columns with missing values (after converting 'unknown' and blanks to NaN):")
print(missing_data)

# 3. Visualize missing data using seaborn
if not missing_data.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_data['Percentage (%)'], y=missing_data.index, palette='viridis')
    plt.title('Percentage of Missing Values per Column')
    plt.xlabel('Percentage (%)')
    plt.ylabel('Columns')
    plt.show()

# ==========================================
# NEXT CELL
# ==========================================

plt.figure(figsize=(12, 5))

# Plot the distribution of laptop prices
plt.subplot(1, 2, 1)
sns.histplot(df['price'], bins=50, kde=True, color='blue')
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')

# Plot using a logarithmic scale to handle long-tail/outliers
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['price']), bins=50, kde=True, color='green')
plt.title('Log Distribution of Price')
plt.xlabel('Log(Price + 1)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# ==========================================
# NEXT CELL
# ==========================================

# Create a copy of the dataframe before filtering, so you can track how many rows are dropped
df_clean = df.copy()
initial_rows = len(df_clean)

# 1. Filter Price: Remove prices below 5,000
# Assuming price is in INR or a currency where < 5k doesn't make sense for a functional laptop
df_clean = df_clean[df_clean['price'] >= 5000]

# 2. Filter Storage: Remove 0 GB storage and anything greater than 8 TB (8192 GB)
# We will use 8192 GB as the strict 8TB cutoff, but 8000 works fine too
df_clean = df_clean[(df_clean['storage_size_gb'] > 0) & (df_clean['storage_size_gb'] <= 8192)]

# 3. Filter RAM: Remove 0 GB RAM and anything greater than 200 GB
df_clean = df_clean[(df_clean['ram_size_gb'] > 0) & (df_clean['ram_size_gb'] <= 200)]

# Calculate how many rows were removed
final_rows = len(df_clean)
dropped_rows = initial_rows - final_rows

print(f"Initial number of records: {initial_rows}")
print(f"Cleaned number of records: {final_rows}")
print(f"Total records removed: {dropped_rows} ({dropped_rows/initial_rows*100:.2f}%)")

# Optional: Re-assign to 'df' if you want to use 'df' for the rest of the pipeline
df = df_clean

# ==========================================
# NEXT CELL
# ==========================================

import re

# Define our keyword dictionaries
# We use lowercase because we will convert the text to lowercase before matching
BRANDS = [
    'hp', 'dell', 'lenovo', 'acer', 'asus', 'apple', 'macbook',
    'msi', 'razer', 'microsoft', 'surface', 'samsung', 'alienware'
]

def has_multiple_brands(text):
    """Returns True if the text mentions 2 or more distinct laptop brands."""
    if not isinstance(text, str):
        return False

    text = text.lower()
    found_brands = 0

    for brand in BRANDS:
        # \b ensures we match whole words only (e.g., 'acer' not 'tracer')
        if re.search(rf'\b{brand}\b', text):
            found_brands += 1

        if found_brands >= 2:
            return True

    return False

def has_conflicting_cpus(text):
    """Returns True if the text mentions both Intel and AMD processors."""
    if not isinstance(text, str):
        return False

    text = text.lower()

    # Check for Intel identifiers
    has_intel = bool(re.search(r'\b(intel|i3|i5|i7|i9|core i)\b', text))

    # Check for AMD identifiers
    has_amd = bool(re.search(r'\b(amd|ryzen|athlon)\b', text))

    return has_intel and has_amd

# ---------------------------------------------------------
# Apply the filters
# ---------------------------------------------------------

initial_rows = len(df)

# Create boolean masks (True means we found keyword stuffing, so we want to DROP it)
mask_title_brands = df['title'].apply(has_multiple_brands)
mask_desc_brands = df['description'].apply(has_multiple_brands)

mask_title_cpus = df['title'].apply(has_conflicting_cpus)
mask_desc_cpus = df['description'].apply(has_conflicting_cpus)

# Combine all conditions: Drop if ANY of these rules are True
drop_mask = mask_title_brands | mask_desc_brands | mask_title_cpus | mask_desc_cpus

# Keep the rows where drop_mask is False (~)
df_clean = df[~drop_mask].copy()

# Calculate statistics
final_rows = len(df_clean)
dropped_rows = initial_rows - final_rows

print(f"Initial records before text filtering: {initial_rows}")
print(f"Records with keyword stuffing (removed): {dropped_rows}")
print(f"Cleaned records remaining: {final_rows}")

# Reassign to df
df = df_clean

# ==========================================
# NEXT CELL
# ==========================================

# ---------------------------------------------------------
# 1. Verify SSD Mentions
# ---------------------------------------------------------
initial_rows = len(df)

# Vectorized SSD mention detection â€” bypasses GIL row-iteration
# by executing at C-level through pandas str accessor (SIMD path)
mask_title_ssd = df['title'].str.contains('ssd', case=False, na=False)
mask_desc_ssd = df['description'].str.contains('ssd', case=False, na=False)
mentions_ssd = mask_title_ssd | mask_desc_ssd

# Find rows that claim their storage_tech is SSD
# (Using fillna('') to prevent errors if storage_tech has NaNs)
claims_ssd = df['storage_tech'].fillna('').str.lower() == 'ssd'

# Find the invalid rows: It claims to be SSD, but NEVER mentions it
invalid_ssd_mask = claims_ssd & ~mentions_ssd

# Keep rows that are NOT invalid
df = df[~invalid_ssd_mask].copy()

invalid_ssd_dropped = initial_rows - len(df)
print(f"Removed {invalid_ssd_dropped} records that claimed SSD but didn't mention it in Title/Desc.")

# ---------------------------------------------------------
# 2. Remove Duplicates
# ---------------------------------------------------------
rows_before_dup = len(df)

# TEMPORAL CAUSALITY FIX: Sort by created_at ascending BEFORE dedup
# so that drop_duplicates(keep='first') retains the earliest listing,
# preventing future states from overwriting past baseline data.
if 'created_at' in df.columns:
    df = df.sort_values('created_at', ascending=True, na_position='last')

# Step A: Drop exact identical rows across all columns
df = df.drop_duplicates()

# Step B: Drop duplicates based on the unique ad identifier
# (Assuming 'ad_id' is your unique column based on standard Kaggle structures)
if 'ad_id' in df.columns:
    # We keep the 'first' occurrence and drop the rest
    df = df.drop_duplicates(subset=['ad_id'], keep='first')

duplicates_dropped = rows_before_dup - len(df)
print(f"Removed {duplicates_dropped} duplicate records.")

# Final summary
print("-" * 30)
print(f"Final clean dataset size: {len(df)}")

# ==========================================
# NEXT CELL
# ==========================================

import numpy as np

initial_rows = len(df)

# 1. Standardize all garbage/blank strings to true np.nan first
drop_placeholders = ["", " ", "n/a", "N/A", "NA", "na", "-", "null", "Null"]
# Restrict placeholder scan to object columns only â€” prevent numeric dtype corruption
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].replace(drop_placeholders, np.nan)
df[cat_cols] = df[cat_cols].replace(r'^\s*$', np.nan, regex=True)

# 2. Drop the row ONLY if the Price or Title is genuinely missing
df.dropna(subset=['price', 'title'], inplace=True)

dropped_critical = initial_rows - len(df)

# 3. Fill NaN only in specific categorical columns that downstream regex taxonomy needs as strings.
# Numeric columns KEEP NaN to preserve pandas' native np.nan handling and prevent type coercion.
categorical_fill_cols = ['gpu', 'cpu', 'brand', 'model_series', 'storage_tech',
                         'city', 'state', 'fingerprint', 'touch', 'warranty',
                         'urgent_sell', 'description', 'display_size']
for _col in categorical_fill_cols:
    if _col in df.columns:
        df[_col] = df[_col].fillna('unknown')

print(f"Removed {dropped_critical} rows that were missing a Price or Title.")
print(f"Remaining clean records: {len(df)}")

# Verify the stuffing worked
if 'gpu' in df.columns:
    unknown_count = (df['gpu'] == 'unknown').sum()
    print(f"Verification: There are now {unknown_count} records with 'unknown' in the GPU column.")

# ==========================================
# NEXT CELL
# ==========================================

# 1. Drop the columns we no longer need for the ML pipeline
# V2 FIX [M3]: ORDERING DEPENDENCY â€” 'title' is consumed by keyword-stuffing filter (lines 189-193).
# This drop MUST execute AFTER the text-based filtering cells above.
cols_to_drop = ['display_tech', 'title', 'url', 'ad_id']
df_working = df.drop(columns=cols_to_drop, errors='ignore').copy()

# 2. Cross-City Deduplication
initial_rows = len(df_working)

# We want to identify duplicates based on the laptop's core features and description.
# We IGNORE city, state, and created_at because these change slightly when a seller spams postings.
ignore_for_dedup = ['city', 'state', 'created_at']

# Get the list of columns to strictly check for exact matches
subset_cols = [col for col in df_working.columns if col not in ignore_for_dedup]

# Drop duplicates (keeping the first occurrence we find)
df_working = df_working.drop_duplicates(subset=subset_cols, keep='first')

# Calculate the spam listings we removed
spam_dropped = initial_rows - len(df_working)

print("--- Working DataFrame Created ---")
print(f"Columns dropped: {cols_to_drop}")
print(f"Cross-city spam listings removed: {spam_dropped}")
print(f"Final df_working size: {len(df_working)} rows x {df_working.shape[1]} columns")

# Let's see what columns are left for our ML pipeline
print("\nRemaining Columns:")
print(df_working.columns.tolist())

# ==========================================
# NEXT CELL
# ==========================================

import pandas as pd

initial_rows = len(df_working)

# 1. Identify rows where ANY of the core specs are labeled as 'unknown'
# We convert to string and lowercase just to be completely safe
mask_cpu = df_working['cpu'].astype(str).str.lower() == 'unknown'
mask_ram = df_working['ram_size_gb'].astype(str).str.lower() == 'unknown'
mask_storage = df_working['storage_size_gb'].astype(str).str.lower() == 'unknown'

# Combine masks: if CPU OR RAM OR Storage is unknown, we flag it
invalid_specs_mask = mask_cpu | mask_ram | mask_storage

# 2. Filter the dataframe to KEEP only rows that are NOT flagged
df_working = df_working[~invalid_specs_mask].copy()

# 3. Convert RAM and Storage back to pure numeric (float/int) types
# Now that the string "unknown" is gone from these columns, we can safely convert them
df_working['ram_size_gb'] = pd.to_numeric(df_working['ram_size_gb'], errors='coerce')
df_working['storage_size_gb'] = pd.to_numeric(df_working['storage_size_gb'], errors='coerce')

# Optional: drop any new NaNs that might have appeared if there was weird text in numeric columns
df_working.dropna(subset=['ram_size_gb', 'storage_size_gb'], inplace=True)

# 4. Calculate and print the results
dropped_rows = initial_rows - len(df_working)
print(f"Removed {dropped_rows} records missing critical specs (CPU, RAM, or Storage).")
print(f"Remaining ML-ready records: {len(df_working)}\n")

# Verify the data types are fixed
print("Data Types of Core Specs:")
print(df_working[['cpu', 'ram_size_gb', 'storage_size_gb']].dtypes)

# ==========================================
# NEXT CELL
# ==========================================

import pandas as pd

cols_to_check = ['brand', 'cpu', 'gpu', 'model_series']

for col in cols_to_check:
    if col in df_working.columns:
        # Get unique values, dropping NaNs
        unique_vals = df_working[col].dropna().unique()

        # Try to sort them alphabetically for easier reading
        try:
            unique_vals = sorted(list(unique_vals))
        except TypeError:
            # Fallback if there are mixed types that can't be sorted
            unique_vals = list(unique_vals)

        # Convert the list into a single-column DataFrame
        unique_df = pd.DataFrame({f'Unique_{col.capitalize()}': unique_vals})

        # Define the file name
        file_name = f"unique_{col}s.csv"

        # Export to CSV (index=False prevents pandas from writing row numbers)
#         unique_df.to_csv(file_name, index=False)

        print(f"Successfully exported {len(unique_vals)} unique values to '{file_name}'")

# Optional: If you also want to see the FREQUENCY of each value exported to CSV
# (This is often more useful for ML so you know which rare values to group together)
for col in cols_to_check:
    if col in df_working.columns:
        # Get counts of each unique value
        val_counts_df = df_working[col].value_counts().reset_index()
        val_counts_df.columns = [col, 'Frequency']

        count_file_name = f"counts_{col}s.csv"
#         val_counts_df.to_csv(count_file_name, index=False)

        print(f"Successfully exported frequency counts to '{count_file_name}'")

# ==========================================
# NEXT CELL
# ==========================================

import re

def compile_hardware_pipeline(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # 1. Isolate the unique entries to minimize regex execution cycles
    unique_entries = df[target_col].dropna().unique()

    # 2. Define compilation rules ordered from highest to lowest specificity
    taxonomy = [
        (r'\b(mac\s?book|mac|apple|imac|a1466|a2337|a1278|a2251)\b', 'Apple Mac'),
        (r'\b(spectre)\b', 'HP Spectre'),
        (r'\b(envy)\b', 'HP Envy'),
        (r'\b(thinkpad|think\s?pad|t\d{3}|x(?!360\b)\d{3}|l\d{3}|p\d{2}|t14s?|x1|thinpad)\b', 'Lenovo ThinkPad'),
        (r'\b(thinkbook|think\s?book)\b', 'Lenovo ThinkBook'),
        (r'\b(ideapad|idea\s?pad|slim\s?3|slim\s?5|v14|v15|81h5)\b', 'Lenovo IdeaPad'),
        (r'\b(loq|legion)\b', 'Lenovo Gaming (LOQ/Legion)'),
        (r'\b(latitude|lattitude|letitude|latutide|e54\d{2}|e64\d{2}|e74\d{2})\b', 'Dell Latitude'),
        (r'\b(vostro|vastro|vodtro)\b', 'Dell Vostro'),
        (r'\b(inspiron|inspirion|insprion|inspration)\b', 'Dell Inspiron'),
        (r'\b(optiplex|optiflex|optipelx|optelex|optiples)\b', 'Dell OptiPlex'),
        (r'\b(precision|presicion|preciaion)\b', 'Dell Precision'),
        (r'\b(xps)\b', 'Dell XPS'),
        (r'\b(vivobook|vivobock|viva\s?book|x540|x515|x415)\b', 'Asus VivoBook'),
        (r'\b(zenbook)\b', 'Asus ZenBook'),
        (r'\b(tuf|tuff)\b', 'Asus TUF Gaming'),
        (r'\b(rog|strix)\b', 'Asus ROG'),
        (r'\b(probook|pro\s?book)\b', 'HP ProBook'),
        (r'\b(pavilion|pavillion|pavalion|pavellion)\b', 'HP Pavilion'),
        (r'\b(victus)\b', 'HP Victus'),
        (r'\b(omen)\b', 'HP Omen'),
        (r'\b(elitebook|elite\s?book|eleitbook|elightbook|elitbook|1030|840|640)\b', 'HP EliteBook'),
        (r'\b(zbook)\b', 'HP ZBook'),
        (r'\b(aspire)\b', 'Acer Aspire'),
        (r'\b(nitro)\b', 'Acer Nitro'),
        (r'\b(predator)\b', 'Acer Predator'),
        (r'\b(extensa|veriton)\b', 'Acer Extensa/Veriton'),
        (r'\b(galaxy\s?book)\b', 'Samsung Galaxy Book'),
        (r'\b(inbook|zerobook)\b', 'Infinix InBook'),
        (r'\b(primebook)\b', 'Primebook'),
        (r'\b(travelmate|travel\s?mate|travellite)\b', 'TravelMate'),
        (r'\b(yoga)\b', 'Lenovo Yoga'),
        (r'\b(assembled|custom|desktop|mini\s?pc|tower)\b', 'Assembled / Custom Desktop Chassis'),
        (r'^(14s|15s|14|15|16|17|240|245|250|255|15-\w+|14-\w+)\b', 'Generic OEM Chassis Series')
    ]

    # Pre-compile patterns into binary instructions for regex engine velocity
    compiled_taxonomy = [(re.compile(pattern, re.IGNORECASE), replacement) for pattern, replacement in taxonomy]

    # 3. Construct conversion matrix via single linear parsing matrix pass
    mapping_dict = {}
    for entry in unique_entries:
        entry_str = str(entry).strip()
        matched = False
        for pattern, replacement in compiled_taxonomy:
            if pattern.search(entry_str):
                mapping_dict[entry] = replacement
                matched = True
                break
        if not matched:
            # This will neatly catch the "unknown" strings we stuffed earlier
            mapping_dict[entry] = 'Other / Unclassified Hardware'

    # 4. Inject structural conversion mapping instantly into raw series vector
    df['clean_model_series'] = df[target_col].map(mapping_dict).astype('category')
    return df

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------

# Apply the pipeline to df_working
df_working = compile_hardware_pipeline(df_working, 'model_series')

# See the results!
print("--- Taxonomy Classification Results ---")
print(df_working['clean_model_series'].value_counts())

# Now we can drop the old messy column if we want to
# df_working.drop(columns=['model_series'], inplace=True)

# ==========================================
# NEXT CELL
# ==========================================

import pandas as pd
import numpy as np
import re

def extract_gpu_topology_matrix(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # Isolate distinct strings to minimize execution footprints
    unique_keys = df[target_col].dropna().unique()

    # Compile high-performance structural lookahead arrays
    discrete_pattern = re.compile(r'\b(rtx|gtx|rx|quadro|t\d{3,4}|arc|firepro|mobility\s?radeon|t500|mx\d{3})\b', re.IGNORECASE)
    intel_pattern = re.compile(r'\b(intel|uhd|hd|iris|arc)\b', re.IGNORECASE)
    amd_pattern = re.compile(r'\b(amd|radeon|vega|rx|apu)\b', re.IGNORECASE)
    nvidia_pattern = re.compile(r'\b(nvidia|nvida|nvdia|gtx|rtx|quadro|geforce|gforce|mx)\b', re.IGNORECASE)

    topology_lookup = {}

    for raw in unique_keys:
        clean = str(raw).lower()

        # 1. Classify Silicon Topology (Discrete vs Integrated)
        # Force default to 0 (integrated) unless explicit high-power dGPU markers exist
        is_discrete = 1 if discrete_pattern.search(clean) else 0
        if 'integrated' in clean or 'igpu' in clean or 'inbuilt' in clean or 'shared' in clean:
            is_discrete = 0

        # 2. Extract Silicon Vendor Flags
        vendor = 'UNKNOWN'
        if nvidia_pattern.search(clean):
            vendor = 'NVIDIA'
        elif amd_pattern.search(clean):
            vendor = 'AMD'
        elif intel_pattern.search(clean):
            vendor = 'INTEL'

        topology_lookup[raw] = {
            'is_discrete': is_discrete,
            'gpu_vendor': vendor
        }

    # Map array records back to main execution dataframe
    lookup_df = pd.DataFrame.from_dict(topology_lookup, orient='index')

    df['is_discrete'] = df[target_col].map(lookup_df['is_discrete']).astype(np.int8)
    df['gpu_vendor'] = df[target_col].map(lookup_df['gpu_vendor']).astype('category')

    return df

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------

# Apply to df_working using 'gpu' as the target column
df_working = extract_gpu_topology_matrix(df_working, 'gpu')

# Print target results verification
print("--- GPU Topology Feature Extraction ---")
print("\n1. Discrete (1) vs Integrated (0) GPU Distribution:")
print(df_working['is_discrete'].value_counts())

print("\n2. GPU Vendor Distribution:")
print(df_working['gpu_vendor'].value_counts())

# ==========================================
# NEXT CELL
# ==========================================

import pandas as pd
import re

def compile_heterogeneous_cpu_pipeline(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    unique_keys = df[target_col].dropna().unique()

    # Pre-compile isolated structural parsing primitives
    # Qualcomm ARM Parsers
    qualcomm_pattern = re.compile(r'snapdragon\s*x\s*(elite|plus)', re.IGNORECASE)
    qualcomm_sku = re.compile(r'(x1e\d{5}|x1p\d{5})', re.IGNORECASE)

    # AMD Ryzen Parsers
    amd_ryzen_ai = re.compile(r'ryzen\s*ai\s*([579])\s*(hx)?\s*(\d{3})', re.IGNORECASE)
    amd_ryzen_std = re.compile(r'ryzen\s*([3579])\s*(\d{4})\s*([a-z]{1,2})?', re.IGNORECASE)
    amd_legacy_apu = re.compile(r'\b(a4|a6|a8|a9|a10|a12|athlon|gold|silver)\s*[-_]?\s*(\d{4})?([a-z])?', re.IGNORECASE)

    # Intel Modern Parsers
    intel_ultra = re.compile(r'(core\s*ultra|ultra)\s*([579])\s*(?:series\s*\d+)?\s*(?:modifier)?\s*(\d{3,5})?([a-z]{1,2})?', re.IGNORECASE)
    intel_core = re.compile(r'(?:core\s*)?i([3579])[-_]?\s*(\d{2,5})([a-z]{1,2})?', re.IGNORECASE)

    composite_lookup = {}

    for raw in unique_keys:
        clean = str(raw).lower().strip()

        # Branch 1: Qualcomm Snapdragon Architecture Execution Loop
        if 'snapdragon' in clean or 'qualcomm' in clean:
            tier_match = qualcomm_pattern.search(clean)
            sku_match = qualcomm_sku.search(clean)
            tier = tier_match.group(1).upper() if tier_match else "X"
            sku = sku_match.group(1).upper() if sku_match else "GENERIC"
            composite_lookup[raw] = f"QUALCOMM_SNAPDRAGON_{tier}_{sku}"
            continue

        # Branch 2: AMD Silicon Architecture Execution Loop
        if 'amd' in clean or 'ryzen' in clean or 'r3' in clean or 'r5' in clean or 'r7' in clean or 'r9' in clean:
            ai_match = amd_ryzen_ai.search(clean)
            std_match = amd_ryzen_std.search(clean)
            legacy_match = amd_legacy_apu.search(clean)

            if ai_match:
                tier = ai_match.group(1)
                suffix = ai_match.group(2).upper() if ai_match.group(2) else "STD"
                sku = ai_match.group(3)
                composite_lookup[raw] = f"AMD_RYZEN_AI{tier}_SKU{sku}_{suffix}"
            elif std_match:
                tier = std_match.group(1)
                sku = std_match.group(2)
                suffix = std_match.group(3).upper() if std_match.group(3) else "STD"
                composite_lookup[raw] = f"AMD_RYZEN_{tier}_SKU{sku}_{suffix}"
            elif legacy_match:
                family = legacy_match.group(1).upper()
                sku = legacy_match.group(2) if legacy_match.group(2) else "GENERIC"
                suffix = legacy_match.group(3).upper() if legacy_match.group(3) else "STD"
                composite_lookup[raw] = f"AMD_LEGACY_{family}_SKU{sku}_{suffix}"
            else:
                composite_lookup[raw] = "AMD_RYZEN_GENERIC_UNKNOWN"
            continue

        # Branch 3: Intel Core / Ultra Architecture Execution Loop
        ultra_match = intel_ultra.search(clean)
        core_match = intel_core.search(clean)

        if ultra_match:
            tier = ultra_match.group(2)
            sku = ultra_match.group(3) if ultra_match.group(3) else "GENERIC"
            suffix = ultra_match.group(4).upper() if ultra_match.group(4) else "STD"
            composite_lookup[raw] = f"INTEL_ULTRA_{tier}_SKU{sku}_{suffix}"
        elif core_match:
            tier = core_match.group(1)
            sku = core_match.group(2)
            suffix = core_match.group(3).upper() if core_match.group(3) else "STD"
            composite_lookup[raw] = f"INTEL_CORE_I{tier}_SKU{sku}_{suffix}"
        else:
            if 'celeron' in clean: composite_lookup[raw] = "INTEL_CELERON_STD"
            elif 'pentium' in clean: composite_lookup[raw] = "INTEL_PENTIUM_STD"
            else: composite_lookup[raw] = "UNCLASSIFIED_HARDWARE_CORE"

    # Step 2: Inject structural dictionary lookup over original string column vectors
    df['cpu_composite_clean'] = df[target_col].map(composite_lookup).astype('category')
    return df

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------

# Apply to df_working using 'cpu' as target column
df_working = compile_heterogeneous_cpu_pipeline(df_working, 'cpu')

# Verify results
print("--- Heterogeneous CPU Feature Summary ---")
print(f"Total Unique CPU Composite Classes: {df_working['cpu_composite_clean'].nunique()}\n")
print("Top 25 Most Common CPU configurations:")
print(df_working['cpu_composite_clean'].value_counts().head(25))

# ==========================================
# NEXT CELL
# ==========================================

initial_rows = len(df_working)

# V3 REVERT [H3]: DROP unclassified CPU rows.
# Rationale: Retaining them as cpu_tier=1 made 59% of the dataset indistinguishable from Celeron.
# The model cannot learn meaningful CPU pricing signal when the majority bucket is noise.
# Dropping them reduces dataset size but concentrates signal-to-noise ratio.
vague_cpu_categories = ['UNCLASSIFIED_HARDWARE_CORE', 'AMD_RYZEN_GENERIC_UNKNOWN']
vague_count = df_working['cpu_composite_clean'].isin(vague_cpu_categories).sum()

df_working = df_working[~df_working['cpu_composite_clean'].isin(vague_cpu_categories)].copy()
df_working['cpu_composite_clean'] = df_working['cpu_composite_clean'].cat.remove_unused_categories()

dropped_vague = initial_rows - len(df_working)
print(f"[V3] Dropped {dropped_vague} rows with unclassified CPU labels (UNCLASSIFIED_HARDWARE_CORE, AMD_RYZEN_GENERIC_UNKNOWN).")
print(f"Remaining records: {len(df_working)}")
print("\nCPU Category Distribution:")
print(df_working['cpu_composite_clean'].value_counts())

# ==========================================
# NEXT CELL
# ==========================================

import pandas as pd
import re

def compile_brand_normalization_pipeline(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # 1. Capture the unique raw fields to limit execution cost to O(U)
    unique_brands = df[target_col].dropna().unique()

    # 2. Compile regular expressions mapping variations to canonical parent targets
    rules = [
        # Composite Multi-Vendor Hardware Bundles
        (r'\b(dell/lenovo|lenovo/dell)\b', 'DELL_LENOVO_BUNDLE'),
        (r'\b(dell/gigabyte|gigabyte/dell)\b', 'DELL_GIGABYTE_BUNDLE'),
        (r'\b(msi/asrock|msi/lg|intel/zotac|ibm\s*/\s*cisco|zotac/asus)\b', 'COMPOSITE_HARDWARE_BUNDLE'),
        (r'^(custom|assembled|pc setup)\b', 'ASSEMBLED_CUSTOM'),

        # Invariant Brand Hierarchies and Typo Matrices
        (r'\b(lenovo|lenevo|lenova|lanovo|lanvo|lenolo|thinkpad)\b', 'LENOVO'),
        (r'\b(hp|hewlett|victus|compaq|hp\s*-\s*compaq)\b', 'HP'),
        (r'\b(dell|pnb\s*3530)\b', 'DELL'),
        (r'\b(asus|assus|aus|rog|strix)\b', 'ASUS'),
        (r'\b(acer|accer|predator)\b', 'ACER'),
        (r'\b(apple|macbook|imac)\b', 'APPLE'),
        (r'\b(msi)\b', 'MSI'),
        (r'\b(samsung|samsang)\b', 'SAMSUNG'),
        (r'\b(sony|vaio)\b', 'SONY'),
        (r'\b(toshiba)\b', 'TOSHIBA'),
        (r'\b(fujitsu)\b', 'FUJITSU'),
        (r'\b(lg)\b', 'LG'),
        (r'\b(microsoft|surface)\b', 'MICROSOFT'),
        (r'\b(realme|real\s*me|redmi|xiaomi|xiomi|mi|honor)\b', 'XIAOMI_REALME_HONOR'),
        (r'\b(huawei|huewei)\b', 'HUAWEI'),
        (r'\b(infinix|zerobook)\b', 'INFINIX'),
        (r'\b(avita)\b', 'AVITA'),
        (r'\b(chuwi|chuwai)\b', 'CHUWI'),
        (r'\b(primebook)\b', 'PRIMEBOOK'),
        (r'\b(motorola)\b', 'MOTOROLA'),

        # Desktop / Component / Accessory Vendors Mapping
        (r'\b(zebronics|zebronic|zeb|zobronics|zebster|zebion|zebonomic)\b', 'ZEBRONICS'),
        (r'\b(gigabyte|gigabite|gygabyte)\b', 'GIGABYTE'),
        (r'\b(asrock|asrok)\b', 'ASROCK'),
        (r'\b(biostar|bio\s*star)\b', 'BIOSTAR'),
        (r'\b(galax|galaxar)\b', 'GALAX'),
        (r'\b(zotac)\b', 'ZOTAC'),
        (r'\b(frontech|forntech|frobtech|fontech|foxin)\b', 'FRONTECH_FOXIN'),
        (r'\b(ant\s*esports|ant\s*sports|antec)\b', 'ANT_ESPORTS'),
        (r'\b(iball|intel|amd|nvidia|wipro|hcl|netgear|cisco|hikvision|nokia|panasonic|benq)\b', 'OTHER_TECH_BRAND')
    ]

    compiled_rules = [(re.compile(p, re.IGNORECASE), repl) for p, repl in rules]

    brand_mapping = {}
    for raw_entry in unique_brands:
        clean_str = str(raw_entry).strip().lower()
        matched = False

        for regex, replacement in compiled_rules:
            if regex.search(clean_str):
                brand_mapping[raw_entry] = replacement
                matched = True
                break

        if not matched:
            brand_mapping[raw_entry] = 'UNKNOWN_OEM'

    # 3. Direct O(1) hash map projection back across the primary vector series
    df['clean_brand'] = df[target_col].map(brand_mapping).astype('category')
    return df

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------

# Apply to df_working using 'brand' as target column
df_working = compile_brand_normalization_pipeline(df_working, 'brand')

# Verify results
print("--- Brand Normalization Summary ---")
print(f"Total Unique Brands Extracted: {df_working['clean_brand'].nunique()}\n")
print("Brand Distribution (Categorical):")
print(df_working['clean_brand'].value_counts())

# ==========================================
# NEXT CELL
# ==========================================

initial_rows = len(df_working)

# V2 FIX [H4]: RETAIN unclassified model_series rows instead of dropping.
# Rationale: 'Other / Unclassified Hardware' are real laptops where regex taxonomy missed.
# Dropping them biases the dataset toward recognizable brands/series.
# 'Assembled / Custom Desktop Chassis' are still dropped (genuinely not laptops).
series_to_drop = ['Assembled / Custom Desktop Chassis']

df_working = df_working[~df_working['clean_model_series'].isin(series_to_drop)].copy()
df_working['clean_model_series'] = df_working['clean_model_series'].cat.remove_unused_categories()

dropped_count = initial_rows - len(df_working)
print(f"Successfully dropped {dropped_count} records (desktop chassis only).")
print(f"[V2] Retained 'Other / Unclassified Hardware' rows (previously dropped).")
print(f"Remaining clean records: {len(df_working)}\n")

print("Verify clean_model_series categories:")
print(df_working['clean_model_series'].value_counts())


# ==========================================
# NEXT CELL
# ==========================================

import pandas as pd

if 'display_size' in df_working.columns:
    # 1. Get unique values, dropping NaNs, and sort them if possible
    unique_displays = df_working['display_size'].dropna().unique()
    try:
        unique_displays = sorted(list(unique_displays))
    except TypeError:
        unique_displays = list(unique_displays)

    # Export unique values to a CSV
    unique_display_df = pd.DataFrame({'Unique_Display_Size': unique_displays})
#     unique_display_df.to_csv('unique_display_sizes.csv', index=False)

    # 2. Get the frequency count of each display size
    counts_display_df = df_working['display_size'].value_counts().reset_index()
    counts_display_df.columns = ['Display_Size', 'Frequency']

    # Export counts to a CSV
#     counts_display_df.to_csv('counts_display_sizes.csv', index=False)

    print(f"Successfully exported {len(unique_displays)} unique display sizes to 'unique_display_sizes.csv'")
    print(f"Successfully exported frequency counts to 'counts_display_sizes.csv'")
else:
    print("Column 'display_size' not found in df_working.")

# ==========================================
# NEXT CELL
# ==========================================

import pandas as pd
import numpy as np
import re

def compile_numeric_display_pipeline(df: pd.DataFrame, target_col: str):
    # Step 1: Compute solely on unique keys to optimize CPU execution loop bounds
    unique_keys = df[target_col].dropna().unique()

    # Pre-compile structural lookaheads
    # Group 1 extracts the main float value. Truncates text scale extensions.
    num_pattern = re.compile(r'(\d{1,2}(?:\.\d{1,2})?)', re.IGNORECASE)

    numeric_lookup = {}

    for raw in unique_keys:
        clean = str(raw).lower().strip()

        # Eliminate resolution arrays and non-display tags greedily
        if '*' in clean or 'fhd' == clean or 'full hd' == clean or 'unknown' in clean or 'hd' == clean:
            numeric_lookup[raw] = np.nan
            continue

        match = num_pattern.search(clean)
        if match:
            try:
                value = float(match.group(1))

                # Check for resolution values that escaped the basic filter pass
                if value > 30.0:
                    # Explicit context: check for centimeter measurement notation
                    if 'cm' in clean:
                        value = round(value / 2.54, 1) # Metric space projection to inches
                    else:
                        value = np.nan # Resolution noise (e.g., 1080) dropped to NaN

                numeric_lookup[raw] = value
            except ValueError:
                numeric_lookup[raw] = np.nan
        else:
            numeric_lookup[raw] = np.nan

    # Step 2: Push mapped conversions back to a low-overhead continuous float column
    df['clean_display_inches'] = df[target_col].map(numeric_lookup).astype('float32')

    # Step 3: Compute exact telemetry sparsity counts for your imputation analysis
    null_count = df['clean_display_inches'].isna().sum()
    valid_count = df['clean_display_inches'].notna().sum()
    null_percentage = (null_count / len(df)) * 100

    print("=== TELEMETRY SPARSITY MATRIX ===")
    print(f"Total Operational Records : {len(df)}")
    print(f"Valid Floating Primitives : {valid_count}")
    print(f"NaN / Unknown Entries      : {null_count}")
    print(f"Sparsity Percentage       : {null_percentage:.2f}%")
    print("=================================")

    return df

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------

# Apply to df_working using 'display_size' as target column
df_working = compile_numeric_display_pipeline(df_working, 'display_size')

# Optional: View a sample of the mapping results
print("\nSample display size transformations:")
print(df_working[[ 'display_size', 'clean_display_inches']].drop_duplicates().head(15))

# ==========================================
# NEXT CELL
# ==========================================

#  Inspect the 'storage_tech' column spread and unique values
# (We search for 'storage_tech' since it maps to the SSD/HDD tech category)
target_col = 'storage_tech'

if target_col in df_working.columns:
    print(f"=== SPREAD & UNIQUES OF '{target_col}' ===")
    print("1. Raw Unique Values:")
    print(df_working[target_col].unique())

    print("\n2. Frequency Distribution (including NaNs):")
    print(df_working[target_col].value_counts(dropna=False))
else:
    # Fallback search if the column name changed in your local notebook
    similar_cols = [c for c in df_working.columns if any(x in c.lower() for x in ['tech', 'storage', 'ssd'])]
    print(f"Error: '{target_col}' column not found in df_working.")
    print(f"Did you mean one of these columns: {similar_cols}?")

# ==========================================
# NEXT CELL
# ==========================================

initial_rows = len(df_working)

# V3 REVERT [H1]: Filter to SSD-only. Business constraint: company only acquires SSD laptops.
# is_ssd becomes constant (always 1) and is evicted from all feature lists.
ssd_mask = df_working['storage_tech'].astype(str).str.lower() == 'ssd'
non_ssd_count = (~ssd_mask).sum()
df_working = df_working[ssd_mask].copy()
df_working = df_working.drop(columns=['storage_tech'], errors='ignore')

print(f"[V3] Dropped {non_ssd_count} non-SSD records (HDD/unknown/SAS). SSD-only enforced.")
print(f"Remaining records: {len(df_working)}")

# ==========================================
# NEXT CELL
# ==========================================

import pandas as pd
import numpy as np
import re

def resolve_complete_vendor_gpu(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Base VRAM boundary isolation
    vram_pattern = re.compile(r'(\d+)\s*(?:gb|mb|vram|b\b)', re.IGNORECASE)

    # 2. Comprehensive Multi-Vendor Architecture Patterns
    model_patterns = [
        # NVIDIA Discrete Tiers
        (re.compile(r'\b(rtx\s*50\d{2}|rtx\s*40\d{2}|rtx\s*30\d{2}|rtx\s*20\d{2})\b', re.IGNORECASE), lambda m: m.group(1).upper().replace(" ", "")),
        (re.compile(r'\b(gtx\s*1660|gtx\s*1650|gtx\s*1050|gtx\s*1060)\b', re.IGNORECASE), lambda m: m.group(1).upper().replace(" ", "")),
        (re.compile(r'\b(2050|2060|3050|3060|4050|4060|4070|4080|4090)\b', re.IGNORECASE), lambda m: f"RTX{m.group(1)}"),
        (re.compile(r'\b(1050|1060|1650|1660)\b', re.IGNORECASE), lambda m: f"GTX{m.group(1)}"),
        (re.compile(r'\b(quadro|p\d{4}|t\d{3,4}|m\d{3,4}m)\b', re.IGNORECASE), lambda m: "NVIDIA_QUADRO_PRO"),

        # AMD Discrete Tiers (RDNA / Polaris)
        (re.compile(r'\brx\s*([45679]\d{2,3}[a-z]{0,2})\b', re.IGNORECASE), lambda m: f"AMD_RX_{m.group(1).upper()}"),
        (re.compile(r'\b(pro\s*555|pro\s*560x?)\b', re.IGNORECASE), lambda m: f"AMD_RADEON_{m.group(1).upper().replace(' ', '')}"),

        # Intel Discrete Tiers (Alchemist)
        (re.compile(r'\barc\s*([a-z]?\d{3}m?|140v)?\b', re.IGNORECASE), lambda m: f"INTEL_ARC_{m.group(1).upper()}" if m.group(1) else "INTEL_ARC_GENERIC"),

        # High-Performance Integrated Tiers
        (re.compile(r'\b(680m|780m|890m)\b', re.IGNORECASE), lambda m: f"AMD_RDNA_{m.group(1).upper()}"),
        (re.compile(r'\b(iris\s*xe|iris\s*plus|iris)\b', re.IGNORECASE), lambda m: "INTEL_IRIS"),
        (re.compile(r'\b(uhd\s?graphics|uhd|hd\s?graphics|620|630)\b', re.IGNORECASE), lambda m: "INTEL_UHD"),

        # Apple Silicon Cores
        (re.compile(r'\b(\d+)\s*core\s*gpu\b', re.IGNORECASE), lambda m: f"APPLE_{m.group(1)}CORE_GPU")
    ]

    def extract_base(raw):
        if pd.isna(raw): return "UNKNOWN_SYSRAM"
        clean = str(raw).lower()

        vram_match = vram_pattern.findall(clean)
        vram = f"{int(vram_match[0])}GB" if vram_match and int(vram_match[0]) <= 24 else "SYSRAM"

        scrubbed = vram_pattern.sub('', clean).strip()
        model = "GENERIC_IGPU"

        for pat, fmt in model_patterns:
            match = pat.search(scrubbed)
            if match:
                model = fmt(match)
                break

        # Vendor-specific generic fallback
        if model == "GENERIC_IGPU":
            if 'nvidia' in scrubbed or 'geforce' in scrubbed or 'mx' in scrubbed: model = "NVIDIA_LEGACY"
            elif 'radeon' in scrubbed or 'amd' in scrubbed or 'vega' in scrubbed: model = "AMD_LEGACY_IGPU"
            elif 'intel' in scrubbed: model = "INTEL_LEGACY_IGPU"

        return f"{model}_{vram}"

    # Vectorized: compute extract_base on unique keys only, then project via .map()
    unique_gpus = df['gpu'].dropna().unique()
    gpu_base_lookup = {raw: extract_base(raw) for raw in unique_gpus}
    df['gpu_base'] = df['gpu'].map(gpu_base_lookup).fillna("UNKNOWN_SYSRAM")

    # 3. Vectorized Topological Inference
    is_generic_gpu = df['gpu_base'].str.contains('GENERIC|LEGACY|UNKNOWN', na=True)
    is_gaming_chassis = df['model_series'].astype(str).str.contains('legion|rog|tuf|nitro|predator|alienware|omen|victus|loq|katana|bravo|cyborg|sword', case=False, na=False)
    is_intel_cpu = df['cpu'].astype(str).str.contains('intel|core|i3|i5|i7|i9', case=False, na=False)
    is_amd_cpu = df['cpu'].astype(str).str.contains('amd|ryzen|athlon|a6|a8|a10', case=False, na=False)

    conditions = [
        (is_generic_gpu & is_gaming_chassis),
        (is_generic_gpu & is_intel_cpu),
        (is_generic_gpu & is_amd_cpu)
    ]

    choices = [
        "INFERRED_ENTRY_DGPU",
        "INTEL_INFERRED_IGPU_SYSRAM",
        "AMD_INFERRED_IGPU_SYSRAM"
    ]

    df['clean_gpu_composite'] = pd.Categorical(np.select(conditions, choices, default=df['gpu_base']))
    df = df.drop(columns=['gpu_base'])
    return df

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------

# Apply the updated pipeline to df_working
df_working = resolve_complete_vendor_gpu(df_working)

# V3 FIX [C11]: Backfill gpu_vendor from clean_gpu_composite prefix.
# extract_gpu_topology_matrix (line 596) set gpu_vendor=UNKNOWN for 72% of rows.
# resolve_complete_vendor_gpu (line 1029) then inferred the vendor from CPU into clean_gpu_composite.
# But gpu_vendor was never updated. This creates an inconsistency where gpu_vendor=UNKNOWN
# but clean_gpu_composite=INTEL_INFERRED_IGPU_SYSRAM. The target encoding on gpu_vendor
# sees 17K UNKNOWN rows as one homogeneous bucket, destroying vendor-specific pricing signal.
def _backfill_gpu_vendor(composite_val):
    s = str(composite_val).upper()
    if s.startswith('INTEL'): return 'INTEL'
    if s.startswith('AMD'): return 'AMD'
    if s.startswith('NVIDIA') or s.startswith('RTX') or s.startswith('GTX'): return 'NVIDIA'
    if s.startswith('APPLE'): return 'APPLE'
    return 'UNKNOWN'

old_unknown = (df_working['gpu_vendor'] == 'UNKNOWN').sum()
df_working['gpu_vendor'] = df_working['clean_gpu_composite'].map(_backfill_gpu_vendor).astype('category')
new_unknown = (df_working['gpu_vendor'] == 'UNKNOWN').sum()
print(f"\n[V3 FIX C11] gpu_vendor backfilled from composite prefix.")
print(f"  UNKNOWN reduced: {old_unknown} -> {new_unknown}")
print(f"  Vendor distribution after backfill:")
print(df_working['gpu_vendor'].value_counts())

# Verify results
print("\n--- Complete Vendor GPU Feature Extraction ---")
print(f"Total Unique GPU Categories: {df_working['clean_gpu_composite'].nunique()}\n")
print("Top 25 GPU Configurations (with precise vendor parsing and topological fallbacks):")
print(df_working['clean_gpu_composite'].value_counts().head(25))

# ==========================================
# NEXT CELL
# ==========================================

import pandas as pd

if 'clean_gpu_composite' in df_working.columns:
    # 1. Get the unique values from the column (ignoring any nulls)
    unique_gpu_vram = df_working['clean_gpu_composite'].dropna().unique()

    # 2. Sort them alphabetically for clean reading
    try:
        unique_gpu_vram = sorted(list(unique_gpu_vram))
    except TypeError:
        unique_gpu_vram = list(unique_gpu_vram)

    # 3. Convert to a DataFrame and export to CSV
    unique_gpu_df = pd.DataFrame({'Unique_GPU_Model_VRAM': unique_gpu_vram})
    output_filename = 'unique_gpu_model_vram.csv'
#     unique_gpu_df.to_csv(output_filename, index=False)

    print(f"Successfully exported {len(unique_gpu_vram)} unique composite GPU configurations to '{output_filename}'!")
else:
    print("Column 'gpu_model_vram' not found in df_working.")

# ==========================================
# NEXT CELL
# ==========================================

target_col = 'clean_gpu_composite'

if target_col in df_working.columns:
    # 1. Get unique values, dropping nulls, and sort them if possible
    unique_vals = df_working[target_col].dropna().unique()
    try:
        unique_vals = sorted(list(unique_vals))
    except TypeError:
        unique_vals = list(unique_vals)

    print(f"==================================================")
    print(f" COLUMN: {target_col.upper()}")
    print(f" TOTAL UNIQUE CATEGORIES: {len(unique_vals)}")
    print(f"==================================================")

    print("\n--- ALL UNIQUE VALUE CODES ---")
    print(unique_vals)

    print("\n--- FULL VALUE DISTRIBUTION (COUNTS) ---")
    print(df_working[target_col].value_counts())
else:
    print(f"Error: Column '{target_col}' not found in df_working.")

# ==========================================
# NEXT CELL
# ==========================================

initial_rows = len(df_working)

# V2 FIX [H2]: RETAIN GENERIC_IGPU_SYSRAM instead of dropping.
# Rationale: Dropping unclassified integrated graphics destroys budget-tier signal.
# The 'is_discrete=0' flag correctly handles their physical topology.
print(f"[V2] Retaining {len(df_working[df_working['clean_gpu_composite'] == 'GENERIC_IGPU_SYSRAM'])} rows with GENERIC_IGPU_SYSRAM (previously dropped).")
print(f"Total records in df_working: {len(df_working)}")

# ==========================================
# NEXT CELL
# ==========================================

# 1. Print all column names currently in df_working
print("=== CURRENT COLUMNS IN df_working ===")
print(df_working.columns.tolist())
print(f"Shape: {df_working.shape}")
print("-" * 60)


# ==========================================
# NEXT CELL
# ==========================================

# 1. Vectorized type coercion to establish structural float uniformity
clean_display_numeric = pd.to_numeric(df_working['clean_display_inches'], errors='coerce')

# 2. Compute outlier mask; IEEE 754 NaN values naturally evaluate to False
unreasonable_mask = (clean_display_numeric < 10.0) | (clean_display_numeric > 18.5)
unreasonable_count = unreasonable_mask.sum()

print(f"Number of records outside 10.0 - 18.5 inches: {unreasonable_count}")

if unreasonable_count > 0:
    print("\nSample of records to be dropped:")

    # 3. Dynamic set intersection to protect against downstream KeyErrors
    potential_cols = ['brand', 'clean_brand', 'clean_display_inches', 'description']
    cols_to_show = [col for col in potential_cols if col in df_working.columns]

    if cols_to_show:
        print(df_working[unreasonable_mask][cols_to_show].head(10))

    # 4. Filter and enforce a contiguous memory block allocation
    df_working = df_working[~unreasonable_mask].copy()
    print(f"\nDropped {unreasonable_count} records.")
    print(f"New shape of df_working: {df_working.shape}")
else:
    print("No records found outside the range.")

# ==========================================
# NEXT CELL
# ==========================================

import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from sklearn.base import BaseEstimator, TransformerMixin

class XGBoostDisplayImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_imputations: int = 7):
        self.n_imputations = n_imputations
        self.target = 'clean_display_inches'
        self.models = []
        self.impute_cols = None
        self.categories_ = {}
        self.categorical_features = ['clean_brand', 'clean_model_series', 'clean_gpu_composite', 'cpu_composite_clean']

    def fit(self, X, y=None):
        df = X.copy()
        
        # GLOBAL QUARANTINE: Initialize invariant boolean index mask if missing
        if 'is_global_train' not in df.columns:
            np.random.seed(42)
            df['is_global_train'] = np.random.rand(len(df)) < 0.75
            
        missing_mask = df[self.target].isna()
        if missing_mask.sum() == 0: return self
        
        train_df = df[~missing_mask].copy()
        
        # Explicit partition: restrict completely to training memory
        train_df = train_df[train_df['is_global_train'] == True]
        
        numeric_features = ['ram_size_gb', 'storage_size_gb', 'is_discrete']
        binary_features = ['fingerprint', 'touch']

        for col in self.categorical_features:
            if col in train_df.columns:
                raw_str = train_df[col].astype(str).str.lower()
                self.categories_[col] = raw_str.unique()
                train_df[col] = raw_str.astype(pd.CategoricalDtype(categories=self.categories_[col]))
        for col in binary_features:
            if col in train_df.columns:
                train_df[col] = train_df[col].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(-1).astype('int8')

        train_df['ram_x_storage'] = train_df['ram_size_gb'] * train_df['storage_size_gb']
        self.impute_cols = numeric_features + self.categorical_features + binary_features + ['ram_x_storage']
        
        X_train = train_df[self.impute_cols]
        y_train = train_df[self.target]

        hyperparam_variants = [
            {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.08, 'subsample': 0.70},
            {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.10, 'subsample': 0.80},
            {'n_estimators': 250, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.75},
            {'n_estimators': 180, 'max_depth': 4, 'learning_rate': 0.12, 'subsample': 0.85},
            {'n_estimators': 220, 'max_depth': 8, 'learning_rate': 0.07, 'subsample': 0.65},
            {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.06, 'subsample': 0.90},
            {'n_estimators': 170, 'max_depth': 6, 'learning_rate': 0.15, 'subsample': 0.70},
        ]
        
        for i in range(self.n_imputations):
            rng = np.random.RandomState(42 + i * 13)
            bootstrap_idx = rng.choice(len(train_df), size=len(train_df), replace=True)
            X_sample = X_train.iloc[bootstrap_idx]
            y_sample = y_train.iloc[bootstrap_idx]

            params = hyperparam_variants[i % len(hyperparam_variants)]
            model = XGBRegressor(**params, tree_method='hist', enable_categorical=True, random_state=rng.randint(0, 100000), verbosity=0)
            model.fit(X_sample, y_sample)
            self.models.append(model)
        
        return self

    def transform(self, X):
        df = X.copy()
        missing_mask = df[self.target].isna()
        if missing_mask.sum() == 0 or len(self.models) == 0: return df
        
        predict_df = df[missing_mask].copy()
        
        binary_features = ['fingerprint', 'touch']
        for col in self.categorical_features:
            if col in predict_df.columns:
                raw_str = predict_df[col].astype(str).str.lower()
                raw_str = raw_str.where(raw_str.isin(self.categories_[col]), np.nan)
                predict_df[col] = raw_str.astype(pd.CategoricalDtype(categories=self.categories_[col]))
        for col in binary_features:
            if col in predict_df.columns:
                predict_df[col] = predict_df[col].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(-1).astype('int8')

        predict_df['ram_x_storage'] = predict_df['ram_size_gb'] * predict_df['storage_size_gb']
        X_predict = predict_df[self.impute_cols]
        
        imputed_results = np.zeros((missing_mask.sum(), self.n_imputations))
        for i, model in enumerate(self.models):
            imputed_results[:, i] = model.predict(X_predict)
            
        pooled_values = np.median(imputed_results, axis=1)
        # V2 FIX [M4]: Added 12.0 to snap grid (common Chromebook / Surface size)
        real_sizes = np.array([10.1, 11.6, 12.0, 12.3, 12.5, 13.3, 14.0, 14.1, 15.0, 15.6, 16.0, 16.1, 17.3, 18.0])
        snapped_values = real_sizes[np.argmin(np.abs(real_sizes[:, None] - pooled_values), axis=0)]
        
        df.loc[missing_mask, self.target] = snapped_values.astype('float32')
        return df

# Run it
# V2 FIX [C1]: Proper stratified split using sklearn on price quantiles instead of np.random.rand()
# Stratifying on price quantiles ensures train and holdout share the identical target geometry.
from sklearn.model_selection import train_test_split

np.random.seed(42)
# Create 10 price bins for stratification
price_bins = pd.qcut(df_working['price'], q=10, labels=False, duplicates='drop')
train_idx, _ = train_test_split(
    df_working.index,
    test_size=0.25,
    random_state=42,
    stratify=price_bins
)

df_working['is_global_train'] = False
df_working.loc[train_idx, 'is_global_train'] = True

print("\n[V2] Stratified Train/Test Mask Generated.")
print(f"Train rows: {df_working['is_global_train'].sum()} | Test rows: {(~df_working['is_global_train']).sum()}")

xgb_imputer = XGBoostDisplayImputer(n_imputations=7)
df_working = xgb_imputer.fit_transform(df_working)

# ==========================================
# NEXT CELL
# ==========================================

print("=== Descriptive Statistics for clean_display_inches ===")
print(df_working['clean_display_inches'].describe())

print("\n=== Unique Snapped Display Sizes (Sorted) & Counts ===")
print(df_working['clean_display_inches'].value_counts().sort_index())

# ==========================================
# NEXT CELL
# ==========================================

columns_to_keep = [
    "price", "state", "city", "created_at", "images_count",
    "storage_size_gb", "ram_size_gb", "fingerprint", "touch",
    "warranty", "urgent_sell", "clean_model_series", "is_discrete",
    "gpu_vendor", "cpu_composite_clean", "clean_brand",
    "clean_display_inches", "clean_gpu_composite", "description"
]
columns_to_keep.append('is_global_train')
df_final = df_working[columns_to_keep].copy()
# Verify shape and columns
print("=== Dataset Dimensions ===")
print(f"Number of Records (Rows): {df_final.shape[0]} (Expected: 5110)")
print(f"Number of Columns      : {df_final.shape[1]} (Expected: 18)")

print("\n=== Columns Match Verification ===")
expected_cols = [
    'price', 'state', 'city', 'created_at', 'images_count',
    'storage_size_gb', 'ram_size_gb', 'fingerprint', 'touch',
    'warranty', 'urgent_sell', 'clean_model_series', 'is_discrete',
    'gpu_vendor', 'cpu_composite_clean', 'clean_brand',
    'clean_display_inches', 'clean_gpu_composite', 'description',
    'is_global_train'
]
actual_cols = list(df_final.columns)
missing = [c for c in expected_cols if c not in actual_cols]
extra = [c for c in actual_cols if c not in expected_cols]

print(f"Missing Columns: {missing}")
print(f"Extra Columns  : {extra}")
print(f"All matched?   : {missing == [] and extra == []}")

# ==========================================
# NEXT CELL
# ==========================================

import pandas as pd
import numpy as np

# Binary columns: 'yes' â†’ 1, everything else â†’ 0
binary_cols = ['fingerprint', 'touch', 'warranty', 'urgent_sell']
for col in binary_cols:
    col_str = df_final[col].fillna('no').astype(str).str.lower().str.strip()
    df_final[col] = np.where(col_str == 'yes', 1, 0)

# Numeric columns: force to float
numeric_cols = ['price', 'ram_size_gb', 'storage_size_gb', 'clean_display_inches']
for col in numeric_cols:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

# Quick sanity check
print(f"Shape: {df_final.shape}")
print(f"NaN counts after coercion:\n{df_final[numeric_cols].isnull().sum()}")
print(f"\nBinary distributions:")
for col in binary_cols:
    print(f"  {col}: {dict(df_final[col].value_counts())}")

# ==========================================
# NEXT CELL
# ==========================================

def map_cpu_tier(cpu_name):
    """Maps CPU composite name to an ordinal performance tier (1-10).

    Hierarchy:
      1  = Celeron / Athlon / A4
      2  = Pentium / Legacy A6-A9
      3  = Core i3 / Ryzen 3 / Legacy A10-A12
      5  = Core i5 / Ryzen 5
      6  = Intel Ultra 5
      7  = Core i7 / Ryzen 7
      8  = Intel Ultra 7 / Ryzen AI 7
      9  = Core i9 / Ryzen 9
      10 = Intel Ultra 9 / Ryzen AI 9
    """
    cpu = str(cpu_name).upper()

    # Budget Intel â€” PENTIUM_GOLD must precede generic PENTIUM (substring shadow)
    if 'CELERON' in cpu: return 1
    if 'PENTIUM_GOLD' in cpu: return 4
    if 'PENTIUM' in cpu: return 2

    # AMD Legacy (check before Ryzen to avoid false matches)
    if 'LEGACY' in cpu:
        if any(x in cpu for x in ['A4', 'ATHLON']): return 1
        if any(x in cpu for x in ['A6', 'A8', 'A9']): return 2
        if any(x in cpu for x in ['A10', 'A12']): return 3
        return 1

    # Intel Ultra (check before Core to avoid substring conflicts)
    if 'ULTRA_9' in cpu: return 10
    if 'ULTRA_7' in cpu: return 8
    if 'ULTRA_5' in cpu: return 6

    # AMD Ryzen AI (check before standard Ryzen)
    if 'RYZEN_AI9' in cpu: return 10
    if 'RYZEN_AI7' in cpu: return 8

    # Standard series
    if 'CORE_I9' in cpu or 'RYZEN_9' in cpu: return 9
    if 'CORE_I7' in cpu or 'RYZEN_7' in cpu: return 7
    if 'CORE_I5' in cpu or 'RYZEN_5' in cpu: return 5
    # V3 FIX: Added missing i3/Ryzen 3 tier (was falling through to budget fallback)
    if 'CORE_I3' in cpu or 'RYZEN_3' in cpu: return 3
    # V2 FIX [H6]: Ryzen AI 5 and Core Ultra 3 map to tier 4
    if 'RYZEN_AI5' in cpu or 'ULTRA_3' in cpu: return 4

    return 1  # fallback for anything unrecognized

df_final['cpu_tier'] = df_final['cpu_composite_clean'].apply(map_cpu_tier)

print("=== CPU Tier Distribution ===")
print(df_final['cpu_tier'].value_counts().sort_index())
print(f"\nTotal mapped: {len(df_final)}/{len(df_final)} rows")

# ==========================================
# NEXT CELL
# ==========================================

def map_gpu_tier(gpu_name):
    """Maps GPU composite name to an ordinal price-impact tier (1-9).

    Rule: maximal specificity first. Broad catch-all tokens ('QUADRO')
    occupy the lowest execution weight at the bottom of the chain.
    """
    gpu = str(gpu_name).upper()

    # V2 FIX [H7]: Intel ARC can be discrete. Only classify as tier 1 if it's integrated or generic.
    if any(x in gpu for x in ['IGPU', 'UHD', 'IRIS']):
        return 1
    if 'ARC' in gpu and 'GENERIC' in gpu:
        return 1
    
    # Intel ARC discrete (e.g., A370M, A730M)
    if 'ARC' in gpu and 'GENERIC' not in gpu:
        return 4

    # Tier 2: AMD integrated / inferred entry discrete
    if 'RDNA' in gpu:
        return 2
    if 'INFERRED_ENTRY' in gpu:
        return 2

    # === DISCRETE GPUs: most specific alphanumeric tokens first ===

    # Tier 9: Flagship
    if 'RTX4090' in gpu or 'RTX5090' in gpu:
        return 9

    # Tier 8: Ultra
    if any(x in gpu for x in ['RTX4070', 'RTX4080', 'RTX5070', 'RTX5080']):
        return 8

    # Tier 7: High + Workstation RTX (specific generation markers BEFORE broad QUADRO)
    if any(x in gpu for x in ['RTX3070', 'RTX3080', 'RTX4060', 'RTX5060',
                                'RTX2000', 'RTX3000', 'RTX5000']):
        return 7

    # Tier 6: Upper-mid
    if any(x in gpu for x in ['RTX2060', 'RTX2070', 'RTX3060', 'RTX4050', 'RTX5050',
                                'RX_6800', 'RX_7600']):
        return 6

    # Tier 5: Entry RTX (no broad tokens)
    if any(x in gpu for x in ['RTX2050', 'RTX3050']):
        return 5

    # Tier 4: GTX 16xx / AMD RX entry
    if any(x in gpu for x in ['GTX1650', 'GTX1660', 'RX_560', 'RX_5500', 'RX_6500', 'RX_600M']):
        return 4

    # Tier 3: GTX 10xx
    if any(x in gpu for x in ['GTX1050', 'GTX1060']):
        return 3

    # Tier 5 fallback: generic QUADRO without generation marker
    # Only reachable if no specific workstation token matched above
    if 'QUADRO' in gpu:
        return 5

    return 1

df_final['gpu_tier'] = df_final['clean_gpu_composite'].apply(map_gpu_tier)

print("=== GPU Tier Distribution ===")
print(df_final['gpu_tier'].value_counts().sort_index())

# ==========================================
# NEXT CELL
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# Only hardware-spec features + price (no behavioral features like images_count)
analysis_features = ['price', 'ram_size_gb', 'storage_size_gb',
                     'clean_display_inches', 'cpu_tier', 'gpu_tier', 'is_discrete']

corr_matrix = df_final[analysis_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
            fmt='.2f', square=True, linewidths=0.5,
            vmin=-1, vmax=1, ax=ax)
ax.set_title('Joint Feature Covariance: Hardware Specs vs Price',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Interpret correlations with price
print("\n=== Correlations with Price ===")
price_corr = corr_matrix['price'].drop('price').sort_values(ascending=False)
for feat, val in price_corr.items():
    direction = "positive" if val > 0 else "negative"
    strength = "STRONG" if abs(val) > 0.5 else "moderate" if abs(val) > 0.3 else "weak"
    print(f"  {feat:<25}: {val:+.3f} ({strength} {direction})")

# ==========================================
# NEXT CELL
# ==========================================

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# Expand matrix dimensions to 3x2 to handle 5 components
fig, axes = plt.subplots(3, 2, figsize=(14, 18))
axes_flat = axes.flat

scatter_pairs = [
    ('ram_size_gb', 'price', 'RAM (GB) vs Price'),
    ('storage_size_gb', 'price', 'Storage (GB) vs Price'),
    ('cpu_tier', 'price', 'CPU Tier vs Price'),
    ('clean_display_inches', 'price', 'Display Size vs Price'),
    ('gpu_tier', 'price', 'GPU Tier vs Price')
]

colors = df_final['is_discrete'].map({0: '#3498db', 1: '#e74c3c'})

# Streamlined execution via linear memory flattening
for idx, (x_col, y_col, title) in enumerate(scatter_pairs):
    ax = axes_flat[idx]
    ax.scatter(df_final[x_col], df_final[y_col], c=colors, alpha=0.3, s=10)
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel('Price (â‚¹)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Purge the unmapped 6th subplot from the rendering pipeline
axes_flat[-1].set_visible(False)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Integrated GPU'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Discrete GPU')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=11,
           bbox_to_anchor=(0.5, 1.01))

plt.suptitle('Market Equilibrium Cloud', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()


# ==========================================
# NEXT CELL
# ==========================================

from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import genpareto

# â”€â”€ STEP 1: Log-Space Baseline Estimator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
hardware_cols = ['ram_size_gb', 'storage_size_gb', 'clean_display_inches',
                 'cpu_tier', 'gpu_tier', 'is_discrete']

y_log = np.log(df_final['price'])

# LEAKAGE FIX: Fit baseline ONLY on training rows
# The baseline must not learn holdout price geometry for threshold derivation
evt_train_mask = df_final['is_global_train'] == True

baseline = GradientBoostingRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    loss='huber', random_state=42
)
baseline.fit(df_final.loc[evt_train_mask, hardware_cols], y_log[evt_train_mask])

# Predict on ALL rows (train + test) to compute residuals for outlier flagging
df_final['predicted_log_price'] = baseline.predict(df_final[hardware_cols])
df_final['expected_price'] = np.exp(df_final['predicted_log_price'])

r2 = baseline.score(df_final.loc[evt_train_mask, hardware_cols], y_log[evt_train_mask])
print(f"Baseline RÂ² (log-space, train-only fit): {r2:.3f}")

# â”€â”€ STEP 2: Log-Residual (symmetric, dimensionless) â”€â”€â”€â”€â”€
df_final['log_residual'] = np.log(df_final['price']) - df_final['predicted_log_price']
df_final['abs_log_residual'] = df_final['log_residual'].abs()

print(f"\n=== Log-Residual Distribution ===")
print(f"  Mean   : {df_final['log_residual'].mean():+.4f}")
print(f"  Median : {df_final['log_residual'].median():+.4f}")
print(f"  Std    : {df_final['log_residual'].std():.4f}")
print(f"  |Max|  : {df_final['abs_log_residual'].max():.4f}")

# â”€â”€ STEP 3: GPD on |log_residual| â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# V2 FIX [C2]: Compute threshold entirely on training set residuals to prevent leakage
train_values = df_final.loc[evt_train_mask, 'abs_log_residual'].values
all_values = df_final['abs_log_residual'].values

u = np.percentile(train_values, 90)
train_excess_mask = train_values > u
exceedances = train_values[train_excess_mask] - u

shape_xi, _, scale_sigma = genpareto.fit(exceedances, floc=0)

print(f"\n=== EVT: GPD Tail Fit on |log_residual| (Train Only) ===")
print(f"  Threshold u (90th pctl) : {u:.4f}")
print(f"  Exceedances             : {train_excess_mask.sum()}")
print(f"  GPD shape (xi)          : {shape_xi:.4f}", end="")
if shape_xi > 0:
    print("  -> Heavy tail")
elif shape_xi < 0:
    print("  -> Bounded tail")
else:
    print("  -> Exponential tail")
print(f"  GPD scale (sigma)       : {scale_sigma:.4f}")

# â”€â”€ STEP 4: Dynamic Threshold at p=0.01 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
p_above_u = train_excess_mask.mean()
target_p = 0.01
conditional_p = min(target_p / p_above_u, 0.999)

evt_excess = genpareto.ppf(1 - conditional_p, shape_xi, loc=0, scale=scale_sigma)
evt_threshold = u + evt_excess

n_outliers = (df_final['abs_log_residual'] > evt_threshold).sum()
df_final['is_outlier'] = (df_final['abs_log_residual'] > evt_threshold).astype(int)

print(f"\n  EVT threshold           : {evt_threshold:.4f}")
print(f"  Outliers detected       : {n_outliers} ({n_outliers/len(df_final)*100:.1f}%)")
print(f"\n  xi <= 0.15 check        : {'PASS' if shape_xi <= 0.15 else 'FAIL'}")

# ==========================================
# NEXT CELL
# ==========================================

outlier_mask = df_final['is_outlier'] == 1

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. |Log Residual| distribution + EVT threshold
ax1 = axes[0][0]
ax1.hist(df_final['abs_log_residual'], bins=60, color='#3498db', edgecolor='white', alpha=0.8)
ax1.axvline(evt_threshold, color='red', linestyle='--', linewidth=2,
            label=f'EVT threshold: {evt_threshold:.3f}')
ax1.set_xlabel('|Log Residual|', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Absolute Log-Residual Distribution', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Expected price vs signed log residual
ax2 = axes[0][1]
ax2.scatter(df_final.loc[~outlier_mask, 'expected_price'],
            df_final.loc[~outlier_mask, 'log_residual'],
            c='#3498db', alpha=0.3, s=10, label='Inlier')
ax2.scatter(df_final.loc[outlier_mask, 'expected_price'],
            df_final.loc[outlier_mask, 'log_residual'],
            c='red', alpha=0.7, s=25, label='Outlier', zorder=5)
ax2.axhline(evt_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax2.axhline(-evt_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Expected Price (Rs)', fontsize=11)
ax2.set_ylabel('Log Residual', fontsize=11)
ax2.set_title('Log Residual vs Expected Price', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. CPU Tier vs Price
ax3 = axes[1][0]
ax3.scatter(df_final.loc[~outlier_mask, 'cpu_tier'],
            df_final.loc[~outlier_mask, 'price'],
            c='#3498db', alpha=0.3, s=10, label='Inlier')
ax3.scatter(df_final.loc[outlier_mask, 'cpu_tier'],
            df_final.loc[outlier_mask, 'price'],
            c='red', alpha=0.7, s=25, label='Outlier', zorder=5)
ax3.set_xlabel('CPU Tier', fontsize=11)
ax3.set_ylabel('Price (Rs)', fontsize=11)
ax3.set_title('CPU Tier vs Price', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. GPU Tier vs Price
ax4 = axes[1][1]
ax4.scatter(df_final.loc[~outlier_mask, 'gpu_tier'],
            df_final.loc[~outlier_mask, 'price'],
            c='#3498db', alpha=0.3, s=10, label='Inlier')
ax4.scatter(df_final.loc[outlier_mask, 'gpu_tier'],
            df_final.loc[outlier_mask, 'price'],
            c='red', alpha=0.7, s=25, label='Outlier', zorder=5)
ax4.set_xlabel('GPU Tier', fontsize=11)
ax4.set_ylabel('Price (Rs)', fontsize=11)
ax4.set_title('GPU Tier vs Price', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.suptitle('Log-Space GPD Outlier Detection', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# â”€â”€ PREMIUM MISCLASSIFICATION MATRIX â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 65)
print("  PREMIUM CONFIG MISCLASSIFICATION MATRIX")
print("=" * 65)
premium = (df_final['cpu_tier'] >= 9) & (df_final['price'] >= 150000)
if premium.sum() > 0:
    n_in = (df_final.loc[premium, 'is_outlier'] == 0).sum()
    n_out = (df_final.loc[premium, 'is_outlier'] == 1).sum()
    status = "PASS" if n_out <= 2 else "FAIL"
    print(f"  Premium listings (CPU>=9, Price>=1.5L) : {premium.sum()}")
    print(f"  Classified INLIER                      : {n_in}")
    print(f"  Misclassified OUTLIER                  : {n_out}")
    print(f"  Constraint (Misclassified <= 2)         : {status}")
    if n_out > 0:
        print(f"\n  Misclassified units:")
        print(df_final.loc[premium & (df_final['is_outlier'] == 1),
                ['price', 'expected_price', 'log_residual',
                 'cpu_tier', 'gpu_tier', 'ram_size_gb',
                 'cpu_composite_clean', 'clean_gpu_composite']])
else:
    print("  No premium listings found.")

# â”€â”€ OUTLIER TABLE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n=== Flagged Outliers ({outlier_mask.sum()}) ===")
print(df_final.loc[outlier_mask,
        ['price', 'expected_price', 'log_residual', 'abs_log_residual',
         'ram_size_gb', 'cpu_tier', 'gpu_tier', 'clean_brand',
         'cpu_composite_clean', 'clean_gpu_composite']].sort_values('abs_log_residual', ascending=False))

# ==========================================
# NEXT CELL
# ==========================================

n_outliers = df_final['is_outlier'].sum()
print(f"Before removal: {len(df_final)} records")

df_final = df_final[df_final['is_outlier'] == 0].copy()

df_final.drop(columns=['predicted_log_price', 'expected_price', 'log_residual',
                        'abs_log_residual', 'is_outlier'], inplace=True)

print(f"After removal : {len(df_final)} records")
print(f"Removed       : {n_outliers} mispriced listings")

# ==========================================
# NEXT CELL
# ==========================================

# 1. Print the columns as a list
print("Columns in df_final:")
print(df_final.columns.tolist())

# 2. View DataFrame details, including non-null counts and data types for all columns
df_final.info()

# 3. View the first few rows of the filtered DataFrame
df_final.head()

# ==========================================
# NEXT CELL
# ==========================================

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def optimized_oof_median_encoding(df, categorical_cols, target_col='price', n_splits=5, m_smoothing=10):
    """
    OOF median encoding with strict train/test quarantine.
    KFold splits traverse ONLY training rows (is_global_train == True).
    Test rows receive stable global training-set medians per category.
    """
    target = df[target_col].to_numpy(dtype=np.float64)
    n_rows = len(df)

    # Isolate train/test masks from the global quarantine flag
    is_train = df['is_global_train'].to_numpy(dtype=bool)
    train_indices = np.where(is_train)[0]
    test_indices = np.where(~is_train)[0]

    output_matrix = np.empty((n_rows, len(categorical_cols)), dtype=np.float64)

    # V2 FIX [C6]: Factorize strings to zero-indexed integers ONCE, explicitly on TRAIN data only
    # Test categories unseen in train will receive -1 and be routed to global median
    factorized_cols = []
    num_classes = []
    train_df = df.iloc[train_indices]
    
    for col in categorical_cols:
        # Fit factorizer on train only
        train_cat = pd.Categorical(train_df[col])
        categories = train_cat.categories
        
        # Transform full dataset using train categories
        codes = pd.Categorical(df[col], categories=categories).codes
        factorized_cols.append(codes)
        num_classes.append(len(categories))

    # â”€â”€ PHASE A: OOF encoding for TRAINING rows only â”€â”€
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # KFold splits strictly over training indices
    for kf_train_pos, kf_val_pos in kf.split(train_indices):
        # Map positional indices back to global dataframe indices
        actual_train_idx = train_indices[kf_train_pos]
        actual_val_idx = train_indices[kf_val_pos]

        train_target = target[actual_train_idx]
        global_median = np.median(train_target)

        for c_idx, codes in enumerate(factorized_cols):
            train_codes = codes[actual_train_idx]
            val_codes = codes[actual_val_idx]
            n_cats = num_classes[c_idx]

            cat_medians = np.full(n_cats, global_median, dtype=np.float64)

            for cat in range(n_cats):
                mask = (train_codes == cat)
                count = np.sum(mask)
                if count > 0:
                    cat_target_slice = train_target[mask]
                    raw_median = np.median(cat_target_slice)
                    cat_medians[cat] = ((count * raw_median) + (m_smoothing * global_median)) / (count + m_smoothing)

            output_matrix[actual_val_idx, c_idx] = cat_medians[val_codes]

    # â”€â”€ PHASE B: Stable encoding for TEST rows â”€â”€
    # Test rows get the full training-set median per category (no fold noise)
    full_train_target = target[train_indices]
    full_global_median = np.median(full_train_target)

    for c_idx, codes in enumerate(factorized_cols):
        full_train_codes = codes[train_indices]
        test_codes = codes[test_indices]
        n_cats = num_classes[c_idx]

        cat_medians = np.full(n_cats, full_global_median, dtype=np.float64)
        for cat in range(n_cats):
            mask = (full_train_codes == cat)
            count = np.sum(mask)
            if count > 0:
                raw_median = np.median(full_train_target[mask])
                cat_medians[cat] = ((count * raw_median) + (m_smoothing * full_global_median)) / (count + m_smoothing)

        # Test codes of -1 (unseen categories) get global median explicitly
        test_encodings = np.full(len(test_codes), full_global_median, dtype=np.float64)
        valid_test_mask = test_codes >= 0
        test_encodings[valid_test_mask] = cat_medians[test_codes[valid_test_mask]]
        
        output_matrix[test_indices, c_idx] = test_encodings

    # Reassemble final DataFrame
    encoded_df = df.copy()
    for c_idx, col in enumerate(categorical_cols):
        encoded_df[f"{col}_target_enc"] = output_matrix[:, c_idx]

    return encoded_df


# ==========================================
# Execution on df_final
# ==========================================

columns_to_encode = [
    'state',
    'city',
    'clean_model_series',
    'gpu_vendor',
    'cpu_composite_clean',
    'clean_brand',
    'clean_gpu_composite'
]

# Execute the optimized array-level pipeline
df_final_encoded = optimized_oof_median_encoding(
    df=df_final,
    categorical_cols=columns_to_encode,
    target_col='price',
    n_splits=5,
    m_smoothing=10
)

# Phase 3: Telemetry & Verification (Leakage Isolation check)
print("=== ALGORITHMIC OOF VERIFICATION ===")
for col in columns_to_encode:
    encoded_col = f"{col}_target_enc"
    corr = df_final_encoded[encoded_col].corr(df_final_encoded['price'])
    print(f"{encoded_col} correlation with price: {corr:.4f}")

# ==========================================
# NEXT CELL
# ==========================================

import numpy as np
import pandas as pd

# V2 FIX [C3]: Replaced hardcoded decay normalization with listing_age_days feature
# Target is simply np.log1p(price)
def create_age_feature_and_target(df, date_col='created_at', target_col='price'):
    """
    Creates listing_age_days feature and log1p price target.
    """
    # Parse series to datetime objects
    date_series = pd.to_datetime(df[date_col], errors='coerce')
    
    # Terminal reference epoch
    valid_mask = date_series.notna()
    if not valid_mask.any():
        df['listing_age_days'] = 0.0
    else:
        # V2 FIX [C7]: Anchor epoch and NaN fill restricted to training rows to prevent leakage
        train_date_mask = valid_mask & (df['is_global_train'] == True)
        if train_date_mask.any():
            terminal_epoch = date_series[train_date_mask].max()
        else:
            terminal_epoch = date_series[valid_mask].max()
        # Compute days difference
        df['listing_age_days'] = (terminal_epoch - date_series).dt.total_seconds() / (24 * 3600)
        # Fill NaNs with median age (train-only)
        if train_date_mask.any():
            median_age = df.loc[train_date_mask, 'listing_age_days'].median()
        else:
            median_age = df.loc[valid_mask, 'listing_age_days'].median()
        df['listing_age_days'] = df['listing_age_days'].fillna(median_age).astype(np.float32)
        
    # Standard stationary target
    df[f'{target_col}_normalized'] = np.log1p(df[target_col].to_numpy(dtype=np.float64))
    
    return df

# ==========================================
# Execution on df_final_encoded
# ==========================================

# 1. Apply age feature and target log
df_final_engineered = create_age_feature_and_target(
    df=df_final_encoded,
    date_col='created_at',
    target_col='price'
)

# Phase 3: Telemetry & Verification

# Check 1: NaN Integrity Assertions
nan_count = df_final_engineered['price_normalized'].isna().sum()
print("=== NaN INTEGRITY ASSERTION ===")
print(f"Empty/Corrupted Final Target Values: {nan_count}")
assert nan_count == 0, "Structural failure in array bounds mapping. Found NaNs in final price target."
print("[PASS] Target vector maintains byte-level integrity.\n")

# Check 2: Immediate Feature Eviction (Rule 1)
# Purge temporal indices from the matrix entirely to prevent structural leakage
if 'created_at' in df_final_engineered.columns:
    df_final_engineered = df_final_engineered.drop(columns=['created_at'])
    print("=== IMMEDIATE FEATURE EVICTION ===")
    print("[PASS] 'created_at' index has been permanently purged from the feature tensor.\n")

# Check 3: Target Realism Audit
preview_targets = ['cpu_tier', 'gpu_tier', 'price', 'price_normalized']
print("=== TRUE STATIONARY TARGET VECTORS ===")
# You can swap .sample(5) for .loc[[1069]] if index 1069 exists in your index range
print(df_final_engineered[preview_targets].sample(5))

print("\n[NOTE]: 'cpu_tier' and 'gpu_tier' remain untouched as independent features in X.")
print("The ML model will now organically learn their nonlinear discount relationships against 'price_normalized'.")

# ==========================================
# NEXT CELL
# ==========================================

import numpy as np
import pandas as pd
import re

# ==========================================
# Phase 2: Production Hardened Macro Engine (v6)
# ==========================================

# V3 Carry-over: Pre-flight negation stripping (now includes / and +)
NEGATION_STRIP_PAT = re.compile(
    r"\b(zero|no|free\s+from|without)\s+[\w\s,\/\+]{1,60}?(?=\b|\.|\n)",
    re.IGNORECASE
)

# V4 Carry-over: Software entity masking
GAME_MASK = re.compile(r"\bextinction\b|\bred\s+dead\s*(redemption\s*\d*)?|\bdead\s+space\b|\bdead\s+island\b", re.IGNORECASE)

# V6 UPGRADE: Character classes expanded to [\\w\\s,\\/\\-]
# Successfully traps strings like "weak/dead battery" without breaking the sequence
BATTERY_MASK = re.compile(
    r"\bbattery\s+([\w\s,\/\-]{1,15}\s+)?(dead|low|issue|backup|faulty|replace|swollen|drain|draining|weak)\b|"
    r"\b(dead|low|faulty|replace|swollen|drain|draining|weak)\s+([\w\s,\/\-]{1,15}\s+)?battery\b|"
    r"\bbattery\s+backup\s+(?:low|weak|bad|poor|reduced|issue[s]?|\d+\s*min(ute)?s?)\b|"
    r"\b(?:low|weak|bad|poor|reduced)\s+battery\s+backup\b",
    re.IGNORECASE
)

PIXEL_MASK = re.compile(r"\bdead\s+pixel[s]?\b|\bblemish\s+on\s+screen\b", re.IGNORECASE)

START_BOUND = r"\b(?<!no\s)(?<!zero\s)(?<!without\s)(?<!free\sfrom\s)"
END_BOUND = r"\b"

# V6 DFA State Tiers
CRITICAL_PAT = re.compile(
    START_BOUND + r"(dead|fried|burnt|no power|parts only|water damage|cracked screen|as-is|broken display|won't turn on|motherboard issue|shattered|logic board|spilled)" + END_BOUND,
    re.IGNORECASE,
)

MAJOR_PAT = re.compile(
    START_BOUND + r"(lines on|dead key|overheat|faulty|missing key|loud fan|thermal issue|flickering|ghosting|lines|__BATTERY_FAIL__)" + END_BOUND,
    re.IGNORECASE,
)

MINOR_PAT = re.compile(
    START_BOUND + r"(scratched|scratches|dent|dents|scuff|scuffs|wear|blemish|faded|peeling|chipped|crack on case|signs of use|__PIXEL_WARN__)" + END_BOUND,
    re.IGNORECASE
)

def calculate_degradation_v6(text: str) -> float:
    """
    V6 Production Hardened Extractor
    Includes character class delimiter fixes and absolute structural floors
    to protect gradient scaling from denominator suppression.
    """
    if not isinstance(text, str):
        return 0.0

    clean_text = text.strip()
    if not clean_text:
        return 0.0

    # Step 1: Sanitation & Entity Masking
    sanitized = NEGATION_STRIP_PAT.sub("", clean_text)
    sanitized = GAME_MASK.sub("__SOFTWARE_TITLE__", sanitized)
    sanitized = BATTERY_MASK.sub("__BATTERY_FAIL__", sanitized)
    sanitized = PIXEL_MASK.sub("__PIXEL_WARN__", sanitized)

    # Step 2: Allocation-free denominator computation
    w_count = max(sanitized.count(' ') + 1, 10)

    # Step 3: DFA count execution
    c3 = len(CRITICAL_PAT.findall(sanitized))
    c2 = len(MAJOR_PAT.findall(sanitized))
    c1 = len(MINOR_PAT.findall(sanitized))

    raw_score = (c3 * 1.0) + (c2 * 0.5) + (c1 * 0.2)

    # Step 4: Premium Mitigation
    if any(p in clean_text.lower() for p in ["mint condition", "flawless", "showroom condition", "like new"]):
        raw_score *= 0.5

    # Base continuous calculation
    final_score = float(raw_score / np.log(w_count))

    # Step 5 (V6 Upgrade): Structural Floor Override
    # If a critical motherboard/brick failure exists, enforce an absolute minimum score.
    # This prevents extremely long, verbose descriptions of destroyed hardware
    # from being mathematically suppressed below minor cosmetic wear.
    if c3 > 0 and "motherboard" in clean_text.lower():
        final_score = max(final_score, 0.4000)

    return final_score

# ==========================================
# Execution on df_final_engineered
# ==========================================

df_final_engineered["condition_degradation_index"] = df_final_engineered["description"].apply(calculate_degradation_v6)

# ==========================================
# Phase 3: Telemetry & Verification (v6 Upgrades)
# ==========================================

test_phrases = {
    "Anomaly 1 (Delimiter Isolation)": "ASUS VivoBook functional but has weak/dead battery.",
    "Anomaly 5 (Structural Floor)": "Lenovo LOQ gaming laptop. The screen is perfect, keys are perfect, case is completely pristine, however the motherboard is completely dead and won't turn on."
}

print("=== V6 PRODUCTION HARDENED TRACE ===")
for name, desc in test_phrases.items():
    score = calculate_degradation_v6(desc)
    print(f"[{name}]")
    print(f" -> Score: {score:.4f} | Desc: '{desc}'")

print("\n=== METRIC STABILITY CONFIRMATION ===")
print("If Anomaly 5 hits 0.4000 exactly, the Structural Floor Override is functional.")

# ==========================================
# NEXT CELL
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== OVERALL DESCRIPTIVE STATISTICS ===")
print(df_final_engineered["condition_degradation_index"].describe())
print("\n")

# 1. Investigate the Non-Zero (Defective) Subset
# Since we expect a large pristine baseline (0.0), looking only at the non-zero values
# gives us a clearer picture of how the actual defect scores are distributed.
defective_subset = df_final_engineered[df_final_engineered["condition_degradation_index"] > 0.0]

print("=== DESCRIPTIVE STATISTICS (DEFECTIVE UNITS ONLY) ===")
print(defective_subset["condition_degradation_index"].describe())
print("\n")

# 2. Check for Extreme Anomalies (Percentiles)
print("=== PERCENTILE DISTRIBUTION ===")
percentiles = [0.90, 0.95, 0.99, 0.999]
print(df_final_engineered["condition_degradation_index"].quantile(percentiles))
print("\n")

# 3. Inspect the Absolute Highest Outliers
# This helps verify if a massive score is due to a valid disaster of a laptop,
# or if it's a math/regex anomaly (e.g. extremely short text with multiple keyword triggers).
print("=== TOP 5 MOST EXTREME ANOMALIES ===")
extreme_cols = ["description", "condition_degradation_index"]
top_anomalies = df_final_engineered.sort_values(by="condition_degradation_index", ascending=False).head(5)

for idx, row in top_anomalies.iterrows():
    print(f"Score: {row['condition_degradation_index']:.4f}")
    print(f"Desc:  {row['description']}")
    print("-" * 50)

# 4. Optional: Visual Distribution (Histogram)
# This requires matplotlib to be installed in your environment
try:
    plt.figure(figsize=(10, 5))

    # Plotting only the defective units to make the distribution visible
    # (otherwise the massive 0.0 bar crushes the scale)
    plt.hist(defective_subset["condition_degradation_index"], bins=50, color='crimson', edgecolor='black')
    plt.title("Distribution of Condition Degradation Index (Excluding 0.0 Pristine Baseline)")
    plt.xlabel("Degradation Score")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()
except Exception as e:
    print(f"\nCould not generate plot: {e}")

# ==========================================
# NEXT CELL
# ==========================================

# 1. Print the columns as a list
print("Columns in df_final_eng:")
print(df_final_engineered.columns.tolist())

# 2. View DataFrame details, including non-null counts and data types for all columns
df_final_engineered.info()

# 3. View the first few rows of the filtered DataFrame
df_final_engineered.head()

# ==========================================
# NEXT CELL
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# Phase 2, Step 1: The Autoencoder Topology
# ==========================================

class LaptopAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(LaptopAutoencoder, self).__init__()

        # Encoder Network
        self.enc_dense1 = nn.Linear(input_dim, 64)
        self.enc_dense2 = nn.Linear(64, 32)

        # Layer Normalization before bottleneck ensures zero-centered, unit variance
        # structure without batch-dependency contamination
        self.layer_norm = nn.LayerNorm(32)
        self.bottleneck = nn.Linear(32, latent_dim)

        # Decoder Network
        self.dec_dense1 = nn.Linear(latent_dim, 32)
        self.dec_dense2 = nn.Linear(32, 64)
        self.output_layer = nn.Linear(64, input_dim)

        # Strictly GELU (Gaussian Error Linear Unit) to prevent dead neurons
        self.activation = nn.GELU()

    def encode(self, x):
        x = self.activation(self.enc_dense1(x))
        x = self.activation(self.enc_dense2(x))
        x = self.layer_norm(x)
        z = self.bottleneck(x)
        return z

    def decode(self, z):
        x = self.activation(self.dec_dense1(z))
        x = self.activation(self.dec_dense2(x))
        x = self.output_layer(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out, z

# ==========================================
# Delta Update: extract_latent_manifold (Hardware Accelerated)
# ==========================================

def extract_latent_manifold(X_df, latent_dim=8, epochs=150, batch_size=64, patience=10):
    """
    Train-quarantined autoencoder with early stopping.
    Scaler fits on train only. AE trains on train only. Z extracted for all rows.
    """
    X_all = X_df.to_numpy(dtype=np.float32)

    # Extract and remove the is_global_train mask (always column 0 in features_to_encode)
    is_train_col = X_all[:, 0].astype(bool)
    X_all = np.delete(X_all, 0, axis=1)  # Drop is_global_train from features

    input_dim = X_all.shape[1]

    # LEAKAGE FIX: Fit scaler ONLY on training rows
    scaler = StandardScaler()
    scaler.fit(X_all[is_train_col])
    X_scaled = scaler.transform(X_all)  # Transform all rows with train statistics

    X_train_scaled = X_scaled[is_train_col]

    # Hardware Device Allocation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{'CUDA ALLOCATED' if device.type == 'cuda' else 'CPU FALLBACK'}] Routing tensor ops to {device}")

    model = LaptopAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # LEAKAGE FIX: Train ONLY on training rows
    tensor_X_train = torch.tensor(X_train_scaled, dtype=torch.float32)

    # Early stopping: split training data into 85% fit / 15% validation
    n_train = len(tensor_X_train)
    n_val = int(n_train * 0.15)
    # V2 FIX [C8]: Seed the permutation for reproducible early-stopping splits
    perm_gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_train, generator=perm_gen)
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]

    tensor_fit = tensor_X_train[fit_idx]
    tensor_val = tensor_X_train[val_idx]

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tensor_fit),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda')
    )

    # Training Loop with Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            batch_x = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

        # Validation loss check
        model.eval()
        with torch.no_grad():
            val_x = tensor_val.to(device)
            val_recon, _ = model(val_x)
            val_loss = criterion(val_recon, val_x).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[EARLY STOP] Epoch {epoch+1}: val_loss stalled at {best_val_loss:.6f} for {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Extract Z for ALL rows (train + test) using the train-fitted model
    model.eval()
    tensor_X_all = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        tensor_X_gpu = tensor_X_all.to(device)
        _, Z = model(tensor_X_gpu)

    Z_numpy = Z.cpu().numpy()

    # Telemetry: Posterior Collapse Check
    z_variance = np.var(Z_numpy, axis=0)
    print("=== AUTOENCODER TELEMETRY ===")
    print(f"Latent Node Variances: {np.round(z_variance, 4)}")
    if np.any(z_variance < 1e-4):
        print("[CRITICAL WARNING] Posterior Collapse: A bottleneck node has died. Reduce regularization.")
    else:
        print("[PASS] Latent manifold is active and propagating variance.")

    return model, scaler, Z_numpy, is_train_col
# ==========================================
# Phase 2, Step 4: GMM Fitting (EM) & BIC
# ==========================================

def fit_gmm_find_k(Z, Z_train_mask, max_k=15):
    """
    GMM fitting with train/test quarantine.
    Fits GMM on train Z only. BIC evaluated on train Z only.
    """
    Z_train = Z[Z_train_mask]
    bics = []
    k_range = list(range(3, max_k + 1))
    models = []

    jitter = 1e-6
    # V2 FIX [C9]: Seed jitter RNG for reproducible GMM boundaries
    jitter_rng = np.random.RandomState(42)
    Z_train_jittered = Z_train + jitter_rng.normal(0, jitter, Z_train.shape)

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, reg_covar=1e-5)
        gmm.fit(Z_train_jittered)
        bics.append(gmm.bic(Z_train_jittered))
        models.append(gmm)

    bics_arr = np.array(bics)
    second_deriv = np.diff(bics_arr, 2)
    optimal_idx = np.argmax(second_deriv) + 1
    optimal_k = k_range[optimal_idx]
    best_gmm = models[optimal_idx]

    print("\n=== GMM TELEMETRY ===")
    print(f"Optimal Expert Cohorts (K) Selected via BIC Elbow: {optimal_k}")

    try:
        plt.figure(figsize=(8, 4))
        plt.plot(k_range, bics, marker='o', color='royalblue', linewidth=2)
        plt.axvline(x=optimal_k, color='crimson', linestyle='--', label=f'Optimal Cutoff K={optimal_k}')
        plt.title('GMM Bayesian Information Criterion (BIC) vs. Clusters')
        plt.xlabel('Number of Clusters / Expert Models (K)')
        plt.ylabel('BIC Score')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    except Exception as e:
        pass

    return best_gmm, optimal_k

# ==========================================
# Phase 2, Step 5: The Routing Matrix
# ==========================================

def generate_routing_matrix(gmm, Z, tau=0.85):
    # Generates soft-assignment N x K probability matrix
    probabilities = gmm.predict_proba(Z)

    max_probs = np.max(probabilities, axis=1)
    cluster_assignments = np.argmax(probabilities, axis=1)

    # Threshold Masking: Route to expert cluster, or fallback (-1) if confidence < tau
    final_routes = np.where(max_probs >= tau, cluster_assignments, -1)

    fallback_count = np.sum(final_routes == -1)
    total = len(final_routes)

    print("\n=== ROUTING TELEMETRY ===")
    print(f"Total Dataset Listings:    {total}")
    print(f"Routed to Expert Models:   {total - fallback_count} (Max Prob >= {tau})")
    print(f"Routed to Global Fallback: {fallback_count} (Max Prob < {tau})")

    return probabilities, final_routes

# ==========================================
# Execution Loop
# ==========================================

# 1. Select the continuous and target-encoded features to build your manifold
# V2 FIX [C4]: target_enc columns REMOVED.
# Routing based on target-encoded columns creates latent space leakage because target_enc 
# contains training-set price information. The GMM must route on purely hardware/topology features.
# Also added listing_age_days and is_ssd.
features_to_encode = [
    'is_global_train', 'ram_size_gb', 'storage_size_gb',
    'clean_display_inches', 'condition_degradation_index',
    'cpu_tier', 'gpu_tier', 'is_discrete', 'listing_age_days'
]

# Use df_final_engineered as active dataframe
X_input = df_final_engineered[features_to_encode].copy()

import os
import joblib

ae_weights = '../models/laptop_encoder_weights.pth'
scaler_path = '../models/laptop_feature_scaler.joblib'
gmm_path = '../models/laptop_routing_gmm.joblib'

if os.path.exists(ae_weights) and os.path.exists(scaler_path) and os.path.exists(gmm_path):
    print("\n[CACHE] Topological Routing Manifolds found! Bypassing Autoencoder & GMM sweeps.")
    
    scaler = joblib.load(scaler_path)
    best_gmm = joblib.load(gmm_path)
    
    X_all = X_input.to_numpy(dtype=np.float32)
    X_all = np.delete(X_all, 0, axis=1) # drop is_global_train
    X_scaled = scaler.transform(X_all)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder_model = LaptopAutoencoder(input_dim=X_scaled.shape[1], latent_dim=8).to(device)
    autoencoder_model.load_state_dict(torch.load(ae_weights, map_location=device, weights_only=True))
    autoencoder_model.eval()
    
    with torch.no_grad():
        tensor_X_gpu = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        _, Z = autoencoder_model(tensor_X_gpu)
    Z_matrix = Z.cpu().numpy()
    
    routing_probabilities, expert_assignments = generate_routing_matrix(best_gmm, Z_matrix, tau=0.85)

else:
    print("\n[INIT] Igniting Hardware-Accelerated Autoencoder & Bayesian GMM Sweep...")
    # 2. Extract Latent Space Z (train-quarantined)
    autoencoder_model, scaler, Z_matrix, ae_train_mask = extract_latent_manifold(X_input, latent_dim=8)

    # 3. Fit Gaussian Mixture on TRAIN Z only, predict_proba on ALL Z
    best_gmm, optimal_k = fit_gmm_find_k(Z_matrix, ae_train_mask, max_k=15)

    # 4. Generate Soft Assignments & Gating Routes (for ALL rows using train-fitted GMM)
    routing_probabilities, expert_assignments = generate_routing_matrix(best_gmm, Z_matrix, tau=0.85)

    # 5. Serialize
    torch.save(autoencoder_model.state_dict(), ae_weights)
    joblib.dump(scaler, scaler_path)
    joblib.dump(best_gmm, gmm_path)
    print("[SYSTEM] Autoencoder and GMM State serialized.")
    sync_to_drive()

# Attach routes back to your main dataframe
df_final_engineered['expert_model_route'] = expert_assignments

# ==========================================
# NEXT CELL
# ==========================================

import torch
import torch.nn as nn
import joblib
import numpy as np
from scipy.stats import entropy

# ==========================================
# 1. Redefine Architecture
# ==========================================
def assign_expert_algorithms_optimized(df):
    HIGH_DEG_THRESH = 0.1500
    HIGH_ENTROPY_THRESH = 1.25
    HIGH_CARDINALITY_THRESH = 15

    expert_config_dict = {}
    print("=== TOPOLOGICAL PROFILING TELEMETRY ===")

    # LATERAL OPTIMIZATION 1: O(N) Hash-map partitioning
    # Drops the O(K * N) Boolean masking bottleneck
    grouped = df[df['expert_model_route'] >= 0].groupby('expert_model_route')

    for cluster_id, subset in grouped:
        n_rows = len(subset)

        if n_rows < 100:
            assigned_algo = "Ridge"
            reason = "Cluster Size < 100 (Forced Linear Fallback)"
            price_cv, gpu_entropy, deg_mean = 0.0, 0.0, 0.0
        else:
            # V2 FIX [H8]: Compute Coefficient of Variation on RAW price, not log-space
            # Computing CV on log(price) gives statistically meaningless results for routing.
            price_array_raw = np.expm1(subset['price_normalized'].values)

            # Mathematical Safety: Prevent division by absolute zero
            price_mean = np.mean(price_array_raw)
            price_cv = np.std(price_array_raw) / price_mean if price_mean != 0 else 0

            deg_mean = np.mean(subset['condition_degradation_index'].values)
            discrete_ratio = np.mean(subset['is_discrete'].values)

            # Entropy on pre-computed value counts
            # V2 FIX [C10]: Strip unused categories before entropy to prevent phantom zero-count inflation
            gpu_probs = subset['gpu_vendor'].cat.remove_unused_categories().value_counts(normalize=True).values
            gpu_entropy = entropy(gpu_probs, base=2)
            brand_cardinality = subset['clean_brand'].nunique()

            # Logic Branching
            if deg_mean > HIGH_DEG_THRESH and price_cv < 0.2:
                assigned_algo, reason = "Ridge", "Scrap/Linear Cohort"
            elif discrete_ratio > 0.8 and price_cv > 0.5:
                assigned_algo, reason = "XGBoost", "Gaming Cohort (High CV)"
            elif gpu_entropy > HIGH_ENTROPY_THRESH or brand_cardinality > HIGH_CARDINALITY_THRESH:
                assigned_algo, reason = "CatBoost", "Sparse Enterprise"
            else:
                assigned_algo, reason = "LightGBM", "Standard Productivity"

        expert_config_dict[cluster_id] = {
            'algorithm': assigned_algo,
            'size_n': n_rows,
            'hyperparameters': f"{assigned_algo}_default_grid"
        }

        print(f"Cluster {cluster_id} | Size: {n_rows} | Alg: {assigned_algo} | CV: {price_cv:.3f}")

    return expert_config_dict

# Execute the hyper-optimized C-backend routing script
expert_configurations = assign_expert_algorithms_optimized(df_final_engineered)

# ==========================================
# NEXT CELL
# ==========================================



# ==========================================
# NEXT CELL
# ==========================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# BLOCK 1: GLOBAL TOPOLOGICAL QUARANTINE
# ==========================================
print("Initiating Memory Partitioning...")

# We anchor the stratification to the GMM routing vector to preserve cluster density
df_train = df_final_engineered[df_final_engineered['is_global_train'] == True].reset_index(drop=True)
df_test = df_final_engineered[df_final_engineered['is_global_train'] == False].reset_index(drop=True)

# Force contiguous memory reallocation to prevent Stacking Index Faults
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"Global Train Matrix Allocated : {len(df_train)} rows")
print(f"Global Holdout Matrix Locked  : {len(df_test)} rows")

# Verify Topological Symmetry
print("\n=== Train Manifold Density ===")
print(df_train['expert_model_route'].value_counts(normalize=True).round(3))
print("\n=== Holdout Manifold Density ===")
print(df_test['expert_model_route'].value_counts(normalize=True).round(3))

# ==========================================
# NEXT CELL
# ==========================================

import os
import joblib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    RandomTreesEmbedding,
    StackingRegressor
)
from sklearn.linear_model import HuberRegressor, RidgeCV, Ridge
from sklearn.compose import TransformedTargetRegressor
from lightgbm import LGBMRegressor

# ==========================================
# BLOCK 2: THE META-ENSEMBLE STACKING ENGINE
# ==========================================
# V2 FIX [M5]: OMP Thread limits should only apply during ProcessPoolExecutor
# Not set globally here.

# The 21-Dimensional Structural Constraint
EXPLICIT_FEATURES = [
    'images_count', 'storage_size_gb', 'ram_size_gb', 'fingerprint',
    'touch', 'warranty', 'urgent_sell', 'is_discrete', 'clean_display_inches',
    'cpu_tier', 'gpu_tier', 'state_target_enc', 'city_target_enc',
    'clean_model_series_target_enc', 'gpu_vendor_target_enc',
    'cpu_composite_clean_target_enc', 'clean_brand_target_enc',
    'clean_gpu_composite_target_enc', 'condition_degradation_index',
    'listing_age_days'
]

def train_cluster_stack(cluster_id, X_train, y_train, target_col):
    print(f"[PID {os.getpid()}] Initiating Cluster {cluster_id} | Compiling Stacking Engine...")

    os.makedirs('weights', exist_ok=True)
    artifact_path = f"weights/expert_c{cluster_id}_Stack.joblib"
    if os.path.exists(artifact_path):
        print(f"[PID {os.getpid()}] [CACHE] Skipping Cluster {cluster_id} | Serialized Artifact Exists.")
        return artifact_path

    # ---------------------------------------------------------
    # DYNAMIC BAYESIAN TUNING (Replacing Hard-Coded Trees)
    # ---------------------------------------------------------
    import optuna
    from sklearn.model_selection import cross_val_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective_et(trial):
        model = ExtraTreesRegressor(
            n_estimators=trial.suggest_int('n_estimators', 50, 200, step=50),
            max_depth=trial.suggest_int('max_depth', 5, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
            max_features=trial.suggest_float('max_features', 0.5, 1.0),
            random_state=42, n_jobs=1
        )
        return np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='r2', n_jobs=1))

    def objective_rf(trial):
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int('n_estimators', 50, 200, step=50),
            max_depth=trial.suggest_int('max_depth', 5, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
            max_features=trial.suggest_float('max_features', 0.5, 1.0),
            random_state=42, n_jobs=1
        )
        return np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='r2', n_jobs=1))

    print(f"[PID {os.getpid()}] Cluster {cluster_id} | Hunting Dynamic Tree Topologies...")
    study_et = optuna.create_study(direction="maximize")
    study_et.optimize(objective_et, n_trials=10)

    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(objective_rf, n_trials=10)

    # ---------------------------------------------------------
    # LEVEL 0: Uncorrelated Base Geometries
    # ---------------------------------------------------------
    level_0 = [
        # Anchor Bagging (Dynamically Injected)
        ('rf', RandomForestRegressor(**study_rf.best_params, random_state=42, n_jobs=1)),
        # Extreme Randomization (Dynamically Injected)
        ('et', ExtraTreesRegressor(**study_et.best_params, random_state=42, n_jobs=1)),
        # Boosting
        ('lgbm', LGBMRegressor(objective='regression', random_state=42, n_jobs=1, verbose=-1)),
        # Scaled Linear Outlier Resistance
        ('huber', make_pipeline(RobustScaler(), HuberRegressor(epsilon=1.35, max_iter=500))),
        # Obscure Topology: Unsupervised Tree Paths mapped to Linear Euclidean Space
        ('rte_ridge', make_pipeline(
            RandomTreesEmbedding(n_estimators=100, max_depth=5, random_state=42, n_jobs=1),
            Ridge(alpha=1.0)
        ))
    ]

    # ---------------------------------------------------------
    # LEVEL 1: RidgeCV Meta-Learner (5-Fold Native)
    # ---------------------------------------------------------
    meta_stack = StackingRegressor(
        estimators=level_0,
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=1 # Critical to prevent nested deadlock
    )

    # ---------------------------------------------------------
    # Target Wrapper & Execution
    # ---------------------------------------------------------
    # V2 FIX [H5]: Explicit Log-Space flag, decoupled from column name heuristic
    TARGET_IS_LOG_SPACE = True

    if not TARGET_IS_LOG_SPACE:
        final_artifact = TransformedTargetRegressor(
            regressor=meta_stack,
            func=np.log1p,
            inverse_func=np.expm1
        )
    else:
        # If target is already logged, do not wrap at all.
        final_artifact = meta_stack

    # Trigger massive internal 5-Fold fitting sequence
    final_artifact.fit(X_train, y_train)

    os.makedirs('weights', exist_ok=True)
    artifact_path = f"weights/expert_c{cluster_id}_Stack.joblib"
    joblib.dump(final_artifact, artifact_path)

    return artifact_path

def execute_stacking_ensemble(df, target_col='price_normalized'):
    tasks = []

    # Target Sanity
    apply_log = ('normalized' not in target_col)
    if apply_log:
        df = df[df[target_col] > 0].copy()
    else:
        df = df.copy()

    valid_clusters = df['expert_model_route'].unique()
    max_workers = max(1, os.cpu_count() - 1)

    print(f"=== INITIATING META-STACKING ENSEMBLE ({max_workers} WORKERS) ===")
    print("WARNING: Level 1 Meta-Fitting executes 500% more computations. Monitoring VRAM...\n")

    # V2 FIX [M5]: Scope thread limits to the ProcessPool block
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for cid in valid_clusters:
            subset = df if cid == -1 else df[df['expert_model_route'] == cid]

            # Explicit 21-Dimensional Slicing to prevent leakage
            X_train = subset[EXPLICIT_FEATURES]
            y_train = subset[target_col]

            future = executor.submit(train_cluster_stack, cid, X_train, y_train, target_col)
            tasks.append(future)

        for future in as_completed(tasks):
            try:
                artifact = future.result()
                print(f"[SYSTEM] Meta-Stack successfully serialized: {artifact}")
            except Exception as e:
                print(f"[FATAL EXCEPTION] Process failed: {e}")
                
    # Restore threads after parallel block
    os.environ.pop("OMP_NUM_THREADS", None)
    os.environ.pop("MKL_NUM_THREADS", None)

# TRIGGER META-ENSEMBLE ON THE 75% QUARANTINE MATRIX
execute_stacking_ensemble(df_train, target_col='price_normalized')
sync_to_drive()

# ==========================================
# NEXT CELL
# ==========================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================
# BLOCK 3: GENERALIZED INFERENCE AUDIT
# ==========================================

def evaluate_stacking_holdout(df_test, target_col='price_normalized'):
    print("==========================================")
    print("=== GENERALIZED INFERENCE AUDIT (25%) ===")
    print("==========================================\n")

    # 1. Target Transformations
    df_eval = df_test.copy()
    
    # V2 FIX [H5]: Explicit boolean flag
    TARGET_IS_LOG_SPACE = True
    
    if not TARGET_IS_LOG_SPACE:
        df_eval = df_eval[df_eval[target_col] > 0]

    valid_clusters = sorted(df_eval['expert_model_route'].unique())

    overall_y_true = []
    overall_y_pred = []

    # 19-Dimensional Structural Constraint
    EXPLICIT_FEATURES = [
        'images_count', 'storage_size_gb', 'ram_size_gb', 'fingerprint',
        'touch', 'warranty', 'urgent_sell', 'is_discrete', 'clean_display_inches',
        'cpu_tier', 'gpu_tier', 'state_target_enc', 'city_target_enc',
        'clean_model_series_target_enc', 'gpu_vendor_target_enc',
        'cpu_composite_clean_target_enc', 'clean_brand_target_enc',
        'clean_gpu_composite_target_enc', 'condition_degradation_index',
        'listing_age_days'
    ]

    for cid in valid_clusters:
        subset = df_eval if cid == -1 else df_eval[df_eval['expert_model_route'] == cid]

        # Explicit 19-Dimensional Slicing
        X_test = subset[EXPLICIT_FEATURES].to_numpy(dtype=np.float32)
        y_test = subset[target_col].values

        print(f"[Cluster {cid}] Processing {len(subset)} Holdout Coordinates...")

        path = f"weights/expert_c{cid}_Stack.joblib"
        if not os.path.exists(path):
            print(f"  -> [FATAL] Stack Artifact {path} missing. Bypassing cluster.")
            continue

        # 2. Singular Meta-Learner Forward Pass
        meta_stack = joblib.load(path)
        preds = meta_stack.predict(X_test)

        # 3. Float64 IEEE-754 Overflow Sanitization
        if not TARGET_IS_LOG_SPACE:
            # Catch expm1() infinities generated internally by the wrapper
            preds = np.nan_to_num(preds, posinf=np.max(y_test)*2, neginf=0.0)
        else:
            preds = np.nan_to_num(preds, posinf=np.max(y_test)*2, neginf=0.0)

        # 4. Cluster Telemetry
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        print(f"  -> Meta-Stack RÂ²  : {r2:7.4f} | MAE: {mae:9,.2f}")
        print("-" * 50)

        overall_y_true.extend(y_test)
        overall_y_pred.extend(preds)

    # 5. Global Generalized Limits
    # V2 FIX [M7]: Report both log-space (optimized) and raw price RÂ²
    global_log_r2 = r2_score(overall_y_true, overall_y_pred)
    
    overall_y_true_raw = np.expm1(overall_y_true) if TARGET_IS_LOG_SPACE else overall_y_true
    overall_y_pred_raw = np.expm1(overall_y_pred) if TARGET_IS_LOG_SPACE else overall_y_pred
    
    global_raw_r2 = r2_score(overall_y_true_raw, overall_y_pred_raw)
    global_mae = mean_absolute_error(overall_y_true_raw, overall_y_pred_raw)
    global_rmse = np.sqrt(mean_squared_error(overall_y_true_raw, overall_y_pred_raw))

    print("==========================================")
    print("=== TRUE GLOBAL GENERALIZED VARIANCE ===")
    print("==========================================")
    print(f"Generalized Global RÂ² (Log)   : {global_log_r2:.4f}")
    print(f"Generalized Global RÂ² (Raw)   : {global_raw_r2:.4f}")
    print(f"Generalized Global MAE        : {global_mae:,.2f} INR")
    print(f"Generalized Global RMSE : {global_rmse:,.2f} INR")
    print("==========================================")

# TRIGGER EVALUATION ON THE 25% HOLDOUT MATRIX
evaluate_stacking_holdout(df_test, target_col='price_normalized')

# ==========================================
# NEXT CELL
# ==========================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# BLOCK 3.1: META-STACK X-RAY DECONSTRUCTION
# ==========================================

def xray_stacking_estimators(df_test, target_col='price_normalized'):
    print("==========================================")
    print("=== META-STACK X-RAY DECONSTRUCTION ===")
    print("==========================================\n")

    df_eval = df_test.copy()
    
    # V2 FIX [H5]: Explicit log flag
    TARGET_IS_LOG_SPACE = True
    
    if not TARGET_IS_LOG_SPACE:
        df_eval = df_eval[df_eval[target_col] > 0]

    valid_clusters = sorted(df_eval['expert_model_route'].unique())

    EXPLICIT_FEATURES = [
        'images_count', 'storage_size_gb', 'ram_size_gb', 'fingerprint',
        'touch', 'warranty', 'urgent_sell', 'is_discrete', 'clean_display_inches',
        'cpu_tier', 'gpu_tier', 'state_target_enc', 'city_target_enc',
        'clean_model_series_target_enc', 'gpu_vendor_target_enc',
        'cpu_composite_clean_target_enc', 'clean_brand_target_enc',
        'clean_gpu_composite_target_enc', 'condition_degradation_index',
        'listing_age_days'
    ]

    for cid in valid_clusters:
        subset = df_eval if cid == -1 else df_eval[df_eval['expert_model_route'] == cid]
        X_test = subset[EXPLICIT_FEATURES].to_numpy(dtype=np.float32)
        y_test = subset[target_col].values

        path = f"weights/expert_c{cid}_Stack.joblib"
        if not os.path.exists(path):
            continue

        master_artifact = joblib.load(path)

        # 1. Penetrate the TransformedTargetRegressor Wrapper
        if hasattr(master_artifact, 'regressor_'):
            stack = master_artifact.regressor_
        else:
            stack = master_artifact

        print(f"\n[CLUSTER {cid}] Topology Breakdown ({len(subset)} rows)")
        print("-" * 60)

        # 2. Extract Meta-Learner Coefficients
        weights = stack.final_estimator_.coef_
        model_names = list(stack.named_estimators_.keys())

        print("Meta-Learner Assigned Weights (Ridge L2):")
        for name, weight in zip(model_names, weights):
            print(f"  -> {name:15}: {weight:7.4f}")
        print("-" * 60)

        # 3. Individual Base-Learner Execution
        print("Isolated Base-Learner Performance (Out-Of-Sample):")
        for name, estimator in stack.named_estimators_.items():

            # Predict purely on the isolated Level 0 model
            raw_preds = estimator.predict(X_test)

            # Apply manual target inversion if wrapper was bypassed (meaning TARGET_IS_LOG_SPACE is True)
            if TARGET_IS_LOG_SPACE:
                preds = np.clip(raw_preds, a_min=-10, a_max=700)
                # We need to un-log both preds and y_test for true MAE, but here we keep it symmetric to target metric
                preds = np.nan_to_num(preds, posinf=np.max(y_test)*2, neginf=0.0)
            else:
                preds = np.nan_to_num(raw_preds, posinf=0.0, neginf=0.0)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)

            print(f"  -> {name:15} | RÂ²: {r2:7.4f} | MAE: {mae:9,.2f}")

        # The Master Stack Prediction for contrast
        master_preds = master_artifact.predict(X_test)
        if TARGET_IS_LOG_SPACE:
            master_preds = np.nan_to_num(master_preds, posinf=np.max(y_test)*2, neginf=0.0)
        master_r2 = r2_score(y_test, master_preds)
        
        # Calculate raw R2
        y_test_raw = np.expm1(y_test) if TARGET_IS_LOG_SPACE else y_test
        master_preds_raw = np.expm1(master_preds) if TARGET_IS_LOG_SPACE else master_preds
        master_r2_raw = r2_score(y_test_raw, master_preds_raw)
        
        print(f"\n  => FINAL STACK   | RÂ²(Log): {master_r2:7.4f} | RÂ²(Raw): {master_r2_raw:7.4f}")
        print("==========================================")

# TRIGGER X-RAY
xray_stacking_estimators(df_test, target_col='price_normalized')

# ==========================================
# NEXT CELL
# ==========================================

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor

# The Obscure Theoretical Weapons
from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import SplineTransformer, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, HuberRegressor

# ==========================================
# BLOCK 4: THE GLOBAL ARCHITECTURAL SWEEP
# ==========================================
print("==========================================")
print("=== GLOBAL OBSCURE TOPOLOGY SWEEP (25%) ===")
print("==========================================\n")

EXPLICIT_FEATURES = [
    'images_count', 'storage_size_gb', 'ram_size_gb', 'fingerprint',
    'touch', 'warranty', 'urgent_sell', 'is_discrete', 'clean_display_inches',
    'cpu_tier', 'gpu_tier', 'state_target_enc', 'city_target_enc',
    'clean_model_series_target_enc', 'gpu_vendor_target_enc',
    'cpu_composite_clean_target_enc', 'clean_brand_target_enc',
    'clean_gpu_composite_target_enc', 'condition_degradation_index',
    'listing_age_days'
]

# Extract pure global matrices
X_train_global = df_train[EXPLICIT_FEATURES].to_numpy(dtype=np.float32)
y_train_global = df_train['price_normalized'].to_numpy(dtype=np.float32)

X_test_global = df_test[EXPLICIT_FEATURES].to_numpy(dtype=np.float32)
y_test_global = df_test['price_normalized'].values

# Define the Orthogonal Contenders
global_contenders = {
    'Global_ExtraTrees': ExtraTreesRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1),
    'Hilbert_KernelRidge': make_pipeline(RobustScaler(), KernelRidge(kernel='rbf', alpha=0.1, gamma=0.01)),
    'Spline_Ridge': make_pipeline(RobustScaler(), SplineTransformer(n_knots=10, degree=3), Ridge(alpha=1.0)),
    'Bagged_Huber': BaggingRegressor(
        estimator=make_pipeline(RobustScaler(), HuberRegressor(epsilon=1.35, max_iter=500)),
        n_estimators=20, random_state=42, n_jobs=1
    )
}

for name, model in global_contenders.items():
    print(f"Compiling Architecture: {name}...")

    # price_normalized is already log-scaled â€” no target wrapper needed
    # 1. Global Fit
    model.fit(X_train_global, y_train_global)

    # 2. Global Predict
    preds = model.predict(X_test_global)

    # 3. IEEE-754 Sanitization
    preds = np.nan_to_num(preds, posinf=np.max(y_test_global)*2, neginf=0.0)

    # 4. Telemetry
    r2 = r2_score(y_test_global, preds)
    mae = mean_absolute_error(y_test_global, preds)

    print(f"  -> Out-Of-Sample RÂ² : {r2:7.4f}")
    print(f"  -> Out-Of-Sample MAE: {mae:9,.2f} INR")
    print("-" * 50)

print("\n[REFERENCE] Meta-Stacking MoE Global RÂ² : 0.7624")
print("==========================================")

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# BLOCK 1: GLOBAL TOPOLOGICAL QUARANTINE
# ==========================================

# Secure the Bayesian Optimization Binaries

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print("Initiating Memory Partitioning...")

# We anchor the stratification to the GMM routing vector to preserve latent density
df_train = df_final_engineered[df_final_engineered['is_global_train'] == True].reset_index(drop=True)
df_test = df_final_engineered[df_final_engineered['is_global_train'] == False].reset_index(drop=True)

# Force contiguous memory reallocation to prevent Optuna Index Faults
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"Global Train Matrix Allocated : {len(df_train)} rows")
print(f"Global Holdout Matrix Locked  : {len(df_test)} rows")

# Verify Topological Symmetry
print("\n=== Train Manifold Density ===")
print(df_train['expert_model_route'].value_counts(normalize=True).round(3))
print("\n=== Holdout Manifold Density ===")
print(df_test['expert_model_route'].value_counts(normalize=True).round(3))

# ==========================================
# NEXT CELL
# ==========================================

import optuna
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor

# ==========================================
# BLOCK 2: OPTUNA SUB-ENSEMBLE A (THE TREES)
# ==========================================
print("==========================================")
print("=== OPTUNA BAYESIAN TPE ENSEMBLE INITIATED ===")
print("==========================================\n")

EXPLICIT_FEATURES = [
    'images_count', 'storage_size_gb', 'ram_size_gb', 'fingerprint',
    'touch', 'warranty', 'urgent_sell', 'is_discrete', 'clean_display_inches',
    'cpu_tier', 'gpu_tier', 'state_target_enc', 'city_target_enc',
    'clean_model_series_target_enc', 'gpu_vendor_target_enc',
    'cpu_composite_clean_target_enc', 'clean_brand_target_enc',
    'clean_gpu_composite_target_enc', 'condition_degradation_index',
    'listing_age_days'
]

# Isolate the Global Training Manifold
X_train_global = df_train[EXPLICIT_FEATURES].to_numpy(dtype=np.float32)
y_train_global = df_train['price_normalized'].values

apply_log = False  # target is price_normalized, already log-scaled
optuna.logging.set_verbosity(optuna.logging.WARNING) # Mute excessive logs

def optimize_trees(trial, model_type):
    # 1. Define the Geometric Boundaries
    n_estimators = trial.suggest_int('n_estimators', 200, 500, step=100)
    max_depth = trial.suggest_int('max_depth', 12, 25)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    max_features = trial.suggest_float('max_features', 0.5, 1.0)

    # 2. Instantiate the Selected Architecture
    if model_type == 'ExtraTrees':
        base_model = ExtraTreesRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, max_features=max_features,
            random_state=42, n_jobs=-1
        )
    else:
        base_model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, max_features=max_features,
            random_state=42, n_jobs=-1
        )

    # 3. Apply Logarithmic Symmetry
    if apply_log:
        wrapped_model = TransformedTargetRegressor(
            regressor=base_model, func=np.log1p, inverse_func=np.expm1
        )
    else:
        wrapped_model = base_model

    # 4. Enforce 5-Fold Geometric Isolation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        wrapped_model, X_train_global, y_train_global,
        cv=kf, scoring='r2', n_jobs=1
    )

    return np.mean(scores)

# ---------------------------------------------------------
# EXECUTE EXTRA TREES HUNT
# ---------------------------------------------------------
if not os.path.exists("../models/global_model_model.joblib"):
    print("Hunting ExtraTrees Topology (30 Trials)...")
    study_et = optuna.create_study(direction="maximize")
    study_et.optimize(lambda trial: optimize_trees(trial, 'ExtraTrees'), n_trials=30)
    print(f"[ExtraTrees] Best Internal 5-Fold RÂ²: {study_et.best_value:.4f}")

    # ---------------------------------------------------------
    # EXECUTE RANDOM FOREST HUNT
    # ---------------------------------------------------------
    print("Hunting RandomForest Topology (30 Trials)...")
    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(lambda trial: optimize_trees(trial, 'RandomForest'), n_trials=30)
    print(f"[RandomForest] Best Internal 5-Fold RÂ²: {study_rf.best_value:.4f}")

    # ---------------------------------------------------------
    # SUMMARY TELEMETRY
    # ---------------------------------------------------------
    print("\n==========================================")
    print("=== MATHEMATICALLY OPTIMAL PARAMETERS ===")
    print("==========================================")

    print("\nExtraTrees Optimal Geometry:")
    for key, value in study_et.best_params.items():
        print(f"  -> {key:18}: {value}")

    print("\nRandomForest Optimal Geometry:")
    for key, value in study_rf.best_params.items():
        print(f"  -> {key:18}: {value}")
    print("==========================================")
else:
    print("[CACHE] Skipping Global ExtraTrees/RandomForest Optuna Hunts.")

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# BLOCK 3 CORRECTION: THE GEOMETRY HUNT
# ==========================================

def optimize_geometries(trial, model_type):
    # 1. Instantiate the Scaling Pipeline bare objects
    pipeline_steps = [RobustScaler()]

    # 2. Define the Geometric Boundaries
    if model_type == 'Spline':
        n_knots = trial.suggest_int('n_knots', 4, 15)
        degree = trial.suggest_int('degree', 2, 4)
        alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)

        pipeline_steps.append(SplineTransformer(n_knots=n_knots, degree=degree))
        pipeline_steps.append(Ridge(alpha=alpha))

    elif model_type == 'Kernel':
        alpha = trial.suggest_float('alpha', 0.001, 10.0, log=True)
        gamma = trial.suggest_float('gamma', 0.0001, 0.1, log=True)

        pipeline_steps.append(KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma))

    # Bare objects unpack successfully into make_pipeline
    base_model = make_pipeline(*pipeline_steps)

    # 3. Apply Logarithmic Symmetry
    if apply_log:
        wrapped_model = TransformedTargetRegressor(
            regressor=base_model, func=np.log1p, inverse_func=np.expm1
        )
    else:
        wrapped_model = base_model

    # 4. Enforce 5-Fold Geometric Isolation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        wrapped_model, X_train_global, y_train_global,
        cv=kf, scoring='r2', n_jobs=1
    )

    return np.mean(scores)

# ---------------------------------------------------------
# RE-EXECUTE SUB-ENSEMBLE B
# ---------------------------------------------------------
if not os.path.exists("../models/global_model_model.joblib"):
    print("Hunting Spline Polynomial Topology (30 Trials)...")
    study_spline = optuna.create_study(direction="maximize")
    study_spline.optimize(lambda trial: optimize_geometries(trial, 'Spline'), n_trials=30)
    print(f"[Spline_Ridge] Best Internal 5-Fold RÂ²: {study_spline.best_value:.4f}")

    print("Hunting KernelRidge RBF Topology (30 Trials)...")
    study_kernel = optuna.create_study(direction="maximize")
    study_kernel.optimize(lambda trial: optimize_geometries(trial, 'Kernel'), n_trials=30)
    print(f"[KernelRidge] Best Internal 5-Fold RÂ²: {study_kernel.best_value:.4f}")

    # SUMMARY TELEMETRY
    print("\n==========================================")
    print("=== MATHEMATICALLY OPTIMAL PARAMETERS ===")
    print("==========================================")

    print("\nSpline_Ridge Optimal Geometry:")
    for key, value in study_spline.best_params.items():
        print(f"  -> {key:18}: {value}")

    print("\nKernelRidge Optimal Geometry:")
    for key, value in study_kernel.best_params.items():
        print(f"  -> {key:18}: {value}")
    print("==========================================")
else:
    print("[CACHE] Skipping Global Spline/Kernel Optuna Hunts.")

# ==========================================
# NEXT CELL
# ==========================================

import os
import joblib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, SplineTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.compose import TransformedTargetRegressor

# ==========================================
# BLOCK 4: THE GLOBAL META-STACK COMPILATION
# ==========================================
print("==========================================")
print("=== COMPILING GLOBAL META-STACK ENGINE ===")
print("==========================================\n")

print("\n[EXECUTING 44-FIT COMPILATION CASCADE. PLEASE WAIT...]")

os.makedirs("weights", exist_ok=True)
path = "../models/global_model_model.joblib"

if os.path.exists(path):
    print(f"[CACHE] Skipping Master Fit | Serialized Artifact Exists.")
    global_model = joblib.load(path)
else:
    # [BLOCK 4.1] LEVEL-0 INSTANTIATION (THE BAYESIAN OPTIMIZED GEOMETRIES)
    print("-> Injecting Sub-Ensemble A: Variance Crushers (Trees)...")
    base_et = ExtraTreesRegressor(**study_et.best_params, random_state=42, n_jobs=-1)
    base_rf = RandomForestRegressor(**study_rf.best_params, random_state=42, n_jobs=-1)

    print("-> Injecting Sub-Ensemble B: Continuous Mappers (Geometries)...")
    base_spline = make_pipeline(
        RobustScaler(),
        SplineTransformer(n_knots=study_spline.best_params['n_knots'], degree=study_spline.best_params['degree']),
        Ridge(alpha=study_spline.best_params['alpha'])
    )

    base_kernel = make_pipeline(
        RobustScaler(),
        KernelRidge(kernel='rbf', alpha=study_kernel.best_params['alpha'], gamma=study_kernel.best_params['gamma'])
    )

    level_0_estimators = [
        ('ExtraTrees', base_et),
        ('RandomForest', base_rf),
        ('Spline_Ridge', base_spline),
        ('Kernel_RBF', base_kernel)
    ]

    # [BLOCK 4.2] LEVEL-1 META-LEARNER ISOLATION
    print("-> Instantiating L2-Regularized Meta-Learner (RidgeCV)...")
    meta_learner = make_pipeline(
        RobustScaler(),
        RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    )

    # [BLOCK 4.3] THE 10-FOLD LEAKAGE QUARANTINE
    print("-> Enforcing Strict 10-Fold Out-Of-Fold (OOF) CV Matrix...")
    strict_10_fold = KFold(n_splits=10, shuffle=True, random_state=42)

    master_stack = StackingRegressor(
        estimators=level_0_estimators,
        final_estimator=meta_learner,
        cv=strict_10_fold,
        n_jobs=1  # HARDWARE LOCK: Prevent Colab OOM (SIGKILL -9) from excessive memory replication
    )

    master_stack.fit(X_train_global, y_train_global)
    global_model = master_stack
    joblib.dump(global_model, path)

print(f"\n[SUCCESS] Global Meta-Stack Compiled and Serialized to: {path}")
print("==========================================")
sync_to_drive()

# ==========================================
# NEXT CELL
# ==========================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================
# BLOCK 5: THE FINAL INFERENCE AUDIT & X-RAY
# ==========================================
print("==========================================")
print("=== GLOBAL META-STACK GENERALIZED AUDIT ===")
print("==========================================\n")

EXPLICIT_FEATURES = [
    'images_count', 'storage_size_gb', 'ram_size_gb', 'fingerprint',
    'touch', 'warranty', 'urgent_sell', 'is_discrete', 'clean_display_inches',
    'cpu_tier', 'gpu_tier', 'state_target_enc', 'city_target_enc',
    'clean_model_series_target_enc', 'gpu_vendor_target_enc',
    'cpu_composite_clean_target_enc', 'clean_brand_target_enc',
    'clean_gpu_composite_target_enc', 'condition_degradation_index',
    'listing_age_days'
]

# 1. Isolate the Holdout Test Manifold
X_test_global = df_test[EXPLICIT_FEATURES].to_numpy(dtype=np.float32)
y_test_global = df_test['price_normalized'].values

# 2. Load the Serialized Master Artifact
path = "../models/global_model_model.joblib"
if not os.path.exists(path):
    raise FileNotFoundError(f"[FATAL] Master artifact not found at {path}")

global_model = joblib.load(path)

# 3. Penetrate Wrappers for X-Ray
print("-> Deconstructing Level-1 RidgeCV Meta-Weights...")
stacking_layer = global_model
meta_pipeline = stacking_layer.final_estimator_
meta_ridge = meta_pipeline.named_steps['ridgecv']

weights = meta_ridge.coef_
model_names = list(stacking_layer.named_estimators_.keys())

print("-" * 50)
print("L2-Regularized Geometric Synthesis:")
for name, weight in zip(model_names, weights):
    print(f"  -> {name:15}: {weight:7.4f}")
print("-" * 50)

# 4. Execute Global Inference Pass
print("\n-> Executing Global Out-Of-Sample Forward Pass...")
preds = global_model.predict(X_test_global)

# --- UN-LOG FOR ABSOLUTE TELEMETRY ---
y_test_global = np.expm1(y_test_global)
preds = np.expm1(preds)

# 5. IEEE-754 Extreme Overflow Sanitization
preds = np.nan_to_num(preds, posinf=np.max(y_test_global)*2, neginf=0.0)

# 6. Final Telemetry Limits
global_r2 = r2_score(y_test_global, preds)
global_mae = mean_absolute_error(y_test_global, preds)
global_rmse = np.sqrt(mean_squared_error(y_test_global, preds))

print("\n==========================================")
print("=== TRUE GLOBAL GENERALIZED VARIANCE ===")
print("==========================================")
print(f"Generalized Global RÂ²   : {global_r2:.4f}")
print(f"Generalized Global MAE  : {global_mae:,.2f} INR")
print(f"Generalized Global RMSE : {global_rmse:,.2f} INR")
print("==========================================")

# ==========================================
# NEXT CELL
# ==========================================

import pandas as pd
import shutil

print("-> Serializing Master Topological Matrix...")
# Save the master dataframe with no index to prevent offset corruption
df_final_engineered.to_csv('../data/Kaggle_Master_Engineered_State.csv', index=False)

print("-> Zipping execution weights (Optional History)...")
# Zip the weights folder just in case you want to audit the global stack later
shutil.make_archive('Global_Weights_Archive', 'zip', 'weights')

print("\n[INITIATING BROWSER DOWNLOAD PROTOCOL]")
try:
    from google.colab import files
    files.download('../data/Kaggle_Master_Engineered_State.csv')
    files.download('Global_Weights_Archive.zip')
except ImportError:
    print("[SYSTEM] Local execution detected. Colab download protocol bypassed.")
except AttributeError:
    print("[SYSTEM] Script execution detected without IPython kernel. Please download files manually from the Colab sidebar.")
except Exception as e:
    print(f"[SYSTEM] Colab download protocol bypassed: {e}")

print("Extraction Complete. You are cleared to terminate the runtime.")
sync_to_drive()

# ==========================================
# NEXT CELL
# ==========================================

# V2 FIX [M8]: Removed redundant train/test memory re-allocation. 
# df_train and df_test are already globally persistent and stratified.

EXPLICIT_FEATURES = [
    'images_count', 'storage_size_gb', 'ram_size_gb', 'fingerprint',
    'touch', 'warranty', 'urgent_sell', 'is_discrete', 'clean_display_inches',
    'cpu_tier', 'gpu_tier', 'state_target_enc', 'city_target_enc',
    'clean_model_series_target_enc', 'gpu_vendor_target_enc',
    'cpu_composite_clean_target_enc', 'clean_brand_target_enc',
    'clean_gpu_composite_target_enc', 'condition_degradation_index',
    'listing_age_days'
]

# ==========================================
# BLOCK 1: THE MOE Model (24-STUDY HUNT)
# ==========================================
print("\n==========================================")
print("=== INITIATING MOE Model BAYESIAN HUNT ===")
print("==========================================\n")

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. The Dynamic Optimization Functions
def optimize_trees(trial, X, y, model_type):
    n_estimators = trial.suggest_int('n_estimators', 200, 500, step=100)
    max_depth = trial.suggest_int('max_depth', 12, 25)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    max_features = trial.suggest_float('max_features', 0.5, 1.0)

    if model_type == 'ExtraTrees':
        base_model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=42, n_jobs=-1)
    else:
        base_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=42, n_jobs=-1)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return np.mean(cross_val_score(base_model, X, y, cv=kf, scoring='r2', n_jobs=1))

def optimize_geometries(trial, X, y, model_type):
    pipeline_steps = [RobustScaler()]

    if model_type == 'Spline':
        n_knots = trial.suggest_int('n_knots', 4, 15)
        degree = trial.suggest_int('degree', 2, 4)
        alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
        pipeline_steps.extend([SplineTransformer(n_knots=n_knots, degree=degree), Ridge(alpha=alpha)])
    else:
        alpha = trial.suggest_float('alpha', 0.001, 10.0, log=True)
        gamma = trial.suggest_float('gamma', 0.0001, 0.1, log=True)
        pipeline_steps.append(KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma))

    base_model = make_pipeline(*pipeline_steps)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return np.mean(cross_val_score(base_model, X, y, cv=kf, scoring='r2', n_jobs=1))

# 2. Master Execution Loop
# V3 FIX: Load serialized params first to prevent KeyError on partial cache hit
cluster_params_path = '../models/optimization_params.joblib'
if os.path.exists(cluster_params_path):
    cluster_params_dict = joblib.load(cluster_params_path)
    print(f"[CACHE] Loaded existing Cluster Params from {cluster_params_path}")
else:
    cluster_params_dict = {}

valid_clusters = sorted(df_train['expert_model_route'].unique())
# valid_clusters will usually be -1 to 5.

for cid in valid_clusters:
    print(f"\n[CLUSTER {cid}] LOCKING TOPOLOGICAL COORDINATES...")
    print("-" * 50)

    path = f"../models/moe_expert_weights/Expert_C{cid}.joblib"
    if os.path.exists(path) and cid in cluster_params_dict:
        print(f"[CACHE] Skipping Cluster Cluster {cid} Optuna Hunt | Serialized Artifact Exists.")
        continue

    # Isolate cluster training space
    subset = df_train if cid == -1 else df_train[df_train['expert_model_route'] == cid]
    X_train_c = subset[EXPLICIT_FEATURES].to_numpy(dtype=np.float32)
    y_train_c = subset['price_normalized'].values

    cluster_params = {}

    # Sub-Ensemble A (The Trees)
    for model_name in ['ExtraTrees', 'RandomForest']:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial, mt=model_name: optimize_trees(trial, X_train_c, y_train_c, mt), n_trials=15)
        cluster_params[model_name] = study.best_params
        print(f"  -> {model_name:15} | Best RÂ²: {study.best_value:7.4f}")

    # Sub-Ensemble B (The Geometries)
    for model_name in ['Spline', 'Kernel']:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial, mt=model_name: optimize_geometries(trial, X_train_c, y_train_c, mt), n_trials=15)
        cluster_params[model_name] = study.best_params
        print(f"  -> {model_name:15} | Best RÂ²: {study.best_value:7.4f}")

    cluster_params_dict[cid] = cluster_params

# V3 FIX: Always re-serialize params after loop to capture any newly computed clusters
joblib.dump(cluster_params_dict, cluster_params_path)
print(f"[SYSTEM] Cluster params serialized to {cluster_params_path}")
sync_to_drive()

print("\n==========================================")
print("=== Optimization phase completed ===")
print("==========================================")

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# BLOCK 2: THE HEX-STACK COMPILATION
# ==========================================
import os
from sklearn.ensemble import StackingRegressor
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import RidgeCV

print("==========================================")
print("=== COMPILING LOCALIZED HEX-STACKS ===")
print("==========================================\n")

os.makedirs("../models/moe_expert_weights", exist_ok=True)
compiled_stacks = {}

for cid in valid_clusters:
    print(f"-> Forging Cluster Stack for Cluster [{cid}]...")

    path = f"../models/moe_expert_weights/Expert_C{cid}.joblib"
    if os.path.exists(path):
        print(f"   [CACHE] Skipping Cluster Cluster {cid} Stack Compilation | Serialized Artifact Exists.")
        compiled_stacks[cid] = joblib.load(path)
        continue

    # 1. Isolate the optimized params for this specific cluster
    p = cluster_params_dict[cid]

    # 2. Reconstruct the base models with the perfect Bayesian specs
    base_et = ExtraTreesRegressor(**p['ExtraTrees'], random_state=42, n_jobs=-1)
    base_rf = RandomForestRegressor(**p['RandomForest'], random_state=42, n_jobs=-1)

    base_spline = make_pipeline(
        RobustScaler(),
        SplineTransformer(n_knots=p['Spline']['n_knots'], degree=p['Spline']['degree']),
        Ridge(alpha=p['Spline']['alpha'])
    )

    base_kernel = make_pipeline(
        RobustScaler(),
        KernelRidge(kernel='rbf', alpha=p['Kernel']['alpha'], gamma=p['Kernel']['gamma'])
    )

    level_0 = [
        ('ExtraTrees', base_et),
        ('RandomForest', base_rf),
        ('Spline', base_spline),
        ('Kernel', base_kernel)
    ]

    # 3. Construct the localized Squad Commander (RidgeCV)
    meta_learner = make_pipeline(RobustScaler(), RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]))

    # 4. Enforce the 10-Fold Quarantine
    strict_cv = KFold(n_splits=10, shuffle=True, random_state=42)
    master_stack = StackingRegressor(estimators=level_0, final_estimator=meta_learner, cv=strict_cv, n_jobs=1)

    # 5. Fit the Stack on the Localized Cluster Data
    subset = df_train if cid == -1 else df_train[df_train['expert_model_route'] == cid]
    X_c = subset[EXPLICIT_FEATURES].to_numpy(dtype=np.float32)
    y_c = subset['price_normalized'].values

    master_stack.fit(X_c, y_c)

    # 6. Serialize to Disk
    path = f"../models/moe_expert_weights/Expert_C{cid}.joblib"
    joblib.dump(master_stack, path)
    compiled_stacks[cid] = master_stack

# ==========================================
# BLOCK 3: Inference testing phase
# ==========================================
print("\n==========================================")
print("=== THE FINAL GENERALIZED AUDIT ===")
print("==========================================\n")

print("-> Routing Holdout Matrix through Cluster Stacks...")

df_test_copy = df_test.copy()
df_test_copy['cluster_prediction'] = np.nan

# Route the testing rows to their respective Expert models
for cid in valid_clusters:
    mask = df_test_copy['expert_model_route'] == cid
    if mask.sum() == 0:
        continue

    X_test_c = df_test_copy.loc[mask, EXPLICIT_FEATURES].to_numpy(dtype=np.float32)
    expert_model = compiled_stacks[cid]

    preds = expert_model.predict(X_test_c)
    df_test_copy.loc[mask, 'cluster_prediction'] = preds

# Fallback mechanism for unrouted data (Defaulting to Global Model Cluster -1)
missing_mask = df_test_copy['cluster_prediction'].isna()
if missing_mask.sum() > 0:
    print(f"[WARNING] {missing_mask.sum()} rows failed routing. Engaging Global Fallback (-1).")
    X_test_fallback = df_test_copy.loc[missing_mask, EXPLICIT_FEATURES].to_numpy(dtype=np.float32)
    fallback_model = compiled_stacks[-1]
    df_test_copy.loc[missing_mask, 'cluster_prediction'] = fallback_model.predict(X_test_fallback)

# Extreme IEEE-754 Sanitization
y_true = np.expm1(df_test_copy['price_normalized'].values)
y_pred = np.expm1(df_test_copy['cluster_prediction'].values)
y_pred = np.nan_to_num(y_pred, posinf=np.max(y_true)*2, neginf=0.0)

global_r2 = r2_score(y_true, y_pred)
global_mae = mean_absolute_error(y_true, y_pred)
global_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("==========================================")
print(f"MOE Model GLOBAL RÂ²   : {global_r2:.4f}")
print(f"MOE Model GLOBAL MAE  : {global_mae:,.2f} INR")
print(f"MOE Model GLOBAL RMSE : {global_rmse:,.2f} INR")
print("==========================================")
sync_to_drive()

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# ISOLATED PHASE A: LOCKING THE CLUSTER STATE
# ==========================================
import json
import joblib

print("-> Extracting Cluster Parameters...")
# Surgically cast numpy.int64 to strings exclusively for the JSON visualizer
json_safe_dict = {str(k): v for k, v in cluster_params_dict.items()}
print(json.dumps(json_safe_dict, indent=4))

# Physically lock the raw matrix to disk
joblib.dump(cluster_params_dict, '../models/optimization_params.joblib')
print("\n[LOCKED] Parameters successfully serialized to ../models/optimization_params.joblib")

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# BLOCK 1.5: THE CLUSTER X-RAY (HUMAN TRANSLATION)
# ==========================================

print("-> Aggregating Physical Hardware Personas per Cluster...")

# Calculate median stats and percentage of gaming GPUs
cluster_xray = df_train.groupby('expert_model_route').agg(
    Total_Laptops=('price', 'count'),
    Median_Price_INR=('price', 'median'),
    Median_RAM_GB=('ram_size_gb', 'median'),
    Percent_Discrete_GPU=('is_discrete', lambda x: (x.mean() * 100).round(1)),
    Most_Common_CPU=('cpu_tier', lambda x: x.mode()[0] if not x.empty else 'Unknown')
).reset_index()

# Format the output for extreme readability
cluster_xray['Median_Price_INR'] = cluster_xray['Median_Price_INR'].apply(lambda x: f"â‚¹ {x:,.0f}")
cluster_xray['Percent_Discrete_GPU'] = cluster_xray['Percent_Discrete_GPU'].astype(str) + "%"

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n=== THE MATHEMATICAL TRANSLATION MATRIX ===")
print(cluster_xray.to_string(index=False))

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# BLOCK 1.6: THE BUDGET ENSEMBLE DEEP X-RAY
# ==========================================

print("-> Isolating the Budget Ensemble (Clusters 1, 2, 3)...")

budget_mask = df_train['expert_model_route'].isin([1, 2, 3])
budget_df = df_train[budget_mask]

# Aggregate the invisible secondary dimensions
budget_xray = budget_df.groupby('expert_model_route').agg(
    Median_Storage_GB=('storage_size_gb', 'median'),
    Median_Display_Inches=('clean_display_inches', 'median'),
    Percent_Touchscreen=('touch', lambda x: (x.mean() * 100).round(1)),
    Percent_Fingerprint=('fingerprint', lambda x: (x.mean() * 100).round(1)),
    Percent_Warranty=('warranty', lambda x: (x.mean() * 100).round(1)),
    Median_Degradation=('condition_degradation_index', 'median')
).reset_index()

budget_xray['Percent_Touchscreen'] = budget_xray['Percent_Touchscreen'].astype(str) + "%"
budget_xray['Percent_Fingerprint'] = budget_xray['Percent_Fingerprint'].astype(str) + "%"
budget_xray['Percent_Warranty'] = budget_xray['Percent_Warranty'].astype(str) + "%"

print("\n=== THE SECONDARY HARDWARE VARIANCE ===")
print(budget_xray.to_string(index=False))

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# PHASE B1: QUARANTINE & INFERENCE GENERATION
# ==========================================
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

print("==========================================")
print("=== PHASE B1: DATA QUARANTINE & INFERENCE ===")
print("==========================================\n")

print("-> Slicing Holdout Matrix for Anti-Leakage Quarantine (50/50)...")
# V2 FIX [C5]: Replace positional indexing with proper random split to avoid positional bias
df_blend_tune, df_blind_audit = train_test_split(df_test, test_size=0.5, random_state=42)
df_blend_tune = df_blend_tune.reset_index(drop=True)
df_blind_audit = df_blind_audit.reset_index(drop=True)

print("-> Loading 156 MB Global generalist model...")
if not os.path.exists('../models/global_model_model.joblib'):
    raise FileNotFoundError("[CRITICAL] ../models/global_model_model.joblib is missing. Upload it immediately.")
global_model = joblib.load('../models/global_model_model.joblib')
print("   [SUCCESS] Global Global loaded into RAM.")

# Function to route laptops through the localized Hex-Stacks in RAM
def generate_moe_predictions(df_input):
    df_pred = df_input.copy()
    df_pred['moe_pred'] = np.nan

    for cid in valid_clusters:
        mask = df_pred['expert_model_route'] == cid
        if mask.sum() > 0:
            X_c = df_pred.loc[mask, EXPLICIT_FEATURES].to_numpy(dtype=np.float32)
            df_pred.loc[mask, 'moe_pred'] = compiled_stacks[cid].predict(X_c)

    # Global Fallback (-1) for routing anomalies
    missing = df_pred['moe_pred'].isna()
    if missing.sum() > 0:
        df_pred.loc[missing, 'moe_pred'] = compiled_stacks[-1].predict(df_pred.loc[missing, EXPLICIT_FEATURES].to_numpy(dtype=np.float32))
    return df_pred['moe_pred'].values

print("\n-> Generating Competing Predictions on df_blend_tune...")
y_true_tune = np.expm1(df_blend_tune['price_normalized'].values)
preds_global_tune = np.expm1(global_model.predict(df_blend_tune[EXPLICIT_FEATURES].to_numpy(dtype=np.float32)))
preds_moe_tune = np.expm1(generate_moe_predictions(df_blend_tune))

print("   [SUCCESS] Tune Predictions Generated.")
print(f"   Shape of True Tuning Targets : {y_true_tune.shape}")
print(f"   Shape of Global Tuning Preds : {preds_global_tune.shape}")
print(f"   Shape of MoE Tuning Preds    : {preds_moe_tune.shape}")

print("\n[PHASE B1 COMPLETE] Ready for Nelder-Mead Optimization.")


# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# PHASE B2: THE NELDER-MEAD OPTIMIZATION
# ==========================================
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error

print("==========================================")
print("=== PHASE B2: NELDER-MEAD OPTIMIZATION ===")
print("==========================================\n")

# The Nelder-Mead Optimization Function
def blend_objective(weights):
    w_global, w_moe = weights
    pred_blend = (w_global * preds_global_tune) + (w_moe * preds_moe_tune)
    return mean_absolute_error(y_true_tune, pred_blend)

print("-> Igniting SciPy Simplex Optimizer on Tuning Matrix...")
# Initial guess is [0.5, 0.5], bounded between 0 and 1
opt_res = minimize(blend_objective, [0.5, 0.5], method='Powell', bounds=[(0, 1), (0, 1)])
w_global_opt, w_moe_opt = opt_res.x

# Normalization Protocol (Ensuring exactly 1.0 total sum)
total_w = w_global_opt + w_moe_opt
w_global_opt /= total_w
w_moe_opt /= total_w

print(f"   [LOCKED] Global Generalist Weight : {w_global_opt:.4f}")
print(f"   [LOCKED] MoE Specialists Weight   : {w_moe_opt:.4f}")

print("\n[PHASE B2 COMPLETE] Ready for True Blind Audit.")

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# PHASE B3: THE ZERO-THEATER BLIND AUDIT
# ==========================================
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("==========================================")
print("=== PHASE B3: TRUE BLIND AUDIT EXECUTION ===")
print("==========================================\n")

print("-> Extracting Quarantined Target Array...")
y_true_blind = np.expm1(df_blind_audit['price_normalized'].values)

print("-> Routing Blind Matrix through Global and MoE Stacks...")
preds_global_blind = np.expm1(global_model.predict(df_blind_audit[EXPLICIT_FEATURES].to_numpy(dtype=np.float32)))
preds_moe_blind = np.expm1(generate_moe_predictions(df_blind_audit))

print("-> Fusing Architectural Weights...")
# Applying the mathematically locked fractions from Phase B2
final_blind_preds = (w_global_opt * preds_global_blind) + (w_moe_opt * preds_moe_blind)

# Extreme IEEE-754 Sanitization (Neutering floating-point anomalies)
final_blind_preds = np.nan_to_num(final_blind_preds, posinf=np.max(y_true_blind)*2, neginf=0.0)

print("-> Calculating Absolute Telemetry...")
final_r2 = r2_score(y_true_blind, final_blind_preds)
final_mae = mean_absolute_error(y_true_blind, final_blind_preds)
final_rmse = np.sqrt(mean_squared_error(y_true_blind, final_blind_preds))

print("\n==========================================")
print(f"Final Model Pipeline RÂ²   : {final_r2:.4f}")
print(f"Final Model Pipeline MAE  : {final_mae:,.2f} INR")
print(f"Final Model Pipeline RMSE : {final_rmse:,.2f} INR")
print("==========================================")

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# PHASE B5: ARCHITECTURE SERIALIZATION
# ==========================================
import joblib

print("==========================================")
print("=== PHASE B5: LOCKING FINAL ARCHITECTURE ===")
print("==========================================\n")

print("-> Compiling Master Inference Payload...")

# We fuse the trained MoE models and the Blending Weights into a single dictionary
final_architecture = {
    'moe_stacks': compiled_stacks,
    'blend_weights': {
        'global_weight': w_global_opt,
        'moe_weight': w_moe_opt
    }
}

print("-> Serializing to Solid State Drive...")
# Physically lock to disk
file_name = '../models/blended_architecture.joblib'
joblib.dump(final_architecture, file_name)
sync_to_drive()

# Get the file size to verify successful chunking
file_size_mb = os.path.getsize(file_name) / (1024 * 1024)

print("\n==========================================")
print(f"[LOCKED] Architecture saved to: {file_name}")
print(f"[METADATA] File Size: {file_size_mb:.2f} MB")
print("==========================================")

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# PHASE C1: BASELINE ASYMMETRIC AUDIT
# ==========================================
import numpy as np

print("==========================================")
print("=== PHASE C1: ASYMMETRIC RISK BASELINE ===")
print("==========================================\n")

def asymmetric_risk_score(y_true, y_pred, over_penalty=10, under_penalty=1):
    """
    Calculates custom risk:
    - Over-valuation (y_pred > y_true) penalized by 10x
    - Under-valuation (y_pred <= y_true) penalized by 1x
    """
    error = y_pred - y_true

    # Apply asymmetrical multipliers
    penalties = np.where(error > 0, error * over_penalty, np.abs(error) * under_penalty)

    return np.mean(penalties)

# Calculate the baseline risk of the CURRENT symmetric model
print("-> Auditing current symmetric predictions against 10x penalty...")
baseline_risk = asymmetric_risk_score(y_true_blind, final_blind_preds)

print("\n==========================================")
print(f"[DANGER] BASELINE ASYMMETRIC RISK SCORE: {baseline_risk:,.2f}")
print("==========================================")
print("\n[PHASE C1 COMPLETE] Ready to optimize markdown scalar.")

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# PHASE C2: THE ASYMMETRIC SHIFT HUNT
# ==========================================
from scipy.optimize import minimize

print("==========================================")
print("=== PHASE C2: NELDER-MEAD SCALAR HUNT ===")
print("==========================================\n")

print("-> Re-compiling Symmetrical Tuning Predictions...")
pred_blend_tune = (w_global_opt * preds_global_tune) + (w_moe_opt * preds_moe_tune)

# The Scalar Objective Function
def scalar_objective(M):
    # Apply the scalar multiplier M to the symmetric predictions
    shifted_preds = pred_blend_tune * M[0]
    return asymmetric_risk_score(y_true_tune, shifted_preds)

print("-> Igniting SciPy Simplex Optimizer against 10x Penalty...")
# Initial guess is a 10% markdown (0.90), bounded between 50% and 100%
opt_res = minimize(scalar_objective, [0.90], method='Powell', bounds=[(0.5, 1.0)])
optimal_scalar = opt_res.x[0]

# Calculate the actual percentage markdown
markdown_percentage = (1.0 - optimal_scalar) * 100

print(f"   [LOCKED] Optimal Markdown Scalar : {optimal_scalar:.4f}")
print(f"   [LOCKED] Effective Business Cut  : {markdown_percentage:.2f}%")

print("\n[PHASE C2 COMPLETE] Ready for Final Arbitrage Deployment.")

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# PHASE C3: THE ARBITRAGE DEPLOYMENT
# ==========================================
from sklearn.metrics import r2_score

print("==========================================")
print("=== PHASE C3: TRUE BLIND DEPLOYMENT ===")
print("==========================================\n")

print(f"-> Applying Locked Scalar ({optimal_scalar:.4f}) to Blind Audit Matrix...")
# Shift the symmetric predictions downward by the mathematically proven margin
shifted_blind_preds = final_blind_preds * optimal_scalar

print("-> Calculating New Asymmetric Risk...")
new_asymmetric_risk = asymmetric_risk_score(y_true_blind, shifted_blind_preds)

# Calculate Risk Reduction
risk_reduction = ((baseline_risk - new_asymmetric_risk) / baseline_risk) * 100

print("-> Calculating New (Degraded) Academic RÂ²...")
degraded_r2 = r2_score(y_true_blind, shifted_blind_preds)

print("\n==========================================")
print(f"ORIGINAL SYMMETRIC RISK : {baseline_risk:,.2f}")
print(f"NEW ARBITRAGE RISK      : {new_asymmetric_risk:,.2f}")
print(f"TOTAL RISK REDUCTION    : {risk_reduction:.2f} %")
print("==========================================")
print(f"\n[METADATA] Academic RÂ² dropped from {final_r2:.4f} to {degraded_r2:.4f}")
print("This RÂ² degradation is the mathematical signature of your profit margin.")

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# PHASE C4: ARBITRAGE SERIALIZATION
# ==========================================
import os
import joblib

print("==========================================")
print("=== PHASE C4: LOCKING ACQUISITION ENGINE ===")
print("==========================================\n")

print("-> Compiling Master Arbitrage Payload...")

# Fusing the 3 Tiers: MoE Models, Blend Weights, and the Asymmetric Scalar
arbitrage_payload = {
    'moe_stacks': compiled_stacks,
    'blend_weights': {
        'global_weight': w_global_opt,
        'moe_weight': w_moe_opt
    },
    'arbitrage_scalar': optimal_scalar
}

print("-> Serializing Arbitrage Engine to Solid State Drive...")
# Physically lock to disk
file_name = '../models/arbitrage_model_moe.joblib'
joblib.dump(arbitrage_payload, file_name)
sync_to_drive()

# Get the file size to verify successful chunking
file_size_mb = os.path.getsize(file_name) / (1024 * 1024)

print("\n==========================================")
print(f"[LOCKED] Arbitrage Engine saved to: {file_name}")
print(f"[METADATA] File Size: {file_size_mb:.2f} MB")
print("==========================================")

# ==========================================
# NEXT CELL
# ==========================================

# ==========================================
# PHASE C5: RISK TOPOGRAPHY BY PRICE BAND
# ==========================================
import numpy as np

print("==========================================")
print("=== PHASE C5: ASYMMETRIC RISK BY PRICE BAND ===")
print("==========================================\n")

print(f"{'PRICE SPAN (INR)':<20} | {'BASE RISK':<12} | {'NEW RISK':<12} | {'RISK REDUCTION'}")
print("-" * 65)

# Iterate from 5k to 85k in 10k steps (5k-15k, 15k-25k, etc.)
for lower_bound in range(5000, 85000, 10000):
    upper_bound = lower_bound + 10000

    # Isolate the laptops that physically fall into this price tranche
    mask = (y_true_blind >= lower_bound) & (y_true_blind < upper_bound)

    # If there are laptops in this bin, calculate the localized risk
    if np.sum(mask) > 0:
        y_true_bin = y_true_blind[mask]
        sym_preds_bin = final_blind_preds[mask]
        arb_preds_bin = shifted_blind_preds[mask]

        # Calculate baseline symmetric risk for this specific bin
        base_risk = asymmetric_risk_score(y_true_bin, sym_preds_bin)

        # Calculate the new safe arbitrage risk for this specific bin
        new_risk = asymmetric_risk_score(y_true_bin, arb_preds_bin)

        # Calculate the localized risk reduction percentage
        reduction = ((base_risk - new_risk) / base_risk) * 100

        print(f"{lower_bound:5,d} to {upper_bound:6,d}   | {base_risk:10,.1f} | {new_risk:10,.1f} |   {reduction:5.1f}%")

print("\n==========================================")
print("TELEMETRY COMPLETE.")


