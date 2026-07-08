import threading
_MODEL_LOCK = threading.Lock()

import os
import sys
import re
import warnings
import joblib

import sys
# Add current dir to path to import knn_target_encoder
sys.path.append(r"./audit_and_fixes\fixes")
try:
    from knn_target_encoder import KnnTargetEncoder
except ImportError as e:
    raise RuntimeError("Critical dependency knn_target_encoder is missing.") from e

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import data_supplier
import sklearn.preprocessing
import scipy.interpolate
import pyarrow as pa
import pyarrow.compute as pc
from datetime import datetime, timezone, timedelta
# HOTFIX: Scikit-learn version mismatch for SplineTransformer unpickling
if hasattr(sklearn.preprocessing, 'SplineTransformer') and not hasattr(sklearn.preprocessing.SplineTransformer, 'handle_missing'):
    sklearn.preprocessing.SplineTransformer.handle_missing = 'error'

# HOTFIX: Scipy version mismatch for BSpline unpickling
if hasattr(scipy.interpolate, 'BSpline'):
    if not hasattr(scipy.interpolate.BSpline, '_asarray'):
        scipy.interpolate.BSpline._asarray = lambda self, x: np.asarray(x)
    
    def _bspline_getattr(self, name):
        if name == '_c' and 'c' in self.__dict__:
            return self.__dict__['c']
        if name == '_t' and 't' in self.__dict__:
            return self.__dict__['t']
        if name == '_k' and 'k' in self.__dict__:
            return self.__dict__['k']
        raise AttributeError(f"'BSpline' object has no attribute '{name}'")
    scipy.interpolate.BSpline.__getattr__ = _bspline_getattr

warnings.filterwarnings('ignore')

# ==========================================
# CONSTANTS & CACHE
# ==========================================
CACHE_PATH = '../models/arbitrage_cache_state.joblib'
ENGINE_PATH = '../models/arbitrage_model_moe.joblib'
SCALER_PATH = '../models/laptop_feature_scaler.joblib'
GMM_PATH = '../models/laptop_routing_gmm.joblib'
AE_WEIGHTS_PATH = '../models/laptop_encoder_weights.pth'

# TARGET_COLS ordering is CRITICAL: 'state' must precede 'city' 
# because 'city' fallback relies on 'state_target_enc' already existing.
TARGET_COLS = [
    'state', 'city', 'clean_model_series', 'gpu_vendor',
    'cpu_composite_clean', 'clean_brand', 'clean_gpu_composite'
]

EXPLICIT_FEATURES = [
    'images_count', 'storage_size_gb', 'ram_size_gb', 'fingerprint',
    'touch', 'warranty', 'urgent_sell', 'is_discrete', 'clean_display_inches',
    'cpu_tier', 'gpu_tier', 'state_target_enc', 'city_target_enc',
    'clean_model_series_target_enc', 'gpu_vendor_target_enc',
    'cpu_composite_clean_target_enc', 'clean_brand_target_enc',
    'clean_gpu_composite_target_enc', 'condition_degradation_index',
    'listing_age_days'
]

class UnseenCategoryError(Exception):
    pass

# ==========================================
# AUTOENCODER TOPOLOGY
# ==========================================
class LaptopAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(LaptopAutoencoder, self).__init__()
        self.enc_dense1 = nn.Linear(input_dim, 64)
        self.enc_dense2 = nn.Linear(64, 32)
        self.layer_norm = nn.LayerNorm(32)
        self.bottleneck = nn.Linear(32, latent_dim)
        self.dec_dense1 = nn.Linear(latent_dim, 32)
        self.dec_dense2 = nn.Linear(32, 64)
        self.output_layer = nn.Linear(64, input_dim)
        self.activation = nn.GELU()

    def encode(self, x):
        x = self.activation(self.enc_dense1(x))
        x = self.activation(self.enc_dense2(x))
        x = self.layer_norm(x)
        return self.bottleneck(x)

    def decode(self, z):
        x = self.activation(self.dec_dense1(z))
        x = self.activation(self.dec_dense2(x))
        return self.output_layer(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

# ==========================================
# PIPELINE FUNCTIONS
# ==========================================

def enforce_bounds(df, price_min=0.0):
    if len(df) == 0: return df
    initial_len = len(df)
    
    # Enforce basic numeric limits on display
    if 'clean_display_inches' in df.columns:
        df['clean_display_inches'] = pd.to_numeric(df['clean_display_inches'], errors='coerce')
        df = df[(df['clean_display_inches'] >= 10.0) & (df['clean_display_inches'] <= 18.5)].copy()
    
    df['storage_size_gb'] = pd.to_numeric(df['storage_size_gb'], errors='coerce')
    df['storage_size_gb'] = df['storage_size_gb'].replace({1000: 1024, 2000: 2048, 500: 512, 128: 128})
    df['storage_size_gb'] = df['storage_size_gb'].replace(0, np.nan).fillna(256)
    df['ram_size_gb'] = pd.to_numeric(df['ram_size_gb'], errors='coerce')
    df['ram_size_gb'] = df['ram_size_gb'].replace(0, np.nan).fillna(8)
    df['storage_tech'] = 'SSD'  # Implicit constraint
    df = df[(df['storage_size_gb'] > 0) & (df['storage_size_gb'] <= 8192)].copy()
    df = df[(df['ram_size_gb'] > 0) & (df['ram_size_gb'] <= 200)].copy()
    
    if 'price' in df.columns and price_min > 0:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[df['price'] >= price_min].copy()

    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"[ARBITRAGE] Enforce bounds dropped {dropped} rows.")
    return df

_HARDWARE_TAXONOMY = [
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
        (r'\b(elitebook|elite\s?book|eleitbook|elightbook|elitbook)\b', 'HP EliteBook'),
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
_COMPILED_HARDWARE = [(re.compile(p, re.IGNORECASE), r) for p, r in _HARDWARE_TAXONOMY]

def compile_hardware_pipeline(df, target_col='model_series'):
    def map_series(entry):
        if pd.isna(entry): return 'Other / Unclassified Hardware'
        e_str = str(entry).strip()
        for pat, repl in _COMPILED_HARDWARE:
            if pat.search(e_str): return repl
        return 'Other / Unclassified Hardware'
        
    df['clean_model_series'] = df[target_col].apply(map_series).astype('category')
    return df

def extract_gpu_topology_matrix(df, target_col='gpu'):
    discrete_pattern = re.compile(r'\b(rtx|gtx|rx|quadro|t\d{3,4}|arc|firepro|mobility\s?radeon|t500|mx\d{3})\b', re.IGNORECASE)
    intel_pattern = re.compile(r'\b(intel|uhd|hd|iris|arc)\b', re.IGNORECASE)
    amd_pattern = re.compile(r'\b(amd|radeon|vega|rx|apu)\b', re.IGNORECASE)
    nvidia_pattern = re.compile(r'\b(nvidia|nvida|nvdia|gtx|rtx|quadro|geforce|gforce|mx)\b', re.IGNORECASE)

    def extract(raw):
        if pd.isna(raw): return 0, 'UNKNOWN'
        clean = str(raw).lower()
        is_discrete = 1 if discrete_pattern.search(clean) else 0
        if 'integrated' in clean or 'igpu' in clean or 'inbuilt' in clean or 'shared' in clean:
            is_discrete = 0
            
        vendor = 'UNKNOWN'
        if nvidia_pattern.search(clean): vendor = 'NVIDIA'
        elif amd_pattern.search(clean): vendor = 'AMD'
        elif intel_pattern.search(clean): vendor = 'INTEL'
        return is_discrete, vendor

    res = df[target_col].apply(extract)
    df[['is_discrete', 'gpu_vendor']] = pd.DataFrame(res.tolist(), index=df.index)
    return df

def compile_heterogeneous_cpu_pipeline(df, target_col='cpu'):
    qualcomm_pattern = re.compile(r'snapdragon\s*x\s*(elite|plus)', re.IGNORECASE)
    qualcomm_sku = re.compile(r'(x1e\d{5}|x1p\d{5})', re.IGNORECASE)
    amd_ryzen_ai = re.compile(r'ryzen\s*ai\s*([579])\s*(hx)?\s*(\d{3})', re.IGNORECASE)
    amd_ryzen_std = re.compile(r'ryzen\s*([3579])\s*(\d{4})\s*([a-z]{1,2})?', re.IGNORECASE)
    amd_legacy_apu = re.compile(r'\b(a4|a6|a8|a9|a10|a12|athlon|gold|silver)\s*[-_]?\s*(\d{4})?([a-z])?', re.IGNORECASE)
    intel_ultra = re.compile(r'(core\s*ultra|ultra)\s*([579])\s*(?:series\s*\d+)?\s*(?:modifier)?\s*(\d{3,5})?([a-z]{1,2})?', re.IGNORECASE)
    intel_core = re.compile(r'(?:core\s*)?i([3579])[-_]?\s*(\d{2,5})([a-z]{1,2})?', re.IGNORECASE)

    def map_cpu(raw):
        if pd.isna(raw): return "UNCLASSIFIED_HARDWARE_CORE"
        clean = str(raw).lower().strip()
        
        if 'snapdragon' in clean or 'qualcomm' in clean:
            tier_match = qualcomm_pattern.search(clean)
            sku_match = qualcomm_sku.search(clean)
            tier = tier_match.group(1).upper() if tier_match else "X"
            sku = sku_match.group(1).upper() if sku_match else "GENERIC"
            return f"QUALCOMM_SNAPDRAGON_{tier}_{sku}"
            
        if 'amd' in clean or 'ryzen' in clean or 'r3' in clean or 'r5' in clean or 'r7' in clean or 'r9' in clean:
            ai_match = amd_ryzen_ai.search(clean)
            std_match = amd_ryzen_std.search(clean)
            legacy_match = amd_legacy_apu.search(clean)

            if ai_match:
                tier = ai_match.group(1)
                suffix = ai_match.group(2).upper() if ai_match.group(2) else "STD"
                sku = ai_match.group(3)
                return f"AMD_RYZEN_AI{tier}_SKU{sku}_{suffix}"
            elif std_match:
                tier = std_match.group(1)
                sku = std_match.group(2)
                suffix = std_match.group(3).upper() if std_match.group(3) else "STD"
                return f"AMD_RYZEN_{tier}_SKU{sku}_{suffix}"
            elif legacy_match:
                family = legacy_match.group(1).upper()
                sku = legacy_match.group(2) if legacy_match.group(2) else "GENERIC"
                suffix = legacy_match.group(3).upper() if legacy_match.group(3) else "STD"
                return f"AMD_LEGACY_{family}_SKU{sku}_{suffix}"
            return "AMD_RYZEN_GENERIC_UNKNOWN"
            
        ultra_match = intel_ultra.search(clean)
        core_match = intel_core.search(clean)

        if ultra_match:
            tier = ultra_match.group(2)
            sku = ultra_match.group(3) if ultra_match.group(3) else "GENERIC"
            suffix = ultra_match.group(4).upper() if ultra_match.group(4) else "STD"
            return f"INTEL_ULTRA_{tier}_SKU{sku}_{suffix}"
        elif core_match:
            tier = core_match.group(1)
            sku = core_match.group(2)
            suffix = core_match.group(3).upper() if core_match.group(3) else "STD"
            return f"INTEL_CORE_I{tier}_SKU{sku}_{suffix}"
        else:
            if 'celeron' in clean: return "INTEL_CELERON_STD"
            elif 'pentium' in clean: return "INTEL_PENTIUM_STD"
            return "UNCLASSIFIED_HARDWARE_CORE"

    df['cpu_composite_clean'] = df[target_col].apply(map_cpu).astype('category')
    return df

_BRAND_TAXONOMY = [
        (r'\b(dell/lenovo|lenovo/dell)\b', 'DELL_LENOVO_BUNDLE'),
        (r'\b(dell/gigabyte|gigabyte/dell)\b', 'DELL_GIGABYTE_BUNDLE'),
        (r'\b(msi/asrock|msi/lg|intel/zotac|ibm\s*/\s*cisco|zotac/asus)\b', 'COMPOSITE_HARDWARE_BUNDLE'),
        (r'^(custom|assembled|pc setup)\b', 'ASSEMBLED_CUSTOM'),
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
_COMPILED_BRAND = [(re.compile(p, re.IGNORECASE), r) for p, r in _BRAND_TAXONOMY]

def compile_brand_normalization_pipeline(df, target_col='brand'):
    def map_brand(raw):
        if pd.isna(raw): return 'UNKNOWN_OEM'
        clean = str(raw).strip().lower()
        for pat, repl in _COMPILED_BRAND:
            if pat.search(clean): return repl
        return 'UNKNOWN_OEM'

    df['clean_brand'] = df[target_col].apply(map_brand).astype('category')
    return df

def compile_numeric_display_pipeline(df, target_col='display_size', cache=None):
    num_pattern = re.compile(r'(\d{1,2}(?:\.\d{1,2})?)', re.IGNORECASE)
    
    def extract_float(raw):
        if pd.isna(raw): return np.nan
        clean = str(raw).lower().strip()
        if '*' in clean or 'fhd' == clean or 'full hd' == clean or 'unknown' in clean or 'hd' == clean:
            return np.nan
        match = num_pattern.search(clean)
        if match:
            try:
                val = float(match.group(1))
                if val > 30.0:
                    if 'cm' in clean: val = round(val / 2.54, 1)
                    else: return np.nan
                return val
            except ValueError:
                return np.nan
        return np.nan

    df['clean_display_inches'] = df[target_col].apply(extract_float).astype('float32')
    
    # SZYMANSKI MANIFOLD PROJECTION IMPUTATION
    missing_mask = df['clean_display_inches'].isna()
    if missing_mask.any() and cache is not None:
        def impute_display(row):
            br, ser, cpu, disc = row['clean_brand'], row['clean_model_series'], row['cpu_tier'], row['is_discrete']
            key1 = f"{br}|{ser}"
            if key1 in cache['display_brand_series']: return cache['display_brand_series'][key1]
            key2 = f"{cpu}|{disc}|{br}"
            if key2 in cache['display_cpu_disc_brand']: return cache['display_cpu_disc_brand'][key2]
            return 15.6
        df.loc[missing_mask, 'clean_display_inches'] = df[missing_mask].apply(impute_display, axis=1).astype('float32')

    return df

def resolve_complete_vendor_gpu(df):
    vram_pattern = re.compile(r'(\d+)\s*(?:gb|mb|vram|b\b)', re.IGNORECASE)
    model_patterns = [
        (re.compile(r'\b(rtx\s*50\d{2}|rtx\s*40\d{2}|rtx\s*30\d{2}|rtx\s*20\d{2})\b', re.IGNORECASE), lambda m: m.group(1).upper().replace(" ", "")),
        (re.compile(r'\b(gtx\s*1660|gtx\s*1650|gtx\s*1050|gtx\s*1060)\b', re.IGNORECASE), lambda m: m.group(1).upper().replace(" ", "")),
        (re.compile(r'\b(2050|2060|3050|3060|4050|4060|4070|4080|4090)\b', re.IGNORECASE), lambda m: f"RTX{m.group(1)}"),
        (re.compile(r'\b(1050|1060|1650|1660)\b', re.IGNORECASE), lambda m: f"GTX{m.group(1)}"),
        (re.compile(r'\b(quadro|p\d{4}|t\d{3,4}|m\d{3,4}m)\b', re.IGNORECASE), lambda m: "NVIDIA_QUADRO_PRO"),
        (re.compile(r'\brx\s*([45679]\d{2,3}[a-z]{0,2})\b', re.IGNORECASE), lambda m: f"AMD_RX_{m.group(1).upper()}"),
        (re.compile(r'\b(pro\s*555|pro\s*560x?)\b', re.IGNORECASE), lambda m: f"AMD_RADEON_{m.group(1).upper().replace(' ', '')}"),
        (re.compile(r'\barc\s*([a-z]?\d{3}m?|140v)?\b', re.IGNORECASE), lambda m: f"INTEL_ARC_{m.group(1).upper()}" if m.group(1) else "INTEL_ARC_GENERIC"),
        (re.compile(r'\b(680m|780m|890m)\b', re.IGNORECASE), lambda m: f"AMD_RDNA_{m.group(1).upper()}"),
        (re.compile(r'\b(iris\s*xe|iris\s*plus|iris)\b', re.IGNORECASE), lambda m: "INTEL_IRIS"),
        (re.compile(r'\b(uhd\s?graphics|uhd|hd\s?graphics|620|630)\b', re.IGNORECASE), lambda m: "INTEL_UHD"),
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
        if model == "GENERIC_IGPU":
            if 'nvidia' in scrubbed or 'geforce' in scrubbed or 'mx' in scrubbed: model = "NVIDIA_LEGACY"
            elif 'radeon' in scrubbed or 'amd' in scrubbed or 'vega' in scrubbed: model = "AMD_LEGACY_IGPU"
            elif 'intel' in scrubbed: model = "INTEL_LEGACY_IGPU"
        return f"{model}_{vram}"

    df['gpu_base'] = df['gpu'].apply(extract_base)
    
    is_generic_gpu = df['gpu_base'].str.contains('GENERIC|LEGACY|UNKNOWN', na=True)
    is_gaming_chassis = df['model_series'].astype(str).str.contains('legion|rog|tuf|nitro|predator|alienware|omen|victus|loq|katana|bravo|cyborg|sword', case=False, na=False)
    is_intel_cpu = df['cpu'].astype(str).str.contains('intel|core|i3|i5|i7|i9', case=False, na=False)
    is_amd_cpu = df['cpu'].astype(str).str.contains('amd|ryzen|athlon|a6|a8|a10', case=False, na=False)
    is_qualcomm_cpu = df['cpu'].astype(str).str.contains('snapdragon|qualcomm', case=False, na=False)

    conditions = [
        (is_generic_gpu & is_gaming_chassis),
        (is_generic_gpu & is_intel_cpu),
        (is_generic_gpu & is_amd_cpu),
        (is_generic_gpu & is_qualcomm_cpu)
    ]
    choices = ["INFERRED_ENTRY_DGPU", "INTEL_INFERRED_IGPU_SYSRAM", "AMD_INFERRED_IGPU_SYSRAM", "QUALCOMM_INFERRED_IGPU_SYSRAM"]
    df['clean_gpu_composite'] = pd.Categorical(np.select(conditions, choices, default=df['gpu_base']))
    df = df.drop(columns=['gpu_base'])
    
    def backfill(val):
        s = str(val).upper()
        if s.startswith('INTEL'): return 'INTEL'
        if s.startswith('AMD'): return 'AMD'
        if s.startswith('NVIDIA') or s.startswith('RTX') or s.startswith('GTX'): return 'NVIDIA'
        if s.startswith('APPLE'): return 'APPLE'
        if s.startswith('QUALCOMM'): return 'QUALCOMM'
        return 'UNKNOWN'
        
    df['gpu_vendor'] = df['clean_gpu_composite'].apply(backfill)
    df['clean_gpu_composite'] = df['clean_gpu_composite'].astype('category')
    return df

def map_cpu_tier(cpu_name):
    cpu = str(cpu_name).upper()
    if 'CELERON' in cpu: return 1
    if 'PENTIUM_GOLD' in cpu: return 4
    if 'PENTIUM' in cpu: return 2
    if 'LEGACY' in cpu:
        if any(x in cpu for x in ['A4', 'ATHLON']): return 1
        if any(x in cpu for x in ['A6', 'A8', 'A9']): return 2
        if any(x in cpu for x in ['A10', 'A12']): return 3
        return 1
    if 'SNAPDRAGON' in cpu:
        if 'ELITE' in cpu: return 6
        if 'PLUS' in cpu: return 4
        return 5
    if 'ULTRA_9' in cpu: return 10
    if 'ULTRA_7' in cpu: return 8
    if 'ULTRA_5' in cpu: return 6
    if 'RYZEN_AI9' in cpu: return 10
    if 'RYZEN_AI7' in cpu: return 8
    if 'CORE_I9' in cpu or 'RYZEN_9' in cpu: return 9
    if 'CORE_I7' in cpu or 'RYZEN_7' in cpu: return 7
    if 'CORE_I5' in cpu or 'RYZEN_5' in cpu: return 5
    if 'CORE_I3' in cpu or 'RYZEN_3' in cpu: return 3
    if 'RYZEN_AI5' in cpu or 'ULTRA_3' in cpu: return 4
    return 1

def map_gpu_tier(gpu_name):
    gpu = str(gpu_name).upper()
    if any(x in gpu for x in ['IGPU', 'UHD', 'IRIS']): return 1
    if 'ARC' in gpu and 'GENERIC' in gpu: return 1
    if 'ARC' in gpu and 'GENERIC' not in gpu: return 4
    if 'RDNA' in gpu or 'INFERRED_ENTRY' in gpu: return 2
    if 'RTX4090' in gpu or 'RTX5090' in gpu: return 9
    if any(x in gpu for x in ['RTX4070', 'RTX4080', 'RTX5070', 'RTX5080']): return 8
    if any(x in gpu for x in ['RTX3070', 'RTX3080', 'RTX4060', 'RTX5060', 'RTX2000', 'RTX3000', 'RTX5000']): return 7
    if any(x in gpu for x in ['RTX2060', 'RTX2070', 'RTX3060', 'RTX4050', 'RTX5050', 'RX_6800', 'RX_7600']): return 6
    if any(x in gpu for x in ['RTX2050', 'RTX3050', 'QUADRO']): return 5
    if any(x in gpu for x in ['GTX1650', 'GTX1660', 'RX_560', 'RX_5500', 'RX_6500', 'RX_600M']): return 4
    if any(x in gpu for x in ['GTX1050', 'GTX1060']): return 3
    return 1

_NEGATION_STRIP = re.compile(r"\b(zero|no|free\s+from|without)\s+[\w\s,\/\+]{1,60}?(?=\b|\.|\n)", re.IGNORECASE)
_GAME = re.compile(r"\bextinction\b|\bred\s+dead\s*(redemption\s*\d*)?|\bdead\s+space\b|\bdead\s+island\b", re.IGNORECASE)
_BATTERY = re.compile(
    r"\bbattery\s+([\w\s,\/\-]{1,15}\s+)?(dead|low|issue|backup|faulty|replace|swollen|drain|draining|weak)\b|"
    r"\b(dead|low|faulty|replace|swollen|drain|draining|weak)\s+([\w\s,\/\-]{1,15}\s+)?battery\b|"
    r"\bbattery\s+backup\s+(?:low|weak|bad|poor|reduced|issue[s]?|\d+\s*min(ute)?s?)\b|"
    r"\b(?:low|weak|bad|poor|reduced)\s+battery\s+backup\b", re.IGNORECASE)
_PIXEL = re.compile(r"\bdead\s+pixel[s]?\b|\bblemish\s+on\s+screen\b", re.IGNORECASE)

_START = r"\b(?<!no\s)(?<!zero\s)(?<!without\s)(?<!free\sfrom\s)"
_END = r"\b"
_CRIT = re.compile(_START + r"(dead|fried|burnt|no power|parts only|water damage|cracked screen|as-is|broken display|won't turn on|motherboard issue|shattered|logic board|spilled)" + _END, re.IGNORECASE)
_MAJ = re.compile(_START + r"(lines on|dead key|overheat|faulty|missing key|loud fan|thermal issue|flickering|ghosting|lines|__BATTERY_FAIL__)" + _END, re.IGNORECASE)
_MIN = re.compile(_START + r"(scratched|scratches|dent|dents|scuff|scuffs|wear|blemish|faded|peeling|chipped|crack on case|signs of use|__PIXEL_WARN__)" + _END, re.IGNORECASE)

def calculate_degradation_v6(text):
    if not isinstance(text, str): return 0.0
    clean = text.strip()
    if not clean: return 0.0
    
    sanitized = _NEGATION_STRIP.sub("", clean)
    sanitized = _GAME.sub("__SOFTWARE_TITLE__", sanitized)
    sanitized = _BATTERY.sub("__BATTERY_FAIL__", sanitized)
    sanitized = _PIXEL.sub("__PIXEL_WARN__", sanitized)
    
    w_count = max(sanitized.count(' ') + 1, 10)
    c3 = len(_CRIT.findall(sanitized))
    c2 = len(_MAJ.findall(sanitized))
    c1 = len(_MIN.findall(sanitized))
    
    raw = (c3 * 1.0) + (c2 * 0.5) + (c1 * 0.2)
    if any(p in clean.lower() for p in ["mint condition", "flawless", "showroom condition", "like new"]):
        raw *= 0.5
        
    score = float(raw / np.log(w_count))
    if c3 > 0 and "motherboard" in clean.lower():
        score = max(score, 0.4000)
    return score

# ==========================================
# BOOTSTRAP CACHE LAYER
# ==========================================

def _bootstrap_cache():
    """Builds and serializes zero-dependency execution mapping dicts."""
    if os.path.exists(CACHE_PATH):
        cache = joblib.load(CACHE_PATH)
        # Synthesize Qualcomm proxy encoding if missing
        if 'clean_gpu_composite' in cache['target_encodings'] and 'QUALCOMM_INFERRED_IGPU_SYSRAM' not in cache['target_encodings']['clean_gpu_composite']:
            g_med = cache['global_median_price']
            intel_enc = cache['target_encodings']['clean_gpu_composite'].get('INTEL_INFERRED_IGPU_SYSRAM', g_med)
            amd_enc = cache['target_encodings']['clean_gpu_composite'].get('AMD_INFERRED_IGPU_SYSRAM', g_med)
            cache['target_encodings']['clean_gpu_composite']['QUALCOMM_INFERRED_IGPU_SYSRAM'] = 0.6 * intel_enc + 0.4 * amd_enc
            
        if 'gpu_vendor' in cache['target_encodings'] and 'QUALCOMM' not in cache['target_encodings']['gpu_vendor']:
            g_med = cache['global_median_price']
            intel_v_enc = cache['target_encodings']['gpu_vendor'].get('INTEL', g_med)
            amd_v_enc = cache['target_encodings']['gpu_vendor'].get('AMD', g_med)
            cache['target_encodings']['gpu_vendor']['QUALCOMM'] = 0.6 * intel_v_enc + 0.4 * amd_v_enc
            cache['target_encodings']['gpu_vendor']['APPLE'] = g_med
            import tempfile
            import shutil
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(CACHE_PATH)))
            os.close(fd)
            try:
                joblib.dump(cache, tmp)
                os.replace(tmp, CACHE_PATH)
            except Exception:
                if os.path.exists(tmp): os.unlink(tmp)
                raise
        return cache
        
    print("[BOOTSTRAP] Compiling operational caches from CSV...")
    if not os.path.exists('../data/Kaggle_Master_Engineered_State.csv'):
        raise FileNotFoundError("[FATAL] Missing ../data/Kaggle_Master_Engineered_State.csv to build cache.")
        
    df_raw = pd.read_csv('../data/Kaggle_Master_Engineered_State.csv')
    train_df = df_raw[df_raw['is_global_train'] == True].copy()
    
    cache = {
        'target_encodings': {},
        'display_brand_series': {},
        'display_cpu_disc_brand': {},
        'hardware_knn': {},
        'global_median_price': train_df['price'].median()
    }
    
    # 1. Target Encoding Dicts
    m_smoothing = 10
    g_med = cache['global_median_price']
    
    for col in TARGET_COLS:
        grouped = train_df.groupby(col)['price'].agg(['count', 'median'])
        smoothed = ((grouped['count'] * grouped['median']) + (m_smoothing * g_med)) / (grouped['count'] + m_smoothing)
        cache['target_encodings'][col] = smoothed.to_dict()
        
        if col not in ['city', 'state']:
            hw_knn = train_df.groupby(['ram_size_gb', 'storage_size_gb', 'cpu_tier'])['price'].median().to_dict()
            cache['hardware_knn'][col] = {f"{float(k[0])}|{float(k[1])}|{float(k[2])}": v for k, v in hw_knn.items()}
        
    # 2. Szymanski Display Manifolds
    modes_bs = train_df.dropna(subset=['clean_display_inches']).groupby(['clean_brand', 'clean_model_series'])['clean_display_inches'].apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)
    for (b, s), val in modes_bs.items():
        if not pd.isna(val): cache['display_brand_series'][f"{b}|{s}"] = val
        
    modes_cdb = train_df.dropna(subset=['clean_display_inches']).groupby(['cpu_tier', 'is_discrete', 'clean_brand'])['clean_display_inches'].apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)
    for (c, d, b), val in modes_cdb.items():
        if not pd.isna(val): cache['display_cpu_disc_brand'][f"{c}|{d}|{b}"] = val
        
    # Inject Qualcomm proxy encoding
    if 'clean_gpu_composite' in cache['target_encodings']:
        intel_enc = cache['target_encodings']['clean_gpu_composite'].get('INTEL_INFERRED_IGPU_SYSRAM', g_med)
        amd_enc = cache['target_encodings']['clean_gpu_composite'].get('AMD_INFERRED_IGPU_SYSRAM', g_med)
        cache['target_encodings']['clean_gpu_composite']['QUALCOMM_INFERRED_IGPU_SYSRAM'] = 0.6 * intel_enc + 0.4 * amd_enc
        
    if 'gpu_vendor' in cache['target_encodings']:
        intel_v_enc = cache['target_encodings']['gpu_vendor'].get('INTEL', g_med)
        amd_v_enc = cache['target_encodings']['gpu_vendor'].get('AMD', g_med)
        cache['target_encodings']['gpu_vendor']['QUALCOMM'] = 0.6 * intel_v_enc + 0.4 * amd_v_enc
        cache['target_encodings']['gpu_vendor']['APPLE'] = g_med

    import tempfile
    import shutil
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(CACHE_PATH)))
    os.close(fd)
    try:
        joblib.dump(cache, tmp)
        os.replace(tmp, CACHE_PATH)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print(f"[BOOTSTRAP] Serialized cache to {CACHE_PATH}. CSV no longer needed.")
    return cache

# ==========================================
# ==========================================
# MASTER EXECUTION PIPELINE
# ==========================================

def enforce_strict_topology_simd(df):
    """
    SIMD-accelerated topological exclusion using PyArrow.
    Drops anomalies like keyword stuffing, unclassified hardware, and out-of-bound manifolds.
    """
    if len(df) == 0:
        return df

    # Desktop & Unclassified Silicon Ejection
    invalid_cpus = df['cpu_composite_clean'].isin(['UNCLASSIFIED_HARDWARE_CORE', 'AMD_RYZEN_GENERIC_UNKNOWN'])
    invalid_chassis = df['model_series'].astype(str).str.contains('Assembled', case=False, na=False)
    
    # Display Manifold Bounds
    invalid_display = (df['clean_display_inches'] < 10.0) | (df['clean_display_inches'] > 18.5)
    
    # Bitwise Aggregation
    drop_mask = np.array(invalid_cpus) | np.array(invalid_chassis) | np.array(invalid_display)
    
    # PyArrow SIMD Text Evaluation for keyword anomalies
    if 'title' in df.columns:
        # Prevent conversion errors with explicit string casting
        title_arr = pc.fill_null(pa.array(df['title'], type=pa.string()), "")
        
        # Intel 13th/14th Gen Ejection (Junk Silicon)
        junk_intel = pc.match_substring_regex(title_arr, r'(?i)\b(13th|14th|i[3579][- ]?1[34]\d{3}[a-z]*)\b')
        drop_mask = drop_mask | np.array(junk_intel)
        
        # Catch 13/14th Gen that were only extracted by LLM (not in title)
        junk_intel_cpu_col = df['cpu'].astype(str).str.contains(r'(?i)\b(13th|14th|i[3579][- ]?1[34]\d{3}[a-z]*)\b', regex=True, na=False)
        junk_intel_composite = df['cpu_composite_clean'].astype(str).str.contains(r'INTEL_CORE_I[3579]_SKU1[34]', regex=True, na=False)
        drop_mask = drop_mask | np.array(junk_intel_cpu_col) | np.array(junk_intel_composite)
        
        intel_mask = pc.match_substring_regex(title_arr, r'(?i)\b(intel|i3|i5|i7|i9|core i)\b')
        amd_mask = pc.match_substring_regex(title_arr, r'(?i)\b(amd|ryzen|athlon)\b')
        cpu_conflict = pc.and_(intel_mask, amd_mask)
        drop_mask = drop_mask | np.array(cpu_conflict)
        
        # Keyword multiple brands
        found_brands = np.zeros(len(df), dtype=int)
        for brand in ['hp', 'dell', 'lenovo', 'acer', 'asus', 'apple', 'macbook', 'msi', 'razer', 'microsoft', 'surface', 'samsung', 'alienware']:
            b_mask = pc.match_substring_regex(title_arr, rf'(?i)\b{brand}\b')
            found_brands += np.array(b_mask).astype(int)
        brand_conflict = found_brands >= 2
        drop_mask = drop_mask | brand_conflict
        
    if 'title' in df.columns and 'description' in df.columns:
        pass # V3 SSD string check removed: relying on LLM to extract storage correctly instead.

    # Return sanitized dataframe
    return df[~drop_mask].copy()

def predict(output_csv="../predict/Arbitrage_Opportunities.csv", input_df=None, margin=0.0, price_min=0.0):
    print("=== Starting arbitrage engine initialization ===")
    cache = _bootstrap_cache()
    
    import threading
    global _MODEL_LOCK, _SCALER, _GMM, _MOE_PAYLOAD, _GLOBAL_MODEL, _AUTOENCODER, _DEVICE
    if '_SCALER' not in globals():
        with _MODEL_LOCK:
            if '_SCALER' not in globals():
                print("-> Loading Neural Architectures and Stacks...")
                _SCALER = joblib.load(SCALER_PATH)
                _GMM = joblib.load(GMM_PATH)
                _MOE_PAYLOAD = joblib.load(ENGINE_PATH)
                _GLOBAL_MODEL = joblib.load('../models/global_stack_model.joblib')
                _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _AUTOENCODER = LaptopAutoencoder(input_dim=8, latent_dim=8).to(_DEVICE)
                _AUTOENCODER.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=_DEVICE, weights_only=True))
                _AUTOENCODER.eval()
    
    scaler, gmm, moe_payload = _SCALER, _GMM, _MOE_PAYLOAD
    global_model, autoencoder, device = _GLOBAL_MODEL, _AUTOENCODER, _DEVICE
    
    if input_df is not None:
        df = input_df.copy()
    else:
        df = data_supplier.get_laptops()
    if df is None: return pd.DataFrame()
    
    # Structural sanitation
    df = df.replace(["unknown", "Unknown", "", " ", "-", "N/A", "NA", "n/a", "na", "null", "Null", "none", "None"], np.nan)
    df = enforce_bounds(df, price_min=price_min)
    
    if len(df) == 0:
        print("[WARNING] Zero records passed bounds enforcement.")
        df['Predicted_Price_INR'] = np.nan
        df['unseen_error'] = ""
        return df
        
    # Taxonomy features
    if 'model_series' in df.columns:
        df = compile_hardware_pipeline(df)
        df = extract_gpu_topology_matrix(df)
        df = compile_heterogeneous_cpu_pipeline(df)
        df = compile_brand_normalization_pipeline(df)
        
        df['cpu_tier'] = df['cpu_composite_clean'].apply(map_cpu_tier)
        df = resolve_complete_vendor_gpu(df)
        df['gpu_tier'] = df['clean_gpu_composite'].apply(map_gpu_tier)
        
        df = compile_numeric_display_pipeline(df, cache=cache)
        
        # [HOOK] SIMD-Accelerated Topological Ejection
        df = enforce_strict_topology_simd(df)
        
        if len(df) == 0:
            print("[WARNING] Zero records passed topological sanitization.")
            df['Predicted_Price_INR'] = np.nan
            df['unseen_error'] = ""
            return df
    
    binary_cols = ['fingerprint', 'touch', 'warranty', 'urgent_sell']
    for col in binary_cols:
        if col in df.columns:
            df[col] = np.where(df[col].fillna('no').astype(str).str.lower().str.strip() == 'yes', 1, 0)
        else:
            df[col] = 0
        
    numeric_cols = ['ram_size_gb', 'storage_size_gb', 'clean_display_inches']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    if 'listing_age_days' not in df.columns:
        if 'created_at' in df.columns:
            def calc_age(x):
                if pd.isna(x): return 0.0
                try:
                    dt = datetime.fromisoformat(str(x))
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
                    return float(max(0, (now - dt).days))
                except ValueError:
                    return 0.0
            df['listing_age_days'] = df['created_at'].apply(calc_age)
        elif 'created_at_first' in df.columns:
            def calc_age(x):
                if pd.isna(x): return 0.0
                try:
                    dt = datetime.fromisoformat(str(x))
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
                    return float(max(0, (now - dt).days))
                except ValueError:
                    return 0.0
            df['listing_age_days'] = df['created_at_first'].apply(calc_age)
        else:
            df['listing_age_days'] = 0.0
            
    if 'condition_degradation_index' not in df.columns:
        if 'description' in df.columns:
            df['condition_degradation_index'] = df['description'].apply(calculate_degradation_v6)
        else:
            df['condition_degradation_index'] = 0.0
    
    # Target Encoding Injection
    df['unseen_error'] = ""
    for col in TARGET_COLS:
        mapping = cache['target_encodings'][col]
        global_med = cache['global_median_price']
        
        # Map known target encodings directly via hash map
        df[f"{col}_target_enc"] = df[col].map(mapping).astype(float)
        
        # Hierarchical Topological Fallback
        if col == 'city':
            # Resolve unseen cities to their parent state encoding instead of generic global median
            df[f"{col}_target_enc"] = df[f"{col}_target_enc"].fillna(df['state_target_enc'])
        else:
            # Semantic Spec-Based K-NN Fallback (Solution 4)
            # Impute missing encodings based on hardware signature (RAM, Storage, CPU Tier)
            if 'cpu_tier' in df.columns and 'ram_size_gb' in df.columns and 'storage_size_gb' in df.columns and col in cache['hardware_knn']:
                # Vectorized dictionary lookup instead of N+1 apply
                keys = df['ram_size_gb'].astype(float).astype(str) + "|" + df['storage_size_gb'].astype(float).astype(str) + "|" + df['cpu_tier'].astype(float).astype(str)
                hardware_signature_median = keys.map(cache['hardware_knn'][col])
                df[f"{col}_target_enc"] = df[f"{col}_target_enc"].fillna(hardware_signature_median)
        
        # Flag catastrophic unseen categories that bypass fallbacks
        fatal_unseen = df[f"{col}_target_enc"].isna() & df[col].notna()
        if fatal_unseen.any():
            df.loc[fatal_unseen, 'unseen_error'] += "not enough data on " + col + ": " + df.loc[fatal_unseen, col].astype(str) + " | "
            
        # Final execution layer: zero-knowledge imputation to global median
        df[f"{col}_target_enc"] = df[f"{col}_target_enc"].fillna(global_med)

        
    # Extract tensor components
    features_to_encode = [
        'ram_size_gb', 'storage_size_gb', 'clean_display_inches',
        'condition_degradation_index', 'cpu_tier', 'gpu_tier', 
        'is_discrete', 'listing_age_days'
    ]
    
    X_enc = df[features_to_encode].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X_enc)
    
    with torch.no_grad():
        tensor_X = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        _, Z = autoencoder(tensor_X)
    Z_matrix = Z.cpu().numpy()
    
    probs = gmm.predict_proba(Z_matrix)
    max_probs = np.max(probs, axis=1)
    assignments = np.argmax(probs, axis=1)
    df['expert_model_route'] = np.where(max_probs >= 0.85, assignments, -1)
    
    # Inference Passes
    # Safely impute missing features dynamically to prevent KeyErrors
    for feature in EXPLICIT_FEATURES:
        if feature not in df.columns:
            df[feature] = 0.0
    X_explicit = df[EXPLICIT_FEATURES].to_numpy(dtype=np.float32)
    
    preds_global = global_model.predict(X_explicit)
    
    preds_moe = np.zeros(len(df))
    routes = df['expert_model_route'].dropna().unique()
    for route in routes:
        route = int(route)
        mask = (df['expert_model_route'] == route).to_numpy()
        if route in moe_payload['moe_stacks']:
            stack = moe_payload['moe_stacks'][route]
            if mask.any():
                preds_moe[mask] = stack.predict(X_explicit[mask])
        else:
            preds_moe[mask] = preds_global[mask]
    
    w_global = moe_payload['blend_weights']['global_weight']
    w_moe = moe_payload['blend_weights']['moe_weight']
    
    pred_blend = (w_global * np.expm1(preds_global)) + (w_moe * np.expm1(preds_moe))
    final_price = pred_blend * moe_payload['arbitrage_scalar']
    
    df['Predicted_Price_INR'] = final_price.astype(float)
    
    # error_mask = df['unseen_error'] != ""
    # df.loc[error_mask, 'Predicted_Price_INR'] = np.nan
    
    print("\n=== PREDICTION RESULTS ===")
    print(df[['clean_brand', 'clean_model_series', 'cpu_composite_clean', 'clean_gpu_composite', 'ram_size_gb', 'Predicted_Price_INR']])
    
    # Save directly from within predict() if being called externally
    if input_df is not None and 'price' in df.columns:
        df['arbitrage_margin_inr'] = df['Predicted_Price_INR'] - df['price']
        
        multiplier = 1.0 + (margin / 100.0)
        negotiable_mask = df['price'] <= (df['Predicted_Price_INR'] * multiplier)
        
        opportunities = df[negotiable_mask].copy()
        opportunities = opportunities.sort_values('arbitrage_margin_inr', ascending=False)
        
        error_rows = df[(df['unseen_error'] != "") & (~negotiable_mask)].copy()
        final_export = pd.concat([opportunities, error_rows], ignore_index=True)
        
        if os.path.dirname(output_csv):
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        final_export.to_csv(output_csv, index=False)
        print(f"[ARBITRAGE] Extracted {len(opportunities)} profitable opportunities and {len(error_rows)} rejected unseen exceptions.")
        print(f"[ARBITRAGE] Saved to {output_csv}")
        
    return df

if __name__ == "__main__":
    df_result = predict()
    
    if 'price' in df_result.columns:
        df_result['arbitrage_margin_inr'] = df_result['Predicted_Price_INR'] - df_result['price']
        
        margin = 0.0
        multiplier = 1.0 + (margin / 100.0)
        negotiable_mask = df_result['price'] <= (df_result['Predicted_Price_INR'] * multiplier)
        
        opportunities = df_result[negotiable_mask].copy()
        opportunities = opportunities.sort_values('arbitrage_margin_inr', ascending=False)
        
        error_rows = df_result[(df_result['unseen_error'] != "") & (~negotiable_mask)].copy()
        
        final_export = pd.concat([opportunities, error_rows], ignore_index=True)
        
        import os
        os.makedirs("predict", exist_ok=True)
        final_export_path = os.path.join("predict", "Arbitrage_Opportunities.csv")
        final_export.to_csv(final_export_path, index=False)
        print(f"\n[ARBITRAGE] Extracted {len(opportunities)} profitable opportunities and {len(error_rows)} rejected unseen exceptions.")
        print(f"[ARBITRAGE] Saved to {final_export_path}")
    else:
        print("\n[ARBITRAGE] 'price' column not found in data feed. Cannot calculate arbitrage.")


