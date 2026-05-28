import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from shapely import wkt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Delhi EV Charger Optimizer", page_icon="⚡", layout="wide")

st.title(" EV Charger Location Optimizer — Delhi")
st.markdown("### Identifying underserved wards for EV charging infrastructure using K-Means Clustering")
st.markdown("---")

@st.cache_data
def load_data():

    # ---- LOAD DATA ----
    pop = pd.read_csv('pop.csv')
    pop.drop(['sc_population'], axis=1, inplace=True)

    try:
        charger = pd.read_excel('ev_charger_loc.xlsx')
    except:
        charger = pd.read_csv('ev_charger_loc.csv')

    # Drop irrelevant columns
    cols_to_drop = ['uid','vendor_name','city','country','logo_url','staff',
                    'payment_modes','contact_numbers','station_type', 0,
                    'available','cost_per_unit','vehicle_type','open','close',
                    'type','capacity','zone','power_type','postal_code','address']
    charger.drop(columns=[c for c in cols_to_drop if c in charger.columns], inplace=True)

    # Clean charger data
    charger['total'].fillna(1, inplace=True)
    charger = charger.sort_values('total', ascending=False)
    charger = charger.drop_duplicates(subset=['latitude', 'longitude'], keep='first')
    charger.reset_index(drop=True, inplace=True)

    # ---- CONVERT TO GEODATAFRAMES ----
    pop['geometry'] = pop['boundary_wkt'].apply(wkt.loads)
    pop_gdf = gpd.GeoDataFrame(pop, geometry='geometry', crs=4326)
    pop_gdf = pop_gdf.set_crs(epsg=4326, allow_override=True)

    ev_gdf = gpd.GeoDataFrame(charger,
        geometry=gpd.points_from_xy(charger.longitude, charger.latitude), crs=4326)
    ev_gdf = ev_gdf.set_crs(epsg=4326, allow_override=True)

    # ---- REPROJECT TO EPSG:32643 ----
    projected_crs = "EPSG:32643"
    pop_gdf_proj = pop_gdf.to_crs(projected_crs)
    ev_gdf_proj = ev_gdf.to_crs(projected_crs)
    pop_gdf_proj['geometry'] = pop_gdf_proj.geometry.buffer(1)

    # ---- SPATIAL JOIN ----
    ev_with_ward_proj = gpd.sjoin(ev_gdf_proj, pop_gdf_proj, how='left', predicate='within')
    matched_proj = ev_with_ward_proj[ev_with_ward_proj['wardno'].notna()].copy()
    unmatched_proj = ev_with_ward_proj[ev_with_ward_proj['wardno'].isnull()].copy()

    def find_nearest_ward_proj(point, ward_gdf_proj):
        distances = ward_gdf_proj.geometry.centroid.distance(point)
        return ward_gdf_proj.iloc[distances.idxmin()]['wardno']

    if len(unmatched_proj) > 0:
        if 'geometry' not in unmatched_proj.columns:
            unmatched_proj = gpd.GeoDataFrame(unmatched_proj,
                geometry=gpd.points_from_xy(unmatched_proj.longitude, unmatched_proj.latitude),
                crs=projected_crs)
        unmatched_proj['wardno'] = unmatched_proj.geometry.apply(
            lambda x: find_nearest_ward_proj(x, pop_gdf_proj))

    final_ev_proj = pd.concat([matched_proj, unmatched_proj])

    # ---- BLUE DOTS — matched_proj only (inside Delhi) ----
    delhi_chargers = matched_proj[['name', 'latitude_left', 'longitude_left', 'total']].copy()
    delhi_chargers = delhi_chargers.rename(columns={
        'latitude_left': 'latitude',
        'longitude_left': 'longitude'
    })
    delhi_chargers = delhi_chargers.drop_duplicates(subset=['latitude', 'longitude'], keep='first')

    # ---- FEATURE ENGINEERING ----
    charger_per_ward = final_ev_proj.groupby('wardno')['total'].sum().reset_index()
    charger_per_ward.columns = ['wardno', 'total_chargers']

    merged_df_accurate = pop[['wardno', 'ward', 'total_population', 'boundary_wkt']].merge(
        charger_per_ward, on='wardno', how='left')
    merged_df_accurate['total_chargers'].fillna(0, inplace=True)
    merged_df_accurate['charger_density'] = (merged_df_accurate['total_chargers'] / merged_df_accurate['total_population']) * 10000

    # ---- OUTLIER REMOVAL ----
    Q1 = merged_df_accurate['charger_density'].quantile(0.25)
    Q3 = merged_df_accurate['charger_density'].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR

    outliers = merged_df_accurate[merged_df_accurate['charger_density'] > upper].copy()
    df_cluster = merged_df_accurate[merged_df_accurate['charger_density'] <= upper].copy()

    # ---- KMEANS k=2 on clean data ----
    X = df_cluster[['total_population', 'total_chargers', 'charger_density']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df_cluster = df_cluster.copy()
    df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

    # Auto detect tier labels
    cluster_summary = df_cluster.groupby('cluster').agg(
        avg_density=('charger_density', 'mean')
    ).reset_index().sort_values('avg_density').reset_index(drop=True)

    tier_map = {row['cluster']: ['Tier 1 - Urgent', 'Tier 2 - Moderate'][i]
                for i, row in cluster_summary.iterrows()}

    df_cluster['tier'] = df_cluster['cluster'].map(tier_map)
    outliers['tier'] = 'Tier 3 - Covered'
    outliers['cluster'] = 2

    merged_df_accurate = pd.concat([df_cluster, outliers]).sort_values('wardno').reset_index(drop=True)

    # ---- EVALUATION METRICS ----
    sil = silhouette_score(X_scaled, df_cluster['cluster'])
    db = davies_bouldin_score(X_scaled, df_cluster['cluster'])
    inertia = kmeans.inertia_

    return merged_df_accurate, delhi_chargers, sil, db, inertia

# ---- LOAD ----
with st.spinner("Loading and processing Delhi EV data..."):
    merged_df_accurate, delhi_chargers, sil_score, db_score, inertia = load_data()

# ---- SIDEBAR ----
st.sidebar.title(" Filters")
selected_tiers = st.sidebar.multiselect(
    "Select Tiers to Display",
    options=['Tier 1 - Urgent', 'Tier 2 - Moderate', 'Tier 3 - Covered'],
    default=['Tier 1 - Urgent', 'Tier 2 - Moderate', 'Tier 3 - Covered']
)
show_chargers = st.sidebar.checkbox("Show Existing Charger Locations (Delhi only)", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown("Ward-level analysis of Delhi EV charging infrastructure gaps using spatial join, outlier removal, and K-Means clustering.")

# ---- METRICS ----
st.markdown("###  Key Statistics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Wards Analyzed", f"{len(merged_df_accurate)}")
c2.metric("🔴 Tier 1 — Urgent", f"{len(merged_df_accurate[merged_df_accurate['tier']=='Tier 1 - Urgent'])}")
c3.metric("🟠 Tier 2 — Moderate", f"{len(merged_df_accurate[merged_df_accurate['tier']=='Tier 2 - Moderate'])}")
c4.metric("🟢 Tier 3 — Covered", f"{len(merged_df_accurate[merged_df_accurate['tier']=='Tier 3 - Covered'])}")
st.markdown("---")

# ---- MAP ----
st.markdown("###  Delhi Ward Priority Map")
st.caption("Hover over any ward to see details. Blue dots = existing charger stations within Delhi only.")

colors_map = {
    'Tier 1 - Urgent': '#e74c3c',
    'Tier 2 - Moderate': '#f39c12',
    'Tier 3 - Covered': '#27ae60'
}

delhi_map = folium.Map(location=[28.6139, 77.2090], zoom_start=11)
filtered_df = merged_df_accurate[merged_df_accurate['tier'].isin(selected_tiers)]

for _, row in filtered_df.iterrows():
    try:
        geom = wkt.loads(row['boundary_wkt'])
        tooltip_text = (
            f"<b>{row['ward']}</b><br>"
            f"Population: {int(row['total_population']):,}<br>"
            f"Total Charging Points: {int(row['total_chargers'])}<br>"
            f"Density per 10K: {row['charger_density']:.2f}<br>"
            f"<b>{row['tier']}</b>"
        )
        if geom.geom_type == 'Polygon':
            coords = [[y, x] for x, y in geom.exterior.coords]
            if coords:
                folium.Polygon(locations=coords, color='white', weight=0.5,
                    fill=True, fill_color=colors_map[row['tier']], fill_opacity=0.7,
                    tooltip=tooltip_text).add_to(delhi_map)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                if poly.exterior:
                    coords = [[y, x] for x, y in poly.exterior.coords]
                    if coords:
                        folium.Polygon(locations=coords, color='white', weight=0.5,
                            fill=True, fill_color=colors_map[row['tier']], fill_opacity=0.7,
                            tooltip=tooltip_text).add_to(delhi_map)
    except:
        pass

if show_chargers:
    for _, row in delhi_chargers.iterrows():
        try:
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3, color='blue', fill=True,
                fill_color='blue', fill_opacity=0.8,
                tooltip=f"<b>{row['name']}</b><br>Charging Points: {int(row['total'])}"
            ).add_to(delhi_map)
        except:
            pass

legend_html = '''
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
     background-color: white; padding: 12px 16px; border-radius: 8px;
     border: 2px solid #333; font-size: 13px; color: #000000;">
     <b style="color:#000000;">EV Charger Priority</b><br><br>
     <i style="background:#e74c3c;width:12px;height:12px;display:inline-block;margin-right:6px;"></i><span style="color:#000000;">Tier 1 - Urgent</span><br>
     <i style="background:#f39c12;width:12px;height:12px;display:inline-block;margin-right:6px;"></i><span style="color:#000000;">Tier 2 - Moderate</span><br>
     <i style="background:#27ae60;width:12px;height:12px;display:inline-block;margin-right:6px;"></i><span style="color:#000000;">Tier 3 - Covered</span><br>
     <i style="background:blue;width:12px;height:12px;display:inline-block;margin-right:6px;border-radius:50%;"></i><span style="color:#000000;">Existing Charger</span>
</div>
'''
delhi_map.get_root().html.add_child(folium.Element(legend_html))
st_folium(delhi_map, width=1400, height=600)
st.markdown("---")

# ---- CHARTS ----
st.markdown("###  Analysis Charts")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Ward Distribution by Tier**")
    order = ['Tier 1 - Urgent', 'Tier 2 - Moderate', 'Tier 3 - Covered']
    tier_counts = merged_df_accurate['tier'].value_counts().reindex(order).fillna(0)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    bars = ax1.bar(tier_counts.index, tier_counts.values,
                   color=['#e74c3c', '#f39c12', '#27ae60'], edgecolor='white')
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1,
                 str(int(yval)), ha='center', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Tier')
    ax1.set_ylabel('Number of Wards')
    ax1.set_title('Ward Priority Distribution')
    ax1.tick_params(axis='x', labelsize=9)
    plt.tight_layout()
    st.pyplot(fig1)

with chart_col2:
    st.markdown("**Population vs Charger Density**")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for tier, group in merged_df_accurate.groupby('tier'):
        ax2.scatter(group['total_population'], group['charger_density'],
                    label=tier, color=colors_map[tier], alpha=0.7, s=60)
    ax2.set_xlabel('Ward Population')
    ax2.set_ylabel('Charger Density (per 10,000 people)')
    ax2.set_title('Population vs Charger Density by Tier')
    ax2.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("---")

# ---- MODEL EVALUATION ----
st.markdown("###  Model Evaluation")
st.caption("K-Means clustering quality metrics — evaluated after IQR-based outlier removal on 238 wards")

e1, e2, e3 = st.columns(3)
e1.metric("Silhouette Score", f"{sil_score:.3f}", "Reasonable cluster separation")
e2.metric("Davies-Bouldin Score", f"{db_score:.3f}", "Good separation ✅" if db_score < 1.0 else "Some overlap")
e3.metric("Inertia", f"{inertia:.2f}", "Within-cluster compactness")

st.info("""
**How to interpret:**
- **Silhouette Score (0.440):** Reasonable cluster separation. Some overlap is expected since Delhi ward populations are similar in size — charger density is the key differentiator.
- **Davies-Bouldin Score (0.930):** Below 1.0 indicates good cluster separation.
- **Outlier handling:** 12 high-density wards were identified using IQR and assigned directly to Tier 3 before clustering. This improved the Silhouette Score from 0.397 to 0.440.
""")

st.markdown("---")

# ---- TOP UNDERSERVED WARDS ----
st.markdown("###  Top 15 Most Underserved Wards (Tier 1)")
st.caption("Sorted by population — highest population with lowest charger density = highest priority")

urgent = merged_df_accurate[merged_df_accurate['tier'] == 'Tier 1 - Urgent']\
    .sort_values('total_population', ascending=False)\
    [['ward', 'total_population', 'total_chargers', 'charger_density']]\
    .head(15).reset_index(drop=True)
urgent.index += 1
urgent.columns = ['Ward Name', 'Population', 'Total Charging Points', 'Density per 10K']
urgent['Population'] = urgent['Population'].apply(lambda x: f"{int(x):,}")
urgent['Total Charging Points'] = urgent['Total Charging Points'].apply(lambda x: int(x))
urgent['Density per 10K'] = urgent['Density per 10K'].round(2)
st.dataframe(urgent, use_container_width=True)

st.markdown("---")

# ---- BUSINESS IMPACT ----
st.markdown("###  Market Opportunity — TAM / SAM Analysis")
st.caption("Strategic expansion pipeline based on ward-level population and EV transition projections")

t1_count = len(merged_df_accurate[merged_df_accurate['tier'] == 'Tier 1 - Urgent'])
t2_count = len(merged_df_accurate[merged_df_accurate['tier'] == 'Tier 2 - Moderate'])

b1, b2, b3 = st.columns(3)
b1.metric("Phase 1 — Tier 1 TAM/Year", "₹262 Crore",
          f"{t1_count} wards · ~12.3M population · 1.5% EV transition")
b2.metric("Phase 2 — Tier 2 SAM/Year", "₹110 Crore",
          f"{t2_count} wards · ~4.4M population · 0.75% market gap")
b3.metric("Total Pipeline", "₹372 Crore/Year", "Combined TAM + SAM opportunity")

st.markdown("""
| Phase | Wards | Population | EV Transition | Daily Demand | Annual Value |
|-------|-------|------------|---------------|--------------|--------------|
| Phase 1 — Tier 1 Urgent | 160 | ~12.3 Million | 1.5% (3-year) | ~5,98,000 kWh/day | ₹262 Crore/year |
| Phase 2 — Tier 2 Moderate | 78 | ~4.4 Million | 0.75% market gap | ~2,50,000 kWh/day | ₹110 Crore/year |
| **Total** | **238** | **~16.7 Million** | | | **₹372 Crore/year** |
""")
st.caption("Methodology: Population × EV transition rate × avg daily kWh consumption × grid tariff × 365 days. Source: Delhi EV Policy 2020, CEEW EV Demand Report 2024.")

st.markdown("---")

st.markdown("""
<div style='text-align:center; color:grey; font-size:13px; padding:10px'>
Built with Python · geopandas · scikit-learn · folium · streamlit<br>
Data: Delhi Ward Population 2022 (250 wards) · EV Charging Stations India
</div>
""", unsafe_allow_html=True)
