# Delhi_Ev_Charger_Optimizer
A two-stage hybrid Geospatial ML pipeline, mapping a ₹350+ Crore Total Addressable Market (TAM) across Delhi's 250 municipal wards to optimize EV charging infrastructure.

## EV Charger Location Optimizer — Delhi

The Story Behind This Project
I own an Ather 450. A few months ago, I noticed a new charging grid installed in my area. My first thought was — how did they decide to put it here?
Being a data science student who likes finding problems and building solutions, I started observing. Over the past year the number of Ather owners in my area had visibly increased. That got me thinking — are EV companies making data-driven decisions about where to place chargers, or are they just guessing?
Then I looked at the bigger picture. There is a real shift happening right now. The Middle-East Tension pushed global fuel prices up. The Indian government is aggressively pushing EV incentives under FAME II. Petrol prices are not coming down. People are switching. EV registrations in Delhi alone jumped 30% in 2024.
The demand is coming. The question is — is the infrastructure ready?
I built this project to answer that question.

------------------------------------------------- 

# What This Project Does
A ward-level analysis of Delhi's EV charging infrastructure that identifies which of Delhi's 250 wards are most underserved — and ranks them by priority for new charger installation.

# The model combines:

* Delhi ward population data (2022 delimitation — 250 wards)
* Existing EV charging station locations across Delhi
* Spatial join to map chargers to wards
* K-Means clustering to segment wards into priority tiers

# Key Finding:
Out of 250 Delhi wards:

🔴 160 wards are Tier 1 Urgent — high population, critically low charger density

🟠 78 wards are Tier 2 Moderate — some infrastructure but still underserved

🟢 12 wards are Tier 3 Covered — adequate infrastructure


96% of Delhi wards need more EV charging infrastructure.
----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Market Opportunity

To translate these clusters into a commercial business case, I built a financial model tracking segmented public grid dependency. Instead of applying a flat    
average consumption rate across all vehicles, the model implements custom user profiles:

Private EV Owners (75% Mix): Rely heavily on cheaper home charging. Modeled at a minor 1.5 kWh/day public grid dependency.

Commercial Fleets / Gig Workers (25% Mix): Rely entirely on public fast hubs for daily uptime. Modeled at a heavy 12.0 kWh/day consumption.

Weighted Average Daily Grid Load : 4.125 kWh / vehicle / day

# Target Horizon | Focus Area           | Vehicle Pool Proxy    | Daily Grid Demand | Annual Market Value
  _______________________________________________________________________________________________________
  Phase 1:TAM    | 160 Urgent Wards     | 1.50% Transition Pool | 655,920 kWh/day   | ₹287.3 Crore / Year
  _______________________________________________________________________________________________________
  Phase 2: SAM,  |  78 Moderate Wards   | 0.75% Expansion Pool  | 157,827 kWh/day   | ₹69.1 Crore / Year
  _______________________________________________________________________________________________________
  TOTAL          | 238 Actionable Zones | Combined Target       | 813,747 kWh/day   | ₹356.4 Crore / Year


# Live Demo
👉 Open the interactive Delhi EV Charger Map
 
# The app shows:

* Color coded Delhi ward map by priority tier
* Hover tooltips showing ward name, population, charger count and density
* Existing charger locations as blue dots
* Top 15 most underserved wards table
* Model evaluation metrics


# Methodology
Data Sources:

* Delhi Ward Population 2022 — Government open data (250 wards)
* EV Charging Stations India — Kaggle

# Pipeline:

* Loaded ward boundary polygons and EV charger coordinates
* Reprojected to EPSG:32643 for accurate spatial operations
* Spatial join — assigned each charger to its ward using polygon boundaries
* Nearest ward fallback for chargers on ward boundaries
* Engineered charger density feature (charging points per 10,000 people)
* IQR-based outlier removal — identified 12 high-density wards as Tier 3 directly
* K-Means clustering (k=2) on remaining 238 wards to identify Tier 1 and Tier 2

# Model Evaluation Metrics
Silhouette Score: 0.440 (Confirms robust, reasonable cluster definitions in dense geospatial boundaries)

Davies-Bouldin Score: 0.930 (Proves strong, definitive separation between cluster centers)

# Tech Stack
Python · pandas · geopandas · shapely · scikit-learn · folium · streamlit

# Version 1 — Current Scope
This is Version 1 of the project. The current model uses 3 features for placement decisions:
 * Ward population
 * Existing charger count
 * Charger density per 10,000 people

# Version 2 is in progress. Planned improvements include:

* Metro station proximity — wards near metro have higher commuter EV usage
* Mall and commercial hub proximity — high dwell time = more charging opportunity
* Road network and traffic density analysis
* Income level data by ward — correlates with EV adoption rate
* Extending the model to other Indian cities

I am actively working on Version 2 and will publish it soon.

## About
Built by a data science student and EV owner who got curious about how charging infrastructure decisions are made — and decided to build a model to make those decisions better.
If you are working on EV infrastructure, urban mobility, or smart city projects and find this useful — feel free to reach out.
