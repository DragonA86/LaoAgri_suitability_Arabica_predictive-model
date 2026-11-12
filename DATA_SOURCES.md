# Data and methodological sources

## Administrative boundary / area of interest

- **FAO Global Administrative Unit Layers (GAUL), 2024 release** — used to define the Lao PDR study-area boundary for Dataset V2. It is a boundary source, not an Arabica crop-requirement source and does not prescribe score weights.
  - https://www.fao.org/agroinformatics/training-and-resources/data-sets/data-set-detail/global-gaul-new-2024-release/en

## Environmental predictors used in Dataset V2

- **Copernicus DEM GLO-30 2024_1** — elevation and terrain derivatives. The product is a Digital Surface Model (DSM), not a bare-earth DTM.
  - https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30_2024_1
- **WorldClim 2.1** — BIO1, BIO12, BIO14, and BIO15 at 30 arc-seconds; historical climate normal baseline 1970–2000.
  - https://www.worldclim.org/data/worldclim21.html
- **SoilGrids 2.0** — soil pH and soil organic carbon (SOC) at 250 m native product resolution before aggregation to the project grid.
  - https://docs.isric.org/globaldata/soilgrids/
  - https://docs.isric.org/globaldata/soilgrids/access_on_gee.html

## Agronomic / methodological references

- **FAO. Arabica Coffee Manual for Lao PDR**, especially Chapter 1 (plant/site selection) and Chapter 6 (soil nutrient levels). This is the principal local Arabica agronomy reference used to inform V3 score-curve anchors.
  - https://www.fao.org/4/ae939e/ae939e03.htm
  - https://www.fao.org/4/ae939e/ae939e06.htm
- **FAO EcoCrop — Coffea arabica** — supporting crop-environment reference.
  - https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=749
- **Grüter et al. (2022), PLOS ONE 17(1), e0261976** — a literature source consulted during the project's suitability-criteria review.
  - https://doi.org/10.1371/journal.pone.0261976
- **Ovalle-Rivera et al. (2015), PLOS ONE 10(4), e0124155** — a global Arabica climate-suitability reference consulted during the project's evidence review.
  - https://doi.org/10.1371/journal.pone.0124155

## Weighting and scoring provenance

FAO GAUL 2024 does not provide Arabica suitability weights. The weights and several interpolation points in V3 are project-level modeling assumptions informed by the local Arabica manual and supporting literature. The app documents these as assumptions and retains sensitivity analyses in the V3 training notebook. They require independent field/expert validation before operational land-allocation use.

## Data reuse

The code license does not automatically apply to the input rasters, derived CSV, or any third-party source datasets. Check each source's terms before redistributing data outside this private project. The raw Dataset V2 CSV is included in the ZIP for local reproducibility but excluded by `.gitignore` to reduce accidental publication.
