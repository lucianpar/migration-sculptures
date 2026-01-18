/**
 * UI Manager
 *
 * Handles user interface interactions and info panel updates.
 */

export class UIManager {
  constructor(sculptureManager) {
    this.sculptureManager = sculptureManager;

    // DOM elements
    this.elements = {
      infoPanel: document.getElementById("info-panel"),
      closePanel: document.getElementById("close-panel"),
      specimenTitle: document.getElementById("specimen-title"),
      infoSpecies: document.getElementById("info-species"),
      infoSeason: document.getElementById("info-season"),
      infoYear: document.getElementById("info-year"),
      infoTracks: document.getElementById("info-tracks"),
      speciesFilter: document.getElementById("species-filter"),
      seasonFilter: document.getElementById("season-filter"),
      viewMode: document.getElementById("view-mode"),

      // Metrics
      metricCoherenceRating: document.getElementById("metric-coherence-rating"),
      metricCoherenceFill: document.getElementById("metric-coherence-fill"),
      metricCoherenceValue: document.getElementById("metric-coherence-value"),
      metricEntropyRating: document.getElementById("metric-entropy-rating"),
      metricEntropyFill: document.getElementById("metric-entropy-fill"),
      metricEntropyValue: document.getElementById("metric-entropy-value"),
      metricDriftDirection: document.getElementById("metric-drift-direction"),
      metricDriftValue: document.getElementById("metric-drift-value"),
      metricVariabilityRating: document.getElementById(
        "metric-variability-rating",
      ),
      metricVariabilityFill: document.getElementById("metric-variability-fill"),
      metricVariabilityValue: document.getElementById(
        "metric-variability-value",
      ),
    };
  }

  /**
   * Setup UI event listeners
   */
  setup() {
    // Scene click handler
    this.sculptureManager.sceneManager.onObjectClick = (object) => {
      this.onSculptureClick(object);
    };

    // Close panel button
    this.elements.closePanel?.addEventListener("click", () => {
      this.hideInfoPanel();
    });

    // Filter controls
    this.elements.speciesFilter?.addEventListener("change", (e) => {
      this.onFilterChange("species", e.target.value);
    });

    this.elements.seasonFilter?.addEventListener("change", (e) => {
      this.onFilterChange("season", e.target.value);
    });

    this.elements.viewMode?.addEventListener("change", (e) => {
      this.onViewModeChange(e.target.value);
    });
  }

  /**
   * Handle sculpture click
   */
  onSculptureClick(object) {
    if (object) {
      const specimen = this.sculptureManager.select(object);
      if (specimen) {
        this.showSpecimenInfo(specimen);
      }
    } else {
      this.sculptureManager.select(null);
      this.hideInfoPanel();
    }
  }

  /**
   * Show specimen information in panel
   */
  showSpecimenInfo(specimen) {
    const { species, season, year, n_tracks, metrics = {} } = specimen;

    // Update title
    const speciesName = this.formatSpeciesName(species);
    this.elements.specimenTitle.textContent = `${speciesName} - ${year}`;

    // Update basic info
    this.elements.infoSpecies.textContent = speciesName;
    this.elements.infoSeason.textContent = this.capitalize(season);
    this.elements.infoYear.textContent = year;
    this.elements.infoTracks.textContent = n_tracks || "—";

    // Update metrics
    this.updateMetrics(metrics);

    // Show panel
    this.elements.infoPanel.classList.remove("hidden");
  }

  /**
   * Update metrics display
   */
  updateMetrics(metrics) {
    // Coherence
    if (metrics.coherence) {
      const coh = metrics.coherence;
      this.elements.metricCoherenceRating.textContent = coh.rating || "—";
      this.elements.metricCoherenceFill.style.width = `${(coh.value || 0) * 100}%`;
      this.elements.metricCoherenceValue.textContent =
        typeof coh.value === "number" ? coh.value.toFixed(3) : "—";
    }

    // Entropy
    if (metrics.entropy) {
      const ent = metrics.entropy;
      this.elements.metricEntropyRating.textContent = ent.rating || "—";
      this.elements.metricEntropyFill.style.width = `${(ent.value || 0) * 100}%`;
      this.elements.metricEntropyValue.textContent =
        typeof ent.value === "number" ? ent.value.toFixed(3) : "—";
    }

    // Centroid drift
    if (metrics.centroid_drift) {
      const drift = metrics.centroid_drift;
      this.elements.metricDriftDirection.textContent = drift.direction || "—";
      this.elements.metricDriftValue.textContent =
        typeof drift.value_km === "number"
          ? `${drift.value_km.toFixed(2)} km`
          : "—";
    }

    // Temporal variability
    if (metrics.temporal_variability) {
      const vari = metrics.temporal_variability;
      this.elements.metricVariabilityRating.textContent = vari.rating || "—";
      this.elements.metricVariabilityFill.style.width = `${(vari.value || 0) * 100}%`;
      this.elements.metricVariabilityValue.textContent =
        typeof vari.value === "number" ? vari.value.toFixed(3) : "—";
    }
  }

  /**
   * Hide info panel
   */
  hideInfoPanel() {
    this.elements.infoPanel.classList.add("hidden");
  }

  /**
   * Handle filter change
   */
  onFilterChange(filterType, value) {
    console.log(`Filter change: ${filterType} = ${value}`);
    this.sculptureManager.applyFilters({ [filterType]: value });
  }

  /**
   * Handle view mode change
   */
  onViewModeChange(mode) {
    console.log(`View mode change: ${mode}`);
    // TODO: Implement switching between bundle and terrain modes
  }

  /**
   * Format species name for display
   */
  formatSpeciesName(species) {
    const names = {
      blue_whale: "Blue Whale",
      fin_whale: "Fin Whale",
      gray_whale: "Gray Whale",
      humpback_whale: "Humpback Whale",
    };
    return names[species] || species;
  }

  /**
   * Capitalize first letter
   */
  capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }
}
