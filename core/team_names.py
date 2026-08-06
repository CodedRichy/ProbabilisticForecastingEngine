"""
Shared team-name normalisation utilities.

TEAM_ALIASES maps every known variant spelling → FIFA official canonical name.
Use normalize_team() to resolve a raw name from any source (ESPN, user input,
historical CSV datasets, etc.) to the canonical form.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical names follow FIFA official designations (as used on FIFA.com and
# in the martj42/international_results dataset that powers EloModel).
# Any alias key pointing to a canonical value should itself NOT appear as a
# value in this dict — canonical values are the ground truth.
# ---------------------------------------------------------------------------

TEAM_ALIASES: dict[str, str] = {
    # United States
    "USA":                          "United States",
    "US":                           "United States",
    "United States":                "United States",

    # South Korea / Korea Republic  (FIFA canonical: "Korea Republic")
    "South Korea":                  "Korea Republic",
    "Korea Republic":               "Korea Republic",

    # North Korea  (FIFA canonical: "Korea DPR")
    "Korea DPR":                    "Korea DPR",
    "North Korea":                  "Korea DPR",

    # Ivory Coast  (martj42/Elo dataset uses "Ivory Coast" — canonical here)
    "Ivory Coast":                  "Ivory Coast",
    "Cote d'Ivoire":                "Ivory Coast",
    "Côte d'Ivoire":                "Ivory Coast",

    # Iran  (FIFA canonical: "IR Iran")
    "Iran":                         "IR Iran",
    "IR Iran":                      "IR Iran",

    # Trinidad & Tobago  (FIFA canonical: "Trinidad and Tobago")
    "Trinidad":                     "Trinidad and Tobago",
    "Trinidad And Tobago":          "Trinidad and Tobago",
    "Trinidad & Tobago":            "Trinidad and Tobago",
    "Trinidad and Tobago":          "Trinidad and Tobago",

    # Bosnia  (FIFA canonical: "Bosnia and Herzegovina")
    "Bosnia":                       "Bosnia and Herzegovina",
    "Bosnia-Herzegovina":           "Bosnia and Herzegovina",
    "Bosnia & Herzegovina":         "Bosnia and Herzegovina",
    "Bosnia and Herzegovina":       "Bosnia and Herzegovina",

    # Czechia / Czech Republic  (FIFA canonical: "Czechia")
    "Czech Republic":               "Czechia",
    "Czechia":                      "Czechia",

    # Cape Verde  (FIFA canonical: "Cape Verde Islands")
    "Cape Verde":                   "Cape Verde Islands",
    "Cape Verde Islands":           "Cape Verde Islands",

    # Congo DR  (FIFA canonical: "DR Congo")
    "Congo DR":                     "DR Congo",
    "DRC":                          "DR Congo",
    "DR Congo":                     "DR Congo",

    # North Macedonia  (FIFA canonical: "North Macedonia")
    "Macedonia":                    "North Macedonia",
    "North Macedonia":              "North Macedonia",

    # China  (FIFA canonical: "China PR")
    "China":                        "China PR",
    "China PR":                     "China PR",

    # Eswatini (formerly Swaziland)  (FIFA canonical: "Eswatini")
    "Swaziland":                    "Eswatini",
    "Eswatini":                     "Eswatini",

    # Kyrgyzstan  (FIFA canonical: "Kyrgyz Republic")
    "Kyrgyzstan":                   "Kyrgyz Republic",
    "Kyrgyz Republic":              "Kyrgyz Republic",

    # Macao  (FIFA canonical: "Macao")
    "Macau":                        "Macao",
    "Macao":                        "Macao",

    # United Arab Emirates
    "UAE":                          "United Arab Emirates",
    "United Arab Emirates":         "United Arab Emirates",

    # Saudi Arabia
    "KSA":                          "Saudi Arabia",
    "Saudi Arabia":                 "Saudi Arabia",

    # Passthrough entries — names that are already canonical
    "New Zealand":                  "New Zealand",
    "Chinese Taipei":               "Chinese Taipei",

    # ---------------------------------------------------------------------------
    # WC2026 — variants seen in The Odds API, ESPN, and the martj42 CSV that are
    # NOT already covered above.  Canonical values follow FIFA.com / martj42.
    # ---------------------------------------------------------------------------

    # Argentina, Brazil, France, Germany, Spain, England — identical across all
    # sources; no alias needed.

    # Portugal
    "Portugal":                     "Portugal",

    # Netherlands / Holland
    "Netherlands":                  "Netherlands",
    "Holland":                      "Netherlands",

    # Belgium
    "Belgium":                      "Belgium",

    # Switzerland
    "Switzerland":                  "Switzerland",

    # Croatia
    "Croatia":                      "Croatia",

    # Denmark
    "Denmark":                      "Denmark",

    # Serbia  (The Odds API may use "Serbia")
    "Serbia":                       "Serbia",
    "Republic of Serbia":           "Serbia",

    # Austria
    "Austria":                      "Austria",

    # Poland
    "Poland":                       "Poland",

    # Hungary
    "Hungary":                      "Hungary",

    # Slovakia
    "Slovakia":                     "Slovakia",

    # Romania
    "Romania":                      "Romania",

    # Ukraine
    "Ukraine":                      "Ukraine",

    # Turkey / Türkiye  (FIFA rebranded to "Türkiye" but older data uses "Turkey")
    "Turkey":                       "Turkey",
    "Türkiye":                      "Turkey",
    "Turkiye":                      "Turkey",

    # Georgia
    "Georgia":                      "Georgia",

    # Scotland
    "Scotland":                     "Scotland",

    # Wales
    "Wales":                        "Wales",

    # Mexico
    "Mexico":                       "Mexico",

    # Canada
    "Canada":                       "Canada",

    # Honduras
    "Honduras":                     "Honduras",

    # Panama
    "Panama":                       "Panama",

    # Costa Rica
    "Costa Rica":                   "Costa Rica",

    # Jamaica
    "Jamaica":                      "Jamaica",

    # Ecuador
    "Ecuador":                      "Ecuador",

    # Uruguay
    "Uruguay":                      "Uruguay",

    # Colombia
    "Colombia":                     "Colombia",

    # Chile
    "Chile":                        "Chile",

    # Venezuela
    "Venezuela":                    "Venezuela",

    # Paraguay
    "Paraguay":                     "Paraguay",

    # Bolivia
    "Bolivia":                      "Bolivia",

    # Peru
    "Peru":                         "Peru",

    # Morocco
    "Morocco":                      "Morocco",

    # Egypt
    "Egypt":                        "Egypt",

    # Senegal
    "Senegal":                      "Senegal",

    # Nigeria
    "Nigeria":                      "Nigeria",

    # Cameroon
    "Cameroon":                     "Cameroon",

    # Ghana
    "Ghana":                        "Ghana",

    # Tunisia
    "Tunisia":                      "Tunisia",

    # Algeria
    "Algeria":                      "Algeria",

    # Mali
    "Mali":                         "Mali",

    # South Africa
    "South Africa":                 "South Africa",

    # Japan
    "Japan":                        "Japan",

    # Australia / Socceroos
    "Australia":                    "Australia",
    "Socceroos":                    "Australia",

    # Saudi Arabia — already above; kept for clarity
    # "Saudi Arabia":               "Saudi Arabia",

    # Qatar
    "Qatar":                        "Qatar",

    # Uzbekistan
    "Uzbekistan":                   "Uzbekistan",

    # Indonesia
    "Indonesia":                    "Indonesia",
}


def normalize_team(name: str) -> str:
    """
    Return the FIFA canonical name for *name*, or *name* unchanged if it is
    not listed in TEAM_ALIASES.

    Parameters
    ----------
    name : str
        Raw team name from any data source.

    Returns
    -------
    str
        Canonical team name.
    """
    if not isinstance(name, str):
        return name
    name = name.strip()
    return TEAM_ALIASES.get(name, name)
