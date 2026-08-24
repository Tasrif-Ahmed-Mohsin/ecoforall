"""
Academically Approved & Peer-Reviewed Data Ingestion Blueprint
================================================================
This script maps official academic data repositories, open API endpoints,
and peer-reviewed publication sources for all 13 indicators across
COLLECTIVE PSYCHOLOGY and SOCIETY.
"""

ACADEMIC_DATA_MAP = {
    # ------------------------------------------------------------------------
    # SOCIETY INDICATORS (Official International Bodies & Academic Datasets)
    # ------------------------------------------------------------------------
    "society_education": {
        "dataset_name": "Barro-Lee Educational Attainment & UNDP HDI",
        "institution": "Harvard University / Brown University / UNDP",
        "peer_reviewed_paper": "Barro, R. J., & Lee, J. W. (2013). 'A new data set of educational attainment in the world, 1950-2010.' Journal of Development Economics, 104, 184-198.",
        "api_endpoint": "https://api.worldbank.org/v2/country/all/indicator/SE.PRM.CMPT.ZS?format=json&per_page=10000",
        "description": "Primary completion rate (% of relevant age group) & Mean Years of Schooling."
    },
    "society_urbanization": {
        "dataset_name": "World Development Indicators (WDI) - Urbanization",
        "institution": "World Bank & United Nations Population Division",
        "peer_reviewed_paper": "United Nations (2019). 'World Urbanization Prospects: The 2018 Revision.' UN DESA.",
        "api_endpoint": "https://api.worldbank.org/v2/country/all/indicator/SP.URB.TOTL.IN.ZS?format=json&per_page=10000",
        "description": "Urban population as percentage of total population."
    },
    "society_population": {
        "dataset_name": "UN World Population Prospects (WPP) & WDI",
        "institution": "United Nations Department of Economic and Social Affairs (DESA)",
        "peer_reviewed_paper": "UN DESA (2022). 'World Population Prospects 2022: Summary of Results.'",
        "api_endpoint": "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=10000",
        "description": "Total midyear country population count."
    },
    "society_age": {
        "dataset_name": "UN WPP Age Structure & Dependency Ratios",
        "institution": "United Nations Population Division & World Bank",
        "peer_reviewed_paper": "Lutz, W., et al. (2018). 'Demographic and Human Capital Scenarios for the 21st Century.' IIASA.",
        "api_endpoint": "https://api.worldbank.org/v2/country/all/indicator/SP.POP.DPND?format=json&per_page=10000",
        "description": "Age dependency ratio (% of working-age population) & Median Age."
    },
    "society_migration": {
        "dataset_name": "UN International Migrant Stock & UNHCR Global Trends",
        "institution": "UN Migration Agency (IOM) & UNHCR",
        "peer_reviewed_paper": "Abel, G. J., & Sander, N. (2014). 'Quantifying global international migration flows.' Science, 343(6178), 1520-1522.",
        "api_endpoint": "https://api.worldbank.org/v2/country/all/indicator/SM.POP.NETM?format=json&per_page=10000",
        "description": "Net international migration count and refugee stock per country."
    },
    "society_religion": {
        "dataset_name": "Pew Research Center Global Religious Futures & ARDA",
        "institution": "Pew Research Center & Association of Religion Data Archives (Penn State)",
        "peer_reviewed_paper": "Pew Research Center (2015). 'The Future of World Religions: Population Growth Projections, 2010-2050.'",
        "public_url": "https://www.globalreligiousfutures.org/data-downloads",
        "description": "Religious diversity index, secularization metrics, and affiliation distributions."
    },
    "society_healthcare": {
        "dataset_name": "WHO Global Health Observatory & World Bank WDI",
        "institution": "World Health Organization (WHO) & World Bank",
        "peer_reviewed_paper": "Fullman, N., et al. (2018). 'Measuring performance on the Healthcare Access and Quality Index.' The Lancet, 391(10136), 2236-2271.",
        "api_endpoint": "https://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN?format=json&per_page=10000",
        "description": "Life expectancy at birth (years) & Universal Health Coverage (UHC) Service Coverage Index."
    },

    # ------------------------------------------------------------------------
    # COLLECTIVE PSYCHOLOGY INDICATORS (Peer-Reviewed Academic Survey Datasets)
    # ------------------------------------------------------------------------
    "psychology_trust": {
        "dataset_name": "World Values Survey (WVS) Wave 1-7 & V-Dem Dataset",
        "institution": "WVSA (JD Systems Institute) & V-Dem Institute (University of Gothenburg)",
        "peer_reviewed_paper": "Inglehart, R., et al. (2020). 'World Values Survey: All Rounds - Country-Pooled Datafile.' JD Systems Institute & WVSA Secretariat.",
        "public_url": "https://www.worldvaluessurvey.org/WVSContents.jsp",
        "vdem_code": "v2x_libdem",
        "description": "Generalized Interpersonal Trust (% responding 'Most people can be trusted') & Institutional Trust Index."
    },
    "psychology_fear": {
        "dataset_name": "Economic Policy Uncertainty (EPU) Index & ACLED Conflict Data",
        "institution": "Northwestern Univ / Stanford Univ / Univ of Chicago (Baker, Bloom, Davis)",
        "peer_reviewed_paper": "Baker, S. R., Bloom, N., & Davis, S. J. (2016). 'Measuring Economic Policy Uncertainty.' The Quarterly Journal of Economics, 131(4), 1593-1636.",
        "public_url": "https://www.policyuncertainty.com/data.html",
        "description": "Global & Country Economic Policy Uncertainty Index (Fear/Security Anxiety proxy)."
    },
    "psychology_optimism": {
        "dataset_name": "Gallup World Poll & OECD Consumer Confidence Index",
        "institution": "Gallup Organization & OECD",
        "peer_reviewed_paper": "Deaton, A. (2008). 'Income, health, and well-being around the world: Evidence from the Gallup World Poll.' Journal of Economic Perspectives, 22(2), 53-72.",
        "public_url": "https://www.gallup.com/analytics/213704/world-poll-development.aspx",
        "description": "Future Life Evaluation & Economic Outlook Score (Cantril Ladder 0-10 scale)."
    },
    "psychology_nationalism": {
        "dataset_name": "V-Dem Nationalism & Identity Module / WVS National Pride",
        "institution": "Varieties of Democracy (V-Dem) Institute & World Values Survey",
        "peer_reviewed_paper": "Coppedge, M., et al. (2023). 'V-Dem Dataset v13.' Varieties of Democracy Institute.",
        "public_url": "https://www.v-dem.net/data/dataset-archive/",
        "description": "In-group solidarity, national pride score (WVS Item G006), and protectionist sentiment index."
    },
    "psychology_social_cohesion": {
        "dataset_name": "V-Dem Political Polarization Index (v2xpol_publ) & SGI",
        "institution": "V-Dem Institute & Bertelsmann Stiftung Sustainable Governance Indicators",
        "peer_reviewed_paper": "Boxell, L., Gentzkow, M., & Shapiro, J. M. (2022). 'Cross-country trends in affective polarization.' Review of Economics and Statistics, 1-48.",
        "vdem_code": "v2xpol_publ",
        "public_url": "https://www.v-dem.net/",
        "description": "Political & Affective Polarization Index (Inverted for Social Harmony/Cohesion)."
    },
    "psychology_confidence": {
        "dataset_name": "Worldwide Governance Indicators (WGI) & OECD Systemic Trust",
        "institution": "World Bank Development Research Group & OECD",
        "peer_reviewed_paper": "Kaufmann, D., Kraay, A., & Mastruzzi, M. (2011). 'The Worldwide Governance Indicators: Methodology and Analytical Issues.' Hague Journal on the Rule of Law, 3(2), 220-246.",
        "api_endpoint": "https://api.worldbank.org/v2/country/all/indicator/GE.PER.RNK?format=json&per_page=10000",
        "description": "Government Effectiveness, Rule of Law, and Systemic Confidence percentile rank scores."
    }
}

def print_academic_provenance():
    print("================================================================================")
    print("      ACADEMIC & PEER-REVIEWED DATA SOURCES FOR COLLECTIVE PSYCHOLOGY & SOCIETY  ")
    print("================================================================Custom\n")
    for key, info in ACADEMIC_DATA_MAP.items():
        print(f"[*] [{key.upper()}]")
        print(f"   - Dataset Name        : {info['dataset_name']}")
        print(f"   - Lead Institution    : {info['institution']}")
        print(f"   - Peer-Reviewed Paper : {info['peer_reviewed_paper']}")
        if "api_endpoint" in info:
            print(f"   - Public API Endpoint : {info['api_endpoint']}")
        if "public_url" in info:
            print(f"   - Direct Data Portal  : {info['public_url']}")
        print(f"   - Indicator Focus     : {info['description']}\n")

if __name__ == "__main__":
    print_academic_provenance()
