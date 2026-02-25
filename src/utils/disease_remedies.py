"""Disease Remedies and Treatment Recommendations"""

# Comprehensive disease information database
DISEASE_REMEDIES = {
    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "severity": "Moderate",
        "description": "Fungal disease caused by Alternaria solani affecting tomato plants, causing dark spots with concentric rings on older leaves.",
        "symptoms": [
            "Dark brown spots with concentric rings (target-like pattern)",
            "Yellowing of leaves around spots",
            "Premature leaf drop",
            "Reduced fruit quality and yield"
        ],
        "causes": [
            "Warm, humid weather conditions",
            "Poor air circulation",
            "Overhead watering",
            "Infected plant debris in soil"
        ],
        "organic_remedies": [
            "Remove and destroy infected leaves immediately",
            "Apply neem oil spray (2-3 tablespoons per gallon of water)",
            "Use copper-based fungicides weekly",
            "Apply compost tea as foliar spray",
            "Mulch around plants to prevent soil splash"
        ],
        "chemical_remedies": [
            "Chlorothalonil-based fungicides (Daconil)",
            "Mancozeb fungicide applications",
            "Azoxystrobin (Quadris) for severe cases",
            "Rotate fungicides to prevent resistance"
        ],
        "prevention": [
            "Practice crop rotation (3-4 year cycle)",
            "Space plants properly for air circulation",
            "Water at soil level, avoid wetting foliage",
            "Remove plant debris at end of season",
            "Use disease-resistant varieties",
            "Apply mulch to prevent soil splash"
        ],
        "best_practices": [
            "Monitor plants weekly for early detection",
            "Maintain proper plant nutrition",
            "Avoid working with plants when wet",
            "Sanitize tools between plants"
        ]
    },
    
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "severity": "Severe",
        "description": "Devastating disease caused by Phytophthora infestans that can destroy entire crops within days.",
        "symptoms": [
            "Water-soaked spots on leaves",
            "White fuzzy growth on leaf undersides",
            "Brown lesions on stems",
            "Firm brown rot on fruits"
        ],
        "causes": [
            "Cool, wet weather (60-70°F)",
            "High humidity and moisture",
            "Infected seed potatoes or transplants",
            "Wind-borne spores from nearby infected plants"
        ],
        "organic_remedies": [
            "Remove and destroy all infected plants immediately",
            "Apply copper fungicide at first sign",
            "Use Bacillus subtilis biological fungicide",
            "Improve air circulation drastically"
        ],
        "chemical_remedies": [
            "Chlorothalonil (preventive)",
            "Mancozeb + copper combinations",
            "Cymoxanil for curative action",
            "Mandipropamid (Revus) for severe outbreaks"
        ],
        "prevention": [
            "Plant certified disease-free transplants",
            "Use resistant varieties when available",
            "Avoid overhead irrigation completely",
            "Space plants widely for air flow",
            "Apply preventive fungicides in wet weather",
            "Monitor weather forecasts for blight conditions"
        ],
        "best_practices": [
            "Scout fields daily during favorable conditions",
            "Act immediately at first symptoms",
            "Coordinate with neighboring farms",
            "Keep detailed spray records"
        ]
    },
    
    "Tomato___healthy": {
        "name": "Healthy Tomato Plant",
        "severity": "None",
        "description": "Plant shows no signs of disease. Continue good cultural practices.",
        "symptoms": [
            "Vibrant green foliage",
            "No spots or discoloration",
            "Strong, upright growth",
            "Healthy fruit development"
        ],
        "causes": [],
        "organic_remedies": [],
        "chemical_remedies": [],
        "prevention": [
            "Maintain current care practices",
            "Continue regular monitoring",
            "Ensure proper watering and nutrition",
            "Keep area clean and weed-free"
        ],
        "best_practices": [
            "Weekly plant inspections",
            "Maintain soil health with compost",
            "Proper spacing and pruning",
            "Balanced fertilization"
        ]
    },
    
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "severity": "Moderate",
        "description": "Fungal disease affecting potato plants, reducing yield and tuber quality.",
        "symptoms": [
            "Brown spots with concentric rings on lower leaves",
            "Yellowing and wilting of affected leaves",
            "Lesions on stems and tubers",
            "Premature defoliation"
        ],
        "causes": [
            "Warm temperatures (75-85°F)",
            "High humidity",
            "Plant stress from drought or poor nutrition",
            "Infected seed tubers"
        ],
        "organic_remedies": [
            "Remove infected foliage promptly",
            "Apply copper-based organic fungicides",
            "Use neem oil spray weekly",
            "Improve soil drainage",
            "Apply compost for plant vigor"
        ],
        "chemical_remedies": [
            "Chlorothalonil applications",
            "Mancozeb fungicide program",
            "Azoxystrobin for systemic protection",
            "Alternate fungicide classes"
        ],
        "prevention": [
            "Plant certified disease-free seed potatoes",
            "Practice 3-4 year crop rotation",
            "Hill soil around plants properly",
            "Maintain adequate plant nutrition",
            "Avoid overhead irrigation",
            "Remove volunteer potatoes"
        ],
        "best_practices": [
            "Monitor lower leaves first",
            "Maintain consistent soil moisture",
            "Apply balanced fertilizer",
            "Harvest at proper maturity"
        ]
    },
    
    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "severity": "Severe",
        "description": "Highly destructive disease that can cause total crop loss in favorable conditions.",
        "symptoms": [
            "Water-soaked lesions on leaves",
            "White mold on leaf undersides",
            "Blackened stems",
            "Brown rot in tubers"
        ],
        "causes": [
            "Cool, wet conditions (50-70°F)",
            "Extended leaf wetness",
            "Infected seed tubers",
            "Airborne spores from infected plants"
        ],
        "organic_remedies": [
            "Destroy all infected plants immediately",
            "Apply copper fungicide preventively",
            "Use Bacillus subtilis products",
            "Improve field drainage"
        ],
        "chemical_remedies": [
            "Chlorothalonil (protective)",
            "Cymoxanil + mancozeb (curative)",
            "Fluazinam for resistant strains",
            "Mandipropamid for systemic action"
        ],
        "prevention": [
            "Use certified disease-free seed",
            "Plant resistant varieties",
            "Avoid irrigation during cool, humid periods",
            "Apply preventive fungicides before symptoms",
            "Monitor blight forecasting systems",
            "Destroy cull piles promptly"
        ],
        "best_practices": [
            "Scout fields twice weekly in risk periods",
            "Kill vines before harvest if infected",
            "Cure tubers properly before storage",
            "Sanitize equipment and storage areas"
        ]
    },
    
    "Potato___healthy": {
        "name": "Healthy Potato Plant",
        "severity": "None",
        "description": "Plant is healthy with no disease symptoms detected.",
        "symptoms": [
            "Dark green, vigorous foliage",
            "No lesions or discoloration",
            "Strong stem growth",
            "Normal tuber development"
        ],
        "causes": [],
        "organic_remedies": [],
        "chemical_remedies": [],
        "prevention": [
            "Continue current management practices",
            "Regular field monitoring",
            "Maintain soil fertility",
            "Proper irrigation scheduling"
        ],
        "best_practices": [
            "Weekly plant health checks",
            "Soil testing annually",
            "Proper hilling and cultivation",
            "Timely harvest"
        ]
    },
    
    "Pepper__bell___Bacterial_spot": {
        "name": "Pepper Bacterial Spot",
        "severity": "Moderate to Severe",
        "description": "Bacterial disease causing leaf spots and fruit lesions, reducing marketability.",
        "symptoms": [
            "Small, dark brown spots on leaves",
            "Raised, scabby lesions on fruits",
            "Yellow halos around leaf spots",
            "Premature leaf drop"
        ],
        "causes": [
            "Warm, wet weather",
            "Overhead irrigation",
            "Contaminated seeds or transplants",
            "Wounds from insects or handling"
        ],
        "organic_remedies": [
            "Remove and destroy infected plants",
            "Apply copper-based bactericides",
            "Use biological controls (Bacillus spp.)",
            "Improve air circulation",
            "Avoid overhead watering"
        ],
        "chemical_remedies": [
            "Copper hydroxide sprays",
            "Copper + mancozeb combinations",
            "Acibenzolar-S-methyl for resistance induction",
            "Streptomycin (where permitted)"
        ],
        "prevention": [
            "Use certified disease-free seeds and transplants",
            "Practice crop rotation (3+ years)",
            "Drip irrigation instead of overhead",
            "Sanitize tools and equipment",
            "Plant resistant varieties",
            "Avoid working with wet plants"
        ],
        "best_practices": [
            "Start with clean transplants",
            "Maintain plant spacing",
            "Remove plant debris",
            "Monitor for early symptoms"
        ]
    },
    
    "Pepper__bell___healthy": {
        "name": "Healthy Pepper Plant",
        "severity": "None",
        "description": "Plant is healthy and disease-free.",
        "symptoms": [
            "Bright green leaves",
            "No spots or lesions",
            "Vigorous growth",
            "Healthy fruit set"
        ],
        "causes": [],
        "organic_remedies": [],
        "chemical_remedies": [],
        "prevention": [
            "Maintain current practices",
            "Regular monitoring",
            "Proper nutrition and watering",
            "Good sanitation"
        ],
        "best_practices": [
            "Weekly inspections",
            "Balanced fertilization",
            "Adequate spacing",
            "Timely harvesting"
        ]
    }
}

def get_remedy(disease_class: str) -> dict:
    """
    Get remedy information for a disease
    
    Args:
        disease_class: Disease class name
    
    Returns:
        Dictionary with remedy information
    """
    return DISEASE_REMEDIES.get(disease_class, {
        "name": disease_class.replace('_', ' '),
        "severity": "Unknown",
        "description": "No information available for this disease class.",
        "symptoms": [],
        "causes": [],
        "organic_remedies": ["Consult with local agricultural extension service"],
        "chemical_remedies": ["Consult with certified agronomist"],
        "prevention": ["Practice good agricultural hygiene"],
        "best_practices": ["Regular monitoring and early detection"]
    })

def is_valid_plant_image(confidence: float, threshold: float = 50.0) -> bool:
    """
    Determine if image is valid based on confidence threshold
    
    Args:
        confidence: Prediction confidence percentage
        threshold: Minimum confidence threshold
    
    Returns:
        True if valid, False otherwise
    """
    return confidence >= threshold
