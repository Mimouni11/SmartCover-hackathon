"""
Insurance claim templates based on real Tunisian insurance forms.
Each template defines required/optional fields and marks which fields contain private data.

PRIVACY: Fields marked as 'private' will be excluded from AI workflow and anonymized.
"""

TEMPLATES = {
    "health": {
        "description": "Bulletin de soins - Health/Medical/Dental care claim",
        "required_fields": [
            "date_of_care",
            "beneficiary_type",  # adhérent/conjoint/enfants/parents
            "care_type",  # soins dentaires/consultations/actes médicaux/pharmacie/hospitalisation/accouchement
            "medical_provider_type",  # dentiste/médecin/hôpital/pharmacie
            "total_amount",
        ],
        "optional_fields": [
            "nature_of_illness",  # Nature de la maladie
            "consultation_details",  # Date, Désignation, Honoraires
            "medical_acts",  # Actes médicaux
            "medications",  # Pharmacie - médicaments prescrits
            "hospitalization_details",  # Dates, services
            "biology_tests",  # Tests biologiques
            "receipts",  # Matricule Fiscal, factures
            "medical_notes",  # Notes confidentielles du médecin
        ],
        "private_fields": [
            # These will be EXCLUDED from AI workflow
            "contract_number",
            "adherent_name",
            "adherent_address",
            "adherent_job",
            "beneficiary_name",
            "national_id",
            "phone_number",
        ],
        "field_types": {
            "date_of_care": "date",
            "beneficiary_type": "enum",  # [adhérent, conjoint, enfants, parents]
            "care_type": "enum",  # [dentaire, consultation, acte_médical, pharmacie, hospitalisation, accouchement, biologie]
            "total_amount": "float",
            "receipts": "list[file]",
        }
    },

    "auto": {
        "description": "Constat amiable - Accident automobile",
        "required_fields": [
            "accident_date",
            "accident_time",
            "accident_location",
            "collision_type",  # rear/front/side/parking/stationnement
            "vehicle_damage_description",
            "other_party_involved",  # bool
        ],
        "optional_fields": [
            "vehicle_brand_model",
            "collision_circumstances",  # Cases à cocher du constat
            "damage_point_of_impact",  # Point de choc initial
            "visible_damages",  # Dégâts apparents
            "other_vehicle_info",  # Marque, type de l'autre véhicule
            "witness_present",  # bool
            "witness_info",  # Nom, adresse des témoins
            "damage_photos",  # list[file]
            "police_report",  # bool
            "injuries",  # bool - blessés
            "injury_description",
            "remarks",  # Observations
        ],
        "private_fields": [
            # These will be EXCLUDED from AI workflow
            "vehicle_plate",  # Immatriculation
            "driver_name",
            "driver_license",  # Permis de conduire
            "insurance_policy_number",
            "other_driver_name",
            "other_driver_license",
            "other_vehicle_plate",
            "other_insurance_details",
            "witness_names",
            "witness_addresses",
        ],
        "field_types": {
            "accident_date": "date",
            "accident_time": "time",
            "accident_location": "string",
            "collision_type": "enum",  # [rear, front, side, parking, stationnement, chaîne]
            "other_party_involved": "bool",
            "witness_present": "bool",
            "injuries": "bool",
            "damage_photos": "list[file]",
        }
    },

    "home": {
        "description": "Sinistre habitation - Home/Property damage claim",
        "required_fields": [
            "incident_date",
            "incident_time",
            "property_address",
            "damage_type",  # dégât des eaux/incendie/vol/tempête/bris de glace
            "damage_description",
        ],
        "optional_fields": [
            "affected_rooms",  # Pièces touchées
            "cause_of_damage",  # Cause du sinistre
            "water_source",  # For dégât des eaux: fuite, rupture canalisation, etc.
            "fire_cause",  # For incendie: électrique, cuisson, etc.
            "stolen_items",  # For vol: list of items
            "forced_entry",  # bool - effraction
            "temporary_repairs",  # Réparations provisoires effectuées
            "damage_photos",
            "police_report",  # bool - especially for theft
            "expert_report",  # Rapport d'expertise
            "estimated_cost",
            "emergency_services_called",  # Pompiers/police appelés
        ],
        "private_fields": [
            # These will be EXCLUDED from AI workflow
            "policy_holder_name",
            "policy_number",
            "property_full_address",
            "phone_number",
            "national_id",
        ],
        "field_types": {
            "incident_date": "date",
            "incident_time": "time",
            "damage_type": "enum",  # [dégât_des_eaux, incendie, vol, tempête, bris_de_glace, catastrophe_naturelle]
            "forced_entry": "bool",
            "police_report": "bool",
            "estimated_cost": "float",
            "damage_photos": "list[file]",
        }
    },

    "travel": {
        "description": "Assurance voyage - Travel insurance claim",
        "required_fields": [
            "incident_date",
            "incident_location",  # Pays/ville
            "trip_destination",
            "trip_start_date",
            "trip_end_date",
            "incident_type",  # annulation/retard/bagages/médical/rapatriement
            "incident_description",
        ],
        "optional_fields": [
            "flight_number",
            "booking_reference",
            "delay_duration",  # For retards
            "cancellation_reason",  # For annulations
            "medical_issue",  # For incidents médicaux
            "hospital_name",  # If hospitalized abroad
            "treatment_received",
            "lost_baggage_details",  # Contents, value
            "baggage_delay_hours",
            "receipts",  # Hotel, meals, emergency purchases
            "airline_compensation",  # Received any compensation from airline?
            "travel_documents",  # Boarding pass, baggage claim receipts
            "medical_receipts",
            "police_report",  # For theft/lost baggage
            "amount_claimed",
        ],
        "private_fields": [
            # These will be EXCLUDED from AI workflow
            "policy_holder_name",
            "policy_number",
            "passport_number",
            "phone_number",
            "emergency_contact",
            "credit_card_number",  # If used for bookings
        ],
        "field_types": {
            "incident_date": "date",
            "trip_start_date": "date",
            "trip_end_date": "date",
            "incident_type": "enum",  # [annulation, retard, bagages_perdus, bagages_retardés, médical, rapatriement, responsabilité_civile]
            "delay_duration": "int",  # hours
            "police_report": "bool",
            "amount_claimed": "float",
            "receipts": "list[file]",
        }
    },
}


# Enum definitions for validation
ENUMS = {
    "beneficiary_type": ["adhérent", "conjoint", "enfants", "parents"],
    "care_type": ["dentaire", "consultation", "acte_médical", "pharmacie", "hospitalisation", "accouchement", "biologie"],
    "collision_type": ["rear", "front", "side", "parking", "stationnement", "chaîne", "autre"],
    "damage_type_home": ["dégât_des_eaux", "incendie", "vol", "tempête", "bris_de_glace", "catastrophe_naturelle", "dégradation"],
    "incident_type_travel": ["annulation", "retard", "bagages_perdus", "bagages_retardés", "médical", "rapatriement", "responsabilité_civile"],
}


def get_all_fields(insurance_type: str) -> list:
    """Get all fields (required + optional) for an insurance type."""
    template = TEMPLATES.get(insurance_type)
    if not template:
        return []
    return template["required_fields"] + template["optional_fields"]


def get_private_fields(insurance_type: str) -> list:
    """Get fields that should be excluded from AI workflow."""
    template = TEMPLATES.get(insurance_type)
    if not template:
        return []
    return template.get("private_fields", [])


def get_required_fields(insurance_type: str) -> list:
    """Get only required fields for an insurance type."""
    template = TEMPLATES.get(insurance_type)
    if not template:
        return []
    return template["required_fields"]


def is_valid_insurance_type(insurance_type: str) -> bool:
    """Check if insurance type is supported."""
    return insurance_type in TEMPLATES


def get_supported_types() -> list:
    """Get list of all supported insurance types."""
    return list(TEMPLATES.keys())