"""
Prompts for Mistral LLM to extract structured insurance claim data.
Supports French and English inputs.
"""

from .templates import TEMPLATES, ENUMS, get_required_fields, get_all_fields


SYSTEM_PROMPT = """Tu es un assistant d'assurance intelligent spécialisé dans l'extraction d'informations structurées à partir de descriptions en langage naturel.

TON RÔLE:
1. Identifier le type d'assurance concerné (santé/auto/habitation/voyage)
2. Extraire UNIQUEMENT les informations EXPLICITEMENT mentionnées
3. Signaler les champs manquants obligatoires
4. Répondre UNIQUEMENT en JSON valide

TYPES D'ASSURANCE SUPPORTÉS:
- "health" (santé) : soins médicaux, dentaires, hospitalisation, pharmacie, blessures, douleurs
- "auto" : accidents de voiture, collisions, dommages véhicule
- "home" (habitation) : dégât des eaux, incendie, vol, tempête
- "travel" (voyage) : annulation, retard, bagages perdus, incidents médicaux à l'étranger

⚠️ DÉTECTION MULTI-TYPES - TRÈS IMPORTANT:
- Si accident de voiture + blessure/douleur/injury → ["auto", "health"]
- Si incident voyage + problème médical → ["travel", "health"]
- Si dégât maison + blessure → ["home", "health"]
- Toujours chercher plusieurs types possibles!

🚫 RÈGLES ANTI-HALLUCINATION - CRITIQUES:
- N'INVENTE JAMAIS de dates, heures, lieux ou détails
- Si l'utilisateur ne mentionne pas une info → mets null
- Si tu n'es pas SÛR d'une valeur → mets null
- Seule exception: "hier"/"yesterday" → calcule la date d'hier
- Copie les mots exacts de l'utilisateur pour les descriptions
- NE PAS utiliser les exemples du prompt comme données réelles

RÈGLES TECHNIQUES:
- Utilise les valeurs enum exactes (ex: "rear", "front" pour collision_type)
- Pour les dates: format YYYY-MM-DD
- Pour les booléens: true/false
- Si une info n'est pas mentionnée: null (pas de "", pas de "unknown")
- NE PAS inclure de données personnelles (noms, plaques, numéros de contrat)

LANGUE:
- Accepte input en français et anglais
- Comprends les expressions familières ("j'ai crashé ma voiture" = accident auto)
- "neck broken" / "cou cassé" / "neck hurts" = health insurance (blessure)
- "paid" / "payé" + montant = total_amount

FORMAT DE RÉPONSE:
Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ou après.
Pas de markdown, pas de ```json, juste le JSON brut.
"""


def build_user_prompt(user_input: str) -> str:
    """
    Build the user prompt with the user's claim description.
    Includes all templates for context.
    """

    # Build detailed template info with ALL fields
    templates_info = ""
    for ins_type, template in TEMPLATES.items():
        required = template["required_fields"]
        optional = template["optional_fields"]

        templates_info += f"\n{ins_type.upper()} - {template['description']}:\n"
        templates_info += f"  OBLIGATOIRES: {', '.join(required)}\n"
        templates_info += f"  OPTIONNELS: {', '.join(optional)}\n"

    # Build enum info
    enums_info = "\nVALEURS ENUM VALIDES:\n"
    for enum_name, values in ENUMS.items():
        enums_info += f"  {enum_name}: {values}\n"

    prompt = f"""DESCRIPTION DE L'INCIDENT PAR L'UTILISATEUR:
"{user_input}"

TEMPLATES D'ASSURANCE DISPONIBLES:
{templates_info}

{enums_info}

RÈGLES CRITIQUES - LIS ATTENTIVEMENT:

1. DÉTECTION DE TYPE D'ASSURANCE:
   - Accident de voiture = "auto"
   - Blessure/injury/douleur/neck/cou = "health" 
   - Accident + blessure = DEUX types: ["auto", "health"]
   - Maison/fuite/incendie/vol = "home"
   - Voyage/vol/bagages = "travel"

2. EXTRACTION:
   - Extrais UNIQUEMENT ce qui est MENTIONNÉ dans la description
   - Si une info n'est PAS mentionnée → utilise null
   - N'INVENTE JAMAIS de données (dates, lieux, détails)
   - Si l'utilisateur dit "yesterday/hier" → calcule la date d'hier
   - Sinon si date non mentionnée → null

3. CHAMPS MANQUANTS:
   - Liste TOUS les champs obligatoires qui ne sont PAS dans la description
   - Compare avec les "OBLIGATOIRES" du template

4. CONFIANCE:
   - Haute (0.8-1.0): beaucoup de détails fournis
   - Moyenne (0.5-0.8): quelques détails
   - Basse (0.0-0.5): très peu d'info

5. FOLLOWUP:
   - Si champs obligatoires manquants → needs_followup: true
   - Demande LE PREMIER champ manquant seulement

STRUCTURE JSON À RETOURNER (copie exactement cette structure):
{{
    "insurance_types": [],
    "structured_claims": {{}},
    "missing_required_fields": {{}},
    "confidence": 0.0,
    "needs_followup": true,
    "followup_question": ""
}}

EXEMPLE DE REMPLISSAGE (NE COPIE PAS CES VALEURS):
Si user dit: "car accident yesterday, neck hurts, paid $500"
→ insurance_types: ["auto", "health"]  // DEUX types car accident + blessure
→ structured_claims: {{
    "auto": {{"accident_date": "2024-12-19", "other_party_involved": null}},
    "health": {{"date_of_care": "2024-12-19", "total_amount": 500}}
  }}
→ missing_required_fields: {{"auto": ["accident_location", "collision_type", ...], "health": ["care_type", ...]}}

MAINTENANT ANALYSE LA DESCRIPTION DE L'UTILISATEUR ET RETOURNE LE JSON:"""

    return prompt


def build_followup_prompt(user_input: str, previous_data: dict, missing_fields: list) -> str:
    """
    Build a followup prompt when user provides additional information.
    """

    prompt = f"""L'utilisateur a fourni des informations supplémentaires:
"{user_input}"

CONTEXTE PRÉCÉDENT:
{previous_data}

CHAMPS MANQUANTS À REMPLIR:
{missing_fields}

INSTRUCTIONS:
1. Extrais UNIQUEMENT les nouvelles informations fournies
2. Ne réécris pas les données déjà présentes
3. Mets à jour seulement les champs concernés

RÉPONDS AVEC CE FORMAT:
{{
    "updated_fields": {{
        "vehicle_plate": "1234-TU-567",
        "other_party_info": "..."
    }},
    "still_missing": ["damage_photos"],
    "needs_followup": true/false,
    "followup_question": "..."
}}

JSON uniquement:"""

    return prompt


# Example prompts for testing
EXAMPLE_INPUTS = {
    "auto_simple": "Hier j'ai eu un accident sur l'autoroute A1. Une voiture m'a percuté par derrière.",

    "auto_detailed": "Accident de voiture ce matin à 8h30 sur la Route de la Marsa. Un camion m'a rentré dedans par l'arrière au feu rouge. Mon pare-chocs est complètement défoncé. Il y avait des témoins.",

    "health_dental": "Je suis allé chez le dentiste hier pour une carie. J'ai payé 150 dinars pour le soin.",

    "health_hospital": "J'ai été hospitalisé pendant 3 jours à l'hôpital Charles Nicolle pour une appendicite. Opération le 15 janvier.",

    "home_water": "Fuite d'eau dans ma cuisine ce matin. L'eau venait du tuyau sous l'évier. Toute la cuisine est inondée.",

    "home_fire": "Incendie dans mon salon hier soir. Ça a commencé à cause d'une bougie. Les pompiers sont venus.",

    "travel_delay": "Mon vol Paris-Tunis a été annulé hier. J'ai dû passer la nuit à l'hôtel. Vol AF1234.",

    "multi_auto_health": "Accident de voiture hier. Un camion m'a percuté et je me suis blessé au cou. J'ai dû aller aux urgences.",

    "english_auto": "I had a car accident yesterday on Route 7. Someone hit me from behind and damaged my rear bumper.",
}


def get_example_prompts():
    """Get example prompts for testing."""
    examples = {}
    for key, user_input in EXAMPLE_INPUTS.items():
        examples[key] = {
            "system": SYSTEM_PROMPT,
            "user": build_user_prompt(user_input)
        }
    return examples