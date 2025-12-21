def rule_engine(claim):
    signals = []

    if claim["total_claim_amount"] > 50000:
        signals.append("montant exceptionnel")

    if claim["months_as_customer"] < 6:
        signals.append("nouveau client")

    if claim["incident_hour_of_the_day"] < 5:
        signals.append("incident nocturne")

    if claim["witnesses"] == 0:
        signals.append("aucun témoin")

    return signals
