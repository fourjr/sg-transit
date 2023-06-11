from models import API, FareType


def calculate_fare(api: API, fare_type: FareType, distance: float):
    if distance == 0:
        return 0

    fare_structures = api.get_fare_structures()
    for i in fare_structures[fare_type]:
        if distance < i.to_distance:
            return i.card_fare, i.cash_fare
