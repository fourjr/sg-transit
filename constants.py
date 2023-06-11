class Routes:
    DATAMALL_BASE = 'http://datamall2.mytransport.sg/ltaodataservice'
    BUS_STOPS = DATAMALL_BASE + '/BusStops'
    BUS_ROUTES = DATAMALL_BASE + '/BusRoutes'
    TRAIN_STATIONS = DATAMALL_BASE + '/GeospatialWholeIsland?ID=TrainStation'

    TRAIN_STATION_NAMES = 'https://datamall.lta.gov.sg/content/dam/datamall/datasets/PublicTransportRelated/Train%20Station%20Codes%20and%20Chinese%20Names.zip'

    FARE_STRUCTURE = 'https://www.ptc.gov.sg/fare-regulation/bus-rail/fare-structure'
