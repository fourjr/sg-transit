from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import gzip
import io
import math
import pickle
import json
from typing import Dict, Generator, List
import zipfile

import xlrd
import requests
import pyproj
import shapefile
from cachetools import TTLCache
from bs4 import BeautifulSoup

from constants import Routes


@dataclass
class BusStop:
    code: str
    road_name: str
    name: str
    latitude: float
    longitude: float

    def __init__(self, data):
        self.code = data['BusStopCode']
        self.road_name = data['RoadName']
        self.name = data['Description']
        self.latitude = data['Latitude']
        self.longitude = data['Longitude']


@dataclass
class BusRouteStop:
    bus_stop: BusStop
    route: 'BusRoute'
    distance: float

    def calculate_distance(self, other: 'BusRouteStop'):
        """Returns the distance between 2 bus stops in km"""
        return abs(self.distance - other.distance)


class Direction(IntEnum):
    FORWARD = 1
    BACKWARD = 2


class FareType(IntEnum):
    STUDENT = 0
    ADULT = 1
    SENIOR_CITIZEN = 2
    WTC = 3
    DISABILITY = 4


@dataclass
class BusRoute:
    service_no: str
    direction: Direction
    operator: str
    stops: List[BusRouteStop] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"<BusRoute svc={self.service_no} dir={self.direction.name} stops={len(self.stops)}>"


@dataclass
class FareStructureDistance:
    fare_type: FareType
    from_distance: float
    to_distance: float
    card_fare: int
    cash_fare: int

class StationType(Enum):
    MRT = 1
    LRT = 2

@dataclass
class TrainStation:
    codes: List[str]
    station_type: StationType
    name: str
    latitude: float
    longitude: float

    def calculate_distance(self, other: 'TrainStation'):
        """Returns the distance between 2 train stations in km"""
        # https://stackoverflow.com/a/21623206/8129786
        p = math.pi / 180
        a = (
            0.5
            - math.cos((other.latitude - self.latitude) * p) / 2
            + math.cos(self.latitude * p)
            * math.cos(other.latitude * p)
            * (1 - math.cos((other.longitude - self.longitude) * p))
            / 2
        )
        return 12742 * math.asin(math.sqrt(a))


class API:
    def __init__(self, account_key):
        self.session = requests.Session()
        self.session.headers = {'AccountKey': account_key, 'accept': 'application/json', 'User-Agent': 'sg-transit'}  # todo
        self.cache = TTLCache(ttl=60 * 60, maxsize=120)
        try:
            with open('_cache.gson', 'rb') as f:
                cache = json.loads(gzip.decompress(f.read()))
                self.cache.update(cache)
        except (FileNotFoundError, json.JSONDecodeError, gzip.BadGzipFile):
            pass
        else:
            print('cachign')

    def _request(self, route, *, cache=True):
        if route not in self.cache or not cache:
            r = self.session.get(route)
            r.raise_for_status()

            if 'application/json' in r.headers['Content-Type']:
                data = r.json()
            elif 'text/html' in r.headers['Content-Type']:
                data = BeautifulSoup(r.text, 'html.parser')
            else:
                data = r.content

            if route.startswith(Routes.DATAMALL_BASE):
                data = data['value']
                while len(data) % 500 == 0:
                    r = self.session.get(route, params={'$skip': len(data)})
                    r.raise_for_status()
                    data.extend(r.json()['value'])

            self.cache[route] = data

        return self.cache[route]

    def get_bus_stops(self):
        data = self._request(Routes.BUS_STOPS)
        return list(map(BusStop, data))

    def get_bus_routes(self):
        stops = self.get_bus_stops()
        data = self._request(Routes.BUS_ROUTES)
        routes = {}
        for i in data:
            key = (i['ServiceNo'], i['Direction'])
            if key not in routes:
                routes[key] = BusRoute(
                    service_no=i['ServiceNo'],
                    direction=Direction(i['Direction']),
                    operator=i['Operator'],
                )

            try:
                stop = next(x for x in stops if x.code == i['BusStopCode'])
            except StopIteration:
                raise ValueError(f"Bus stop {i['BusStopCode']} not found")

            bus_route_stop = BusRouteStop(bus_stop=stop, distance=i['Distance'], route=routes[key])
            routes[key].stops.append(bus_route_stop)


        return routes.values()

    def get_fare_structures(self) -> Dict[FareType, list]:
        soup = self._request(Routes.FARE_STRUCTURE)
        tabs = soup.find_all('div', attrs={'class': 'tab-pane'})

        fare_structures = {}
        for ntab, tab in enumerate(tabs):
            try:
                fare_type = FareType(ntab)
            except ValueError:
                continue
            distances = []

            rows = tab.find('tbody').find_all('tr')
            for row in rows[2:]:
                distance, card, cash = (i.get_text() for i in row.find_all('td'))

                if 'up to' in distance.lower():
                    from_distance = 0
                    to_distance = float(distance.split(' ')[-2])
                elif 'over' in distance.lower():
                    from_distance = float(distance.split(' ')[-2])
                    to_distance = math.inf
                else:
                    from_distance, to_distance = (float(i.split(' ')[-2]) for i in distance.split(' - '))

                distances.append(FareStructureDistance(fare_type, from_distance, to_distance, int(card), int(cash)))

            fare_structures[fare_type] = distances

        return fare_structures

    def get_train_stations(self) -> Dict[str, List[TrainStation]]:
        train_stations = {}
        station_codes = self.get_train_station_codes()
        print(station_codes)

        if 'train_stations_shp' not in self.cache:
            data = self._request(Routes.TRAIN_STATIONS, cache=False)
            url = data[0]['Link']
            zip_data = self._request(url)
            self.cache['train_stations_shp'] = zip_data
            zip_stream = io.BytesIO(zip_data)
        else:
            zip_stream = self.cache['train_stations_shp']

        with zipfile.ZipFile(zip_stream) as zf:
            for fn in zf.namelist():
                if fn.endswith('.prj'):
                    prj_fn = fn
                if fn.endswith('.shp'):
                    shp_fn = fn
                if fn.endswith('.shx'):
                    shx_fn = fn
                if fn.endswith('.dbf'):
                    dbf_fn = fn

            projection = pyproj.Proj(zf.read(prj_fn).decode('utf-8'))

            with shapefile.Reader(
                shp=io.BytesIO(zf.read(shp_fn)),
                shx=io.BytesIO(zf.read(shx_fn)),
                dbf=io.BytesIO(zf.read(dbf_fn))
            ) as shps:
                # shp = shapefile.Reader(f)
                for sr in shps.iterShapeRecords():
                    s = sr.shape
                    lon1, lat1 = projection(*s.bbox[2:], inverse=True, errcheck=True)
                    lon2, lat2 = projection(*s.bbox[:2], inverse=True, errcheck=True)
                    center_lon = (lon1 + lon2) / 2
                    center_lat = (lat1 + lat2) / 2
                    r = sr.record
                    if 'MRT STATION' not in r.STN_NAM_DE and 'LRT STATION' not in r.STN_NAM_DE:
                        continue

                    name = ' '.join(r.STN_NAM_DE.split(' ')[:-2])  # remove MRT STATION/LRT STATION
                    try:
                        fmt_name = next(k for k in station_codes.keys() if k.lower() == name.lower())
                    except StopIteration:
                        continue

                    yield TrainStation(
                        codes=station_codes[fmt_name],
                        station_type=StationType[r.TYP_CD_DES],
                        name=fmt_name,
                        latitude=center_lat,
                        longitude=center_lon,
                    )


    def get_train_station_codes(self) -> Dict[str, List[str]]:
        """Returns Dict[station_name, List[station_code]]]"""
        zip_data = self._request(Routes.TRAIN_STATION_NAMES)
        zip_stream = io.BytesIO(zip_data)

        train_station_names = defaultdict(list)

        with zipfile.ZipFile(zip_stream) as zf:
            assert len(zf.namelist()) == 1
            fn = zf.namelist()[0]
            with zf.open(fn) as f:
                book = xlrd.open_workbook(file_contents=f.read())
                sheet = book.sheet_by_index(0)
                data = []
                for cols in range(1, sheet.nrows):
                    code, name, *_ = sheet.row_values(cols)
                    train_station_names[name].append(code)

        return dict(train_station_names)

    def close(self):
        cache_data = {k: v for k, v in self.cache.items() if not isinstance(v, bytes)}
        with open('_cache.gson', 'wb') as f:
            f.write(gzip.compress(json.dumps(cache_data).encode('utf8')))

        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
