# # -*- coding: utf-8 -*-
# city GML = City Geography Markup Language
import copy
import os
import math
import geopandas as gpd
from suncalc import get_position
import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
from cjio import cityjson as cj
import pandas as pd
from shapely.geometry import Polygon, Point, LineString
import itertools
from scipy.spatial import Delaunay, distance
import sys
import pyproj
import random
import pickle
import progressbar
from pathlib import Path


# Get the directory of the current script
project_dir = Path(__file__).resolve().parent

"""input information"""

cityjson_file=sys.argv[1] if len(sys.argv) > 1 else "684_5334.json"
desired_building_key = sys.argv[2] if len(sys.argv) > 2 else 'DEBY_LOD2_4615927'
examplery_hour=int(sys.argv[3]) if len(sys.argv) > 3 else None


# Construct the path to the 3D model file
model_file = project_dir / "input" / "3DModel" / f"{cityjson_file}"
climate_data=project_dir / "input" / "weather"
Baujahr_file_path=project_dir / "input" / "ConstructionYear_Map" / "LHM_PLAN_Gebaeudedaten2014.shp"







Baujahr = gpd.read_file(Baujahr_file_path)



def lineseg_dist(p, a, b):

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p - a, d)

    return np.hypot(h, np.linalg.norm(c))


class building_footprints():
    def create_footprints(model_file):

        buildings_function = {'Residential': '31001_1000', 'Business or commercial buildings': '31001_2000', 'Youth hostel': '31001_2072', 'Parking garage': '31001_2461', 'Underground car park': '31001_2465',
                              'Water container': '31001_2513', 'converter': '31001_2523', 'Buildings for public purposes': '31001_3000', 'Rathaus': '31001_3012', 'District_adminstration': '31001_3017',
                              'District government': '31001_3018', 'Unclassified': '31001_9998', 'Police': '31001_3071', 'Hospital': '31001_3051', 'Church': '31001_3041', 'Kaserne': '31001_3073', 'Gebäude für Bildung und Forschung': '31001_3020',
                              'Überdachung': '51009_1610', 'Schloss': '31001_3031', 'Burg, Festung': '31001_3038', 'Synagoge': '31001_3042', 'Moschee': '31001_3046', 'Kinderkrippe, Kindergarten, Kindertagesstätte': '31001_3065',
                              'Justizvollzugsanstalt': '31001_3075', 'Bahnhofsgebäude': '31001_3091', 'Touristisches Informationszentrum': '31001_3290', 'Kloster': '31001_3048', 'Tempel': '31001_3047', 'Feuerwehr': '31001_3072', 'Sanatorium': '31001_3242'}

        building_dict_keys = list(buildings_function.keys())
        building_dict_values = list(buildings_function.values())

       
        building_id = []
        height = []
        GroundSurfaceArea_dataSet = []
        RoofSurfaceArea_dataSet = []
        GroundSurfaceArea_Calculated = []
        tile_building_footprint = []
        geo = []
        building_type = []
        cm = cj.load(model_file)
        cm.set_epsg('25832')
        tranformation_object = cm.transform
        buildings_parts = cm.get_cityobjects(type=['building', 'buildingpart'])
        for i in range(len(buildings_parts)):

            building_of_id_i = buildings_parts[list( buildings_parts.keys())[i]]
            building_id.append(building_of_id_i.id)
            height.append(building_of_id_i.attributes['measuredHeight'])
            function_code = building_of_id_i.attributes['function']

            try:
                position = building_dict_values.index(function_code)
                building_type.append(building_dict_keys[position])
            except ValueError:
                building_type.append('function missing')

            geom = building_of_id_i.geometry[0]
            geom = geom.transform(tranformation_object)
            floor = geom.get_surfaces(type='GroundSurface')
            GroundSurfaceArea_dataSet.append(float(floor[list(floor.keys())[0]]['attributes']['Flaeche']))

            floor_boundaries = []
            floor_boundaries_2d = []
            for r in floor.values():
                floor_boundaries.append(geom.get_surface_boundaries(r))
            floor_boundaries = next(floor_boundaries[0])

            for i in range(len(floor_boundaries[0])):
                floor_boundaries_2d.append(list(floor_boundaries[0][i]))
            tile_building_footprint.append(floor_boundaries_2d)

            Roof = geom.get_surfaces(type='RoofSurface')
            RoofSurfaceArea_building = []
            for r_num in range(len(Roof.values())):
                RoofSurfaceArea_building.append(float(Roof[list(Roof.keys())[r_num]]['attributes']['Flaeche']))
            RoofSurfaceArea_dataSet.append(RoofSurfaceArea_building)

        for i in range(len(buildings_parts)):
            geo.append(Polygon(tile_building_footprint[i]))
            GroundSurfaceArea_Calculated.append(geo[i].area)


        max_height = max(height)
        data = {'id': building_id, 'geometry': geo,'function': building_type, 'height': height}
        gdf = gpd.GeoDataFrame(data, crs="EPSG:25832")
        res_with_baujahr = gpd.sjoin(gdf, Baujahr, how="inner", op='intersects')
        res_with_baujahr = res_with_baujahr.reset_index()

        res_with_baujahr["year_class"] = len(res_with_baujahr)*["None"]
        for i in range(len(res_with_baujahr)):
            if res_with_baujahr.loc[i, "G34"] >= 1860 and res_with_baujahr.loc[i, "G34"] <= 1918:
                res_with_baujahr.loc[i, "year_class"] = "1860-1918"
            elif res_with_baujahr.loc[i, "G34"] >= 1919 and res_with_baujahr.loc[i, "G34"] <= 1948:
                res_with_baujahr.loc[i, "year_class"] = "1919-1948"
            elif res_with_baujahr.loc[i, "G34"] >= 1949 and res_with_baujahr.loc[i, "G34"] <= 1957:
                res_with_baujahr.loc[i, "year_class"] = "1949-1957"
            elif res_with_baujahr.loc[i, "G34"] >= 1958 and res_with_baujahr.loc[i, "G34"] <= 1968:
                res_with_baujahr.loc[i, "year_class"] = "1958-1968"
            elif res_with_baujahr.loc[i, "G34"] >= 1969 and res_with_baujahr.loc[i, "G34"] <= 1978:
                res_with_baujahr.loc[i, "year_class"] = "1969-1978"
            elif res_with_baujahr.loc[i, "G34"] >= 1979 and res_with_baujahr.loc[i, "G34"] <= 1983:
                res_with_baujahr.loc[i, "year_class"] = "1979-1983"
            elif res_with_baujahr.loc[i, "G34"] >= 1984 and res_with_baujahr.loc[i, "G34"] <= 1994:
                res_with_baujahr.loc[i, "year_class"] = "1984-1994"
            elif res_with_baujahr.loc[i, "G34"] >= 1995 and res_with_baujahr.loc[i, "G34"] <= 2001:
                res_with_baujahr.loc[i, "year_class"] = "1995-2001"
            elif res_with_baujahr.loc[i, "G34"] >= 2002 and res_with_baujahr.loc[i, "G34"] <= 2009:
                res_with_baujahr.loc[i, "year_class"] = "2002-2009"
            elif res_with_baujahr.loc[i, "G34"] >= 2010 and res_with_baujahr.loc[i, "G34"] <= 2016:
                res_with_baujahr.loc[i, "year_class"] = "2010-2016"
            else:
                res_with_baujahr.loc[i, "year_class"] = "None"

        res_with_baujahr = res_with_baujahr.reset_index()
        res_with_baujahr = res_with_baujahr[['id', 'year_class', 'geometry', 'function', 'height']]
        return res_with_baujahr, max_height, buildings_parts


class get_sun_position():
    def sun_position(max_height, region):

        def hour_of_year(dt):
            beginning_of_year = datetime.datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)
            return (dt - beginning_of_year).total_seconds() // 3600

        lon=tile_footprints.to_crs('EPSG:4326').geometry.values[0].centroid.x
        lat=tile_footprints.to_crs('EPSG:4326').geometry.values[0].centroid.y
        def generate_datetimes(date_from_str='2022-01-01', days=365):
            date_from = datetime.datetime.strptime(date_from_str, '%Y-%m-%d')
            for hour in range(24*days):
                yield date_from + datetime.timedelta(hours=hour)

        timestamp = []
        for date in generate_datetimes():
            timestamp.append(date.strftime('%Y-%m-%d %H:%M:%S'))


        date_format = '%Y-%m-%d %H:%M:%S'
        timestamp = [datetime.datetime.strptime(date, date_format) for date in timestamp]
        tz = pytz.timezone('CET')
       
        timestamp = [x.replace(tzinfo=tz) for x in timestamp]
        timestamp = [x.astimezone(tz=datetime.timezone.utc) for x in timestamp]

        sun_loc = [get_position(date, lon, lat) for date in timestamp]


        azimuth = [sun_loc[i]['azimuth']*180 /np.pi for i in range(len(sun_loc))]
        altitude = [sun_loc[i]['altitude']*180 / np.pi for i in range(len(sun_loc))]
        tan_altitude = [np.tan(sun_loc[i]['altitude'])
                        for i in range(len(sun_loc))]
        hour = [hour_of_year(timestamp[i]) for i in range(24*365)]

        sun_position = pd.DataFrame(np.array([hour, azimuth, altitude, tan_altitude]).T, columns=['hour', 'azimuth', 'altitude', 'tan_altitude'])
        sun_position['maximum_shading_line'] = max_height/sun_position['tan_altitude']

        return sun_position


tile_footprints, max_height, buildings_cityjson = building_footprints.create_footprints(model_file)
residential_footprints = tile_footprints[( tile_footprints['function'] == 'Residential')]
sun_position = get_sun_position.sun_position(max_height,tile_footprints)
sun_alt_azimuth = sun_position[['altitude','azimuth']]
sun_position.drop(sun_position[sun_position['altitude'] < 5].index, inplace=True)


class group_sun_position_angeles():
    """This class returns a list of examplery sun positions to reduce simulation time

    a single calculation loop is performed for a group of hours where the difference 
    in sun azimuth angle is less than 3 deg 
    and the difference in alititude angle is less than 3 deg """

    def group_similar_rows(self, df, reference_row, azimuth_threshold, altitude_threshold):

        # Calculate the absolute difference between each row in the dataframe and the first row (reference row)
        df['azimuth_diff'] = (df['azimuth'] - reference_row['azimuth']).abs()
        df['altitude_diff'] = ( df['altitude'] - reference_row['altitude']).abs()

        # Keep only the rows where the sum of differences is greater than the threshold
        different_sun_positions = df[(df['azimuth_diff'] > azimuth_threshold) | (df['altitude_diff'] > altitude_threshold)]
        single_sun_position_group = df[(df['azimuth_diff'] < azimuth_threshold) & (df['altitude_diff'] < altitude_threshold)]

        return different_sun_positions, single_sun_position_group

    def determine_hours_to_be_calculated(self):

        df = sun_position.sort_values('azimuth')

        reference_rows = []

        reference_row = df.iloc[0]
        reference_rows = [reference_row]

        # define thresholds for the different between sun position angles within a group
        azimuth_threshold = 3
        altitude_threshold = 3

        different_sun_positions, single_sun_position_group = self.group_similar_rows(df, reference_row, azimuth_threshold, altitude_threshold)
        filtered_dfs_list = [different_sun_positions]
        length = [len(single_sun_position_group)]
        deleted = [single_sun_position_group]

        iterations = []
        iterations_exc = []
        for i in range(len(different_sun_positions)):
            try:
                iterations.append(i)
                reference_row = different_sun_positions.iloc[0]
                different_sun_positions, single_sun_position_group = self.group_similar_rows(
                    different_sun_positions, reference_row, azimuth_threshold, altitude_threshold)
                # print(len(different_sun_positions))
                filtered_dfs_list.append(different_sun_positions)
                reference_rows.append(reference_row)
                length.append(len(single_sun_position_group))
                deleted.append(single_sun_position_group)
            except:
                iterations_exc.append(i)
                pass

        hours_to_be_calculated = pd.concat(
            [pd.concat(reference_rows, axis=1).T, filtered_dfs_list[-1]])

        print(len(reference_rows))
        print(sum(length))
        len(iterations_exc)

        return hours_to_be_calculated, deleted


hours_to_be_calculated, hours_deleted = group_sun_position_angeles().determine_hours_to_be_calculated()


class building_observer_points:

    def angle_between_vectors(self, vector1, vector2):
        # Calculate the dot product of the two vectors
        dot_product = np.dot(vector1, vector2)

        # Calculate the magnitudes (lengths) of the vectors
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        # Calculate the cosine of the angle
        cosine_theta = dot_product / (magnitude1 * magnitude2)

        # Calculate the angle in radians
        theta_rad = np.arccos(np.clip(cosine_theta, -1.0, 1.0))

        # Convert the angle to degrees
        theta_deg = np.degrees(theta_rad)

        return theta_deg

    def calculate_distance(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5

    def find_connected_triangles(self, vertices):
        triangles = []

        for comb in itertools.combinations(vertices, 3):
            distances = [self.calculate_distance(comb[0], comb[1]),
                         self.calculate_distance(comb[1], comb[2]),
                         self.calculate_distance(comb[0], comb[2])]
            if all(distance > 1e-5 for distance in distances):
                triangles.append(comb)

        return triangles
    """ First method for calculating 3D polygon area  - Not using it """

    def compute_3D_polygon_area(self, points):

        if (len(points) < 3):
            return 0.0
        P1X, P1Y, P1Z = points[0][0], points[0][1], points[0][2]
        P2X, P2Y, P2Z = points[1][0], points[1][1], points[1][2]
        P3X, P3Y, P3Z = points[2][0], points[2][1], points[2][2]
        a = pow(((P2Y-P1Y)*(P3Z-P1Z)-(P3Y-P1Y)*(P2Z-P1Z)), 2) + pow(((P3X-P1X)*(P2Z-P1Z) - (P2X-P1X)*(P3Z-P1Z)), 2) + pow(((P2X-P1X)*(P3Y-P1Y)-(P3X-P1X)*(P2Y-P1Y)), 2)
        cosnx = ((P2Y-P1Y)*(P3Z-P1Z)-(P3Y-P1Y)*(P2Z-P1Z))/(pow(a, 1/2))
        cosny = ((P3X-P1X)*(P2Z-P1Z)-(P2X-P1X)*(P3Z-P1Z))/(pow(a, 1/2))
        cosnz = ((P2X-P1X)*(P3Y-P1Y)-(P3X-P1X)*(P2Y-P1Y))/(pow(a, 1/2))
        s = cosnz*((points[-1][0])*(P1Y)-(P1X)*(points[-1][1])) + cosnx*((points[-1][1])*(
            P1Z)-(P1Y)*(points[-1][2])) + cosny*((points[-1][2])*(P1X)-(P1Z)*(points[-1][0]))
        for i in range(len(points)-1):
            p1 = points[i]
            p2 = points[i+1]
            ss = cosnz*((p1[0])*(p2[1])-(p2[0])*(p1[1])) + cosnx*((p1[1]) * (p2[2])-(p2[1])*(p1[2])) + cosny*((p1[2])*(p2[0])-(p2[2])*(p1[0]))
            s += ss

        s = abs(s/2.0)

        return s

    """ Second method for calculating 3D polygon area  - USED """

    def poly_area(self, poly, unit_normal):
        if len(poly) < 3:  # not a plane - no area
            return 0
        total = [0, 0, 0]
        N = len(poly)
        for i in range(N):
            vi1 = poly[i]
            vi2 = poly[(i+1) % N]
            prod = np.cross(vi1, vi2)
            total[0] += prod[0]
            total[1] += prod[1]
            total[2] += prod[2]
        result = np.dot(total, unit_normal)
        return abs(result/2)

    def create_points(self, model_file, desired_building_key):

        building_id = []
        height = []
        GroundSurfaceArea_dataSet = []
        storeysAboveGround_dataSet =[]
        tile_building_footprint = []
        
        cm = cj.load(model_file)
        cm.set_epsg('25832')
        tranformation_object = cm.transform
        buildings_parts=cm.get_cityobjects(id=[desired_building_key])


        dic = []
        dic_roof = []

        for i in range(len(buildings_parts)):
            building_of_id_i = buildings_parts[list(buildings_parts.keys())[i]]
            try:
                storeysAboveGround_dataSet.append(int(building_of_id_i.attributes["storeysAboveGround"]))
            except:
                storeysAboveGround_dataSet.append(
                    int(building_of_id_i.attributes["measuredHeight"]/3.5))
            touching_polygons = tile_footprints[tile_footprints.touches(
                (tile_footprints[tile_footprints['id'] == building_of_id_i.id]['geometry']).unary_union)]
            touching_segments = gpd.GeoDataFrame(geometry=[])
            given_polygon = tile_footprints[tile_footprints['id']
                                            == building_of_id_i.id]['geometry']
            # Iterate through the touching polygons
            for index, touching_polygon in touching_polygons.iterrows():
                intersection = touching_polygon['geometry'].intersection(
                    given_polygon.unary_union)
                if intersection.geom_type == 'LineString':
                    touching_segments = pd.concat([touching_segments, pd.DataFrame([{'geometry': intersection}])], ignore_index=True)

                if intersection.geom_type == 'MultiLineString':

                    touching_segments = pd.concat([touching_segments, pd.DataFrame(
                        [{'geometry': intersection}])], ignore_index=True)
                    touching_segments = touching_segments.explode(ignore_index=True)

            building_id.append(building_of_id_i.id)
            height.append(building_of_id_i.attributes['measuredHeight'])
            geom = building_of_id_i.geometry[0]
            geom = geom.transform(tranformation_object)
            floor = geom.get_surfaces(type='GroundSurface')
            GroundSurfaceArea_dataSet.append(float(floor[list(floor.keys())[0]]['attributes']['Flaeche']))
            floor_boundaries = []
            for r in floor.values():
                floor_boundaries.append(geom.get_surface_boundaries(r))
            floor_boundaries = next(floor_boundaries[0])
            ground_elevation = floor_boundaries[0][0][2]

            wall = geom.get_surfaces(type='WallSurface')
            geom.get_vertices()
            wall_boundaries = []

            for r in wall.values():
                wall_boundaries.append(geom.get_surface_boundaries(r))

            wall_boundaries = [next(wall_boundaries[i])
                               for i in range(len(wall_boundaries))]
            walls_list = list(wall.keys())

            # """ getting width of the wall at certian elevation """
            vertices_3d = wall_boundaries[0][0]
            polygon_3d = Polygon(wall_boundaries[0][0])


            all_wall_areas = []
            all_walls_col = []
            all_walls_ground = []
            all_orient = []
            all_wall_windows = []
            all_wall_glazing_area = []
            observer_points = []
            for index, wall_boundary in enumerate(wall_boundaries):

                polygon_vertices = wall_boundary[0]

                """ check if the wall is attached to another building wall """
                polygon_3d = Polygon(polygon_vertices)
                line_strings_poly = [LineString(polygon_vertices[i:i+2]) for i in range(len(polygon_vertices) - 1)]
                line_strings_poly.append(LineString([polygon_vertices[len(polygon_vertices)-1], polygon_vertices[0]]))
                wall_ground_elevation = float(wall[walls_list[index]]['attributes']['Z_MIN_ASL'])

                vertical_line_strings1 = [line for line in line_strings_poly if (line.coords[0][0] == line.coords[1][0] and line.coords[0][1] == line.coords[1][1])]
                ground_line = [line for line in line_strings_poly if line.coords[0][2] == wall_ground_elevation and line.coords[1][2] == wall_ground_elevation]

                common_line_strings = []
                for line1 in line_strings_poly:
                    for line2 in touching_segments['geometry']:
                        if line1.equals(line2):
                            common_line_strings.append(line1)

                """ if wall is not free, don't generate point grid for this wall (terminate this for loop iteration)"""
                if len(common_line_strings) != 0:
                    continue

                elevations = [polygon_vertices[i][2] for i in range(len(polygon_vertices))]

                """
                skip the wall if the height is less than 0.5 m (some wall are divided into small segments / 
                some objects are missclassified as buildings) 
                """

                if max(elevations) - min(elevations) < 1.0:
                    continue

                non_collinear_vertices = self.find_connected_triangles(
                    polygon_vertices)

                if len(non_collinear_vertices) > 1:
                    non_collinear_vertices = non_collinear_vertices[1]
                else:
                    non_collinear_vertices = non_collinear_vertices[0]

                v1 = np.array(non_collinear_vertices[1])-np.array(non_collinear_vertices[0])
                v2 = np.array(non_collinear_vertices[2])-np.array(non_collinear_vertices[0])
                normal_vector = np.cross(v1, v2)
                normalized_normal_vector = normal_vector / np.linalg.norm(normal_vector)
                theta = np.arctan2(normalized_normal_vector[1], normalized_normal_vector[0])
                theta = np.degrees(theta)
                wall_orientation = (theta+360) % 360  # theta

                try:
                    wall_surface_area = self.poly_area(polygon_vertices, normalized_normal_vector)
                except:
                    wall_surface_area = 'error'

                """shifting the veritices by buffer distance from the wall for two reasons: 
                    1- visual 
                    2- to avoid that the point is slightly behind the wall of the building itself. Otherwise, later the code will consider that 
                    the point is shadowed by the building itself. """

                buffer_distance = 0.5  # meters
                polygon_vertices = [[polygon_vertices[i][0] + buffer_distance * np.cos(np.radians(wall_orientation)), polygon_vertices[i][1] + buffer_distance * np.sin(
                    np.radians(wall_orientation)), polygon_vertices[i][2]] for i in range(len(polygon_vertices))]



                walls_list = list(wall.keys())
                wall_area = float(wall[walls_list[index]]['attributes']['Flaeche'])
                wall_height = float(wall[walls_list[index]]['attributes']['Z_MAX'])
                wall_ground_elevation = float(wall[walls_list[index]]['attributes']['Z_MIN_ASL'])

                try:
                    storeysAboveGround = int(building_of_id_i.attributes['storeysAboveGround'])
                except:
                    if building_of_id_i.attributes['measuredHeight'] < 3.5:
                        storeysAboveGround = 1
                    else:
                        storeysAboveGround = int(
                            building_of_id_i.attributes['measuredHeight']/3.5)
                        
                WWR = 0.3
                if wall_height >= 3.5 and wall_height/storeysAboveGround >= 1:
                    storey_height = wall_height/storeysAboveGround
                    window_area = (wall_area/storeysAboveGround)*WWR

                elif wall_height <= 3.5 and wall_height >= 1:
                    storey_height = wall_height
                    window_area = (wall_area/1)*WWR

                elif wall_height >= 3.5 and wall_height/storeysAboveGround < 1:
                    if wall_height/2.5 < 2:
                        storeys = int(wall_height/2.5)
                        storey_height = wall_height/storeys
                        window_area = (wall_area/storeys)*WWR
                else:
                    storey_height = wall_height
                    window_area = (wall_area/1)*WWR

                
                window_height = storey_height*WWR
                rounded_integer = round(window_height)
                rounded_half = round(window_height * 2) / 2
                diff_integer = abs(window_height - rounded_integer)
                diff_half = abs(window_height - rounded_half)

                if diff_integer <= diff_half:
                    window_height = rounded_integer

                else:
                    window_height = rounded_half

                window_width = window_area/window_height

                rounded_integer = round(window_width)
                rounded_half = round(window_width * 2) / 2
                diff_integer = abs(window_width - rounded_integer)
                diff_half = abs(window_width - rounded_half)

                if diff_integer <= diff_half:
                    window_width = rounded_integer
                else:
                    window_width = rounded_half

                if window_width >= wall_area/wall_height:
                    window_width = window_width-0.5
                    # continue

                if window_width >= 1.5:
                    if window_width % 1.5 == 0:
                        number_of_windows_per_floor = window_width/1.5
                        single_window_H = window_height
                        single_window_W = 1.5
                    else:
                        number_of_windows_per_floor = int(window_width/1)
                        single_window_H = window_height
                        single_window_W = 1.0

                else:
                    single_window_H = window_height
                    single_window_W = window_width-0.2


                floor_min_elevation = [wall_ground_elevation+storey_height*i for i in range(storeysAboveGround)]
                window_low_elevation = [item+0.75 for item in floor_min_elevation]
                window_top_elevation = [item+single_window_H for item in window_low_elevation]

                """ getting width of the wall at certian elevation """
                vertices_3d = polygon_vertices 


                """ check if the wall is attached to another building wall """
                polygon_3d = Polygon(vertices_3d)
                line_strings_poly = [LineString(vertices_3d[i:i+2]) for i in range(len(vertices_3d) - 1)]
                line_strings_poly.append(LineString( [vertices_3d[len(vertices_3d)-1], vertices_3d[0]]))

                vertical_line_strings1 = [line for line in line_strings_poly if (line.coords[0][0] == line.coords[1][0] and line.coords[0][1] == line.coords[1][1])]
                ground_line = [line for line in line_strings_poly if line.coords[0][2] == wall_ground_elevation and line.coords[1][2] == wall_ground_elevation]

                # """ end free wall check """

                if len(ground_line) != 0:
                    windowpoints_building = []
                    windowpoints_building_matrix = []
                    for wle in range(len(window_low_elevation)):
                        modified_line = LineString(
                            [(x, y, window_low_elevation[wle]) for x, y, z in ground_line[0].coords])

                        def flatten(lst):
                            return [item for sublist in lst for item in (flatten(sublist) if isinstance(sublist, list) else [sublist])]

                        elevations = [[line.coords[0][2], line.coords[1][2]]
                                      for line in vertical_line_strings1]
                        elevations = flatten(elevations)

                        if wall_ground_elevation in elevations:
                            elevations = [x for x in elevations if x != wall_ground_elevation]
                        else:
                            continue
                        """ handling non-horizontal wall top"""
                        if window_top_elevation[wle] > min(elevations):
                            
                            elev_diff=window_top_elevation[wle]- min(elevations)
                            window_top_elevation[wle]=window_top_elevation[wle]-elev_diff
                            
                            wall_height=window_top_elevation[wle]-wall_ground_elevation
                            
                            if wall_height >= 3.5:
                                if wall_height/2.5 > 2:
                                    storeys = int(wall_height/2.5)
                                    storey_height = wall_height/storeys
                                    window_area = (wall_area/storeys)*WWR
                            else:
                                storey_height = wall_height
                                
                            
                            floor_min_elevation = [
                                wall_ground_elevation+storey_height*i for i in range(storeysAboveGround)]
                            window_low_elevation = [
                                item+0.75 for item in floor_min_elevation]
                            window_top_elevation = [
                                item+single_window_H for item in window_low_elevation]
                                


                        filtered_list = [value for value in window_low_elevation if value < min(elevations)+single_window_H]

                        wall_width = ground_line[0].length

                        if wall_width < 0.5:
                            windowpoints_building_matrix = []
                            wall_glazing_area = 0
                            continue
                        
                        elif wall_width > 1.5 and wall_width/1.5 >= 1:
                            single_window_W = 1.5
                            number_of_windows_per_floor = int(wall_width/1.5)
                        
                        elif wall_width < 1.5 and wall_width/1.5 < 1:
                            single_window_W = wall_width-0.3
                            number_of_windows_per_floor = 1



                        total_spacing = wall_width -(single_window_W*number_of_windows_per_floor)

                        single_spacing = total_spacing /((number_of_windows_per_floor-1)+2)
                        if single_spacing > 0:
                            windows = int(number_of_windows_per_floor) * [single_window_W]
                        elif single_spacing < 0:
                            windows = int( number_of_windows_per_floor) * [single_window_W]

                        resulting_list = [val for pair in zip(windows, [single_spacing] * (len(windows) - 1)) for val in pair]
                        resulting_list.append(windows[-1])
                        distance_to_move = [single_spacing] + resulting_list + [single_spacing]


                        positions = []

                        # Create a Point at the starting position
                        current_position = Point(ground_line[0].coords[0][:2])

                        # Initialize a variable to keep track of the remaining distance
                        remaining_distance = distance_to_move

                        for j in range(len(distance_to_move)):
                            # Iterate through the LineString's coordinates
                            positions.append(current_position)
                            for i in range(len(modified_line.coords) - 1):
                                segment_start = current_position
                                segment_end = Point(
                                    ground_line[0].coords[1][:2])

                                # Calculate the length of the LineString segment
                                segment_length = segment_start.distance(
                                    segment_end)
                                # Calculate the direction vector and move the remaining distance along the segment
                                direction_vector = (
                                    segment_end.x - segment_start.x, segment_end.y - segment_start.y)
                                direction_unit_vector = (
                                    direction_vector[0] / segment_length, direction_vector[1] / segment_length)
                                moved_vector = (
                                    direction_unit_vector[0] * remaining_distance[j], direction_unit_vector[1] * remaining_distance[j])
                                current_position = Point(
                                    segment_start.x + moved_vector[0], segment_start.y + moved_vector[1])
                                if j == len(distance_to_move):
                                    positions.append(current_position)

                            sum(distance_to_move)+single_spacing

                        line_strings = [LineString([p1, p2]) for p1, p2 in zip(
                            positions, positions[1:])]
                        lengths = [line.length for line in line_strings]

                        line_strings_3d_1 = [LineString([Point(p[0], p[1], window_low_elevation[wle]) for p in line.coords]) for line in line_strings]
                        line_strings_3d_2 = [LineString([Point(p[0], p[1], window_top_elevation[wle]) for p in line.coords]) for line in line_strings]

                        vertical_line_strings = []

                        # Iterate through the horizontal LineStrings and create vertical LineStrings
                        for line1, line2 in zip(line_strings_3d_1, line_strings_3d_2):
                            # Swap the x and y coordinates to create a vertical LineString
                            vertical_line = LineString([(line1.coords[0][0], line1.coords[0][1], line1.coords[0][2]), (line2.coords[0][0], line2.coords[0][1], line2.coords[0][2])])
                            vertical_line_strings.append(vertical_line)
                        vertical_line_strings.append(LineString([(line1.coords[1][0], line1.coords[1][1], line1.coords[1][2]), (line2.coords[1][0], line2.coords[1][1], line2.coords[1][2])]))

                        windowlines = []
                        for i in range(1, len(line_strings_3d_1), 2):
                            windowline = [vertical_line_strings[i], line_strings_3d_2[i],
                                          line_strings_3d_1[i], vertical_line_strings[i+1]]
                            windowlines.append(windowline)

                        windowpoints = []
                        # Iterate through the LineStrings and extract distinct coordinates
                        for wl in range(len(windowlines)):
                            # Create a set to store unique points
                            unique_points = set()
                            for line in windowlines[wl]:
                                unique_points.update(line.coords)
                            unique_points = list(unique_points)
                            
                            # Remove the first element and store it in a variable
                            first_element = unique_points.pop(0)

                            # Insert the first element at the third position (index 2)
                            unique_points.insert(2, first_element)

                            windowpoints.append(unique_points)

                        windowpoints_building_matrix.append(windowpoints)
                        total_number_of_windows_per_wall = len(windowpoints_building_matrix)*len(windowpoints_building_matrix[0])
                        wall_glazing_area = total_number_of_windows_per_wall*single_window_W*0.75

                        windowpoints = flatten(windowpoints)

                        windowpoints_building.append(windowpoints)

                else:
                    windowpoints_building_matrix = []
                    wall_glazing_area = 0
                wground_points = []
                grid_points = []
                grid_points_vert = []

                edge_columns = []
                edge_ground = []

                for i in range(len(polygon_vertices)):
                    if i < len(polygon_vertices)-1:
                        vector = np.array(
                            polygon_vertices[i+1])-np.array(polygon_vertices[i])

                        if (polygon_vertices[i+1][0] - polygon_vertices[i][0] == 0 and polygon_vertices[i+1][1] - polygon_vertices[i][1] == 0):
                            vertical_move = 'yes'
                        else:
                            vertical_move = 'No'

                    else:
                        vector = np.array(
                            polygon_vertices[0])-np.array(polygon_vertices[i])

                        if (polygon_vertices[0][0] - polygon_vertices[i][0] == 0 and polygon_vertices[0][1] - polygon_vertices[i][1] == 0):
                            vertical_move = 'yes'
                        else:
                            vertical_move = 'No'

                    edge_x_values = [item[0] for item in polygon_vertices]

                    max_x = max(edge_x_values)
                    min_x = min(edge_x_values)

                    Vx, Vy, Vz = vector[0], vector[1], vector[2]
                    Vlen = np.sqrt(Vx**2+Vy**2+Vz**2)
                    
                    n = math.floor(Vlen/0.5)

                    if math.floor(Vlen/1) > 0:
                        stepx, stepy, stepz = Vx/n, Vy/n, Vz/n
                    else:
                        stepx, stepy, stepz = 0, 0, 0

                    col_point_edge = []
                    ground_points_edge = []
                    for z in range(n):

                        grid_point = polygon_vertices[i][0]+z*stepx, polygon_vertices[i][1] +  z*stepy, polygon_vertices[i][2]+z*stepz

                        if grid_point[2] != ground_elevation and vertical_move != 'yes' and grid_point[0] <= max_x-0.02 and grid_point[0] >= min_x+0.02:
                            grid_points.append(grid_point)

                        if vertical_move == 'yes':
                            grid_points_vert.append(grid_point)

                            if stepz < 0 and z < n-1:
                                column_point = grid_point
                                col_point_edge.append(column_point)

                            if stepz < 0 and z == n-1:
                                wground_points.append(grid_point)
                                ground_points_edge.append(grid_point)

                            if stepz > 0 and z == 0:
                                wground_points.append(grid_point)
                                ground_points_edge.append(grid_point)
                            if stepz > 0 and z > 0:
                                column_point = grid_point
                                col_point_edge.append(column_point)

                    if len(col_point_edge) != 0:
                        edge_columns.append(col_point_edge)
                    if len(ground_points_edge) != 0:
                        edge_ground.append(ground_points_edge)

                fill = []

                middle_cols = []
                middle_ground = []
                middle_orinet = []

                for point in grid_points:

                    # if vertical_move=='No': # and point[2] != ground_elevation # and point[2] not in [item[2] for item in polygon_vertices]:

                    if ground_elevation in [item[2] for item in polygon_vertices]:
                        n = math.floor((point[2]-ground_elevation)/0.5)

                        stepx, stepy, stepz = 0, 0, -0.5
                        list_col_points = []
                        middle_ground_points = []
                        for i in range(n):
                            grid_point = point[0]+i*stepx, point[1] + \
                                i*stepy, point[2] + \
                                i*stepz
                            if i < n-1:
                                column_point = grid_point
                                list_col_points.append((column_point))
                            if i == n-1:
                                wground_points.append(grid_point)
                                middle_ground_points.append(grid_point)

                            fill.append(grid_point)

                        middle_cols.append(list_col_points)
                        middle_ground.append(middle_ground_points)
                        middle_orinet.append(wall_orientation)

                all_walls_ground.append(edge_ground+middle_ground)
                all_walls_col.append(edge_columns+middle_cols)

                all_orient.append(wall_orientation)
                all_wall_areas.append(wall_surface_area)
                all_wall_windows.append(windowpoints_building_matrix)
                all_wall_glazing_area.append(wall_glazing_area)

                grid_points = grid_points+grid_points_vert+fill
                observer_points = observer_points+grid_points

            """grid on roof surfaces"""

            Roof = geom.get_surfaces(type='RoofSurface')
            

            Roof_boundaries = []

            for r in Roof.values():

                Roof_boundaries.append(geom.get_surface_boundaries(r))

            Roof_boundaries = [next(Roof_boundaries[i]) for i in range(len(Roof_boundaries))]

            grid_points_all_roof_segments = []
            roof_segment_slope = []
            roof_segment_orientation = []
            roof_segment_surface_area = []
            
            for r in list(Roof.keys()):
                roof_segment_slope.append(float(geom.get_surfaces(type='RoofSurface')[r]['attributes']['Dachneigung']))
                roof_segment_orientation.append(float(geom.get_surfaces(type='RoofSurface')[r]['attributes']['Dachorientierung']))
                

            for r in range(len(Roof_boundaries)):
                roof_segment_surface_area.append(float(Roof[list(Roof.keys())[r]]['attributes']['Flaeche']))
                
                polygon_vertices_origin = Roof_boundaries[r][0]
                polygon_vertices = Roof_boundaries[r][0]
                flat = True
                elevations = [polygon_vertices[i][2]
                              for i in range(len(polygon_vertices))]
                if all(x == elevations[0] for x in elevations):
                    roof = flat
                else:
                    flat = not flat
                if flat:
                    slope_angle_degrees = 0
                    roof_orientation_degrees = 90
                    buffer_distance = 0.5  # meters
                    polygon_vertices = [[polygon_vertices[i][0] + buffer_distance * np.cos(np.radians(roof_orientation_degrees)), polygon_vertices[i][1] + buffer_distance * np.sin(
                        np.radians(roof_orientation_degrees)), polygon_vertices[i][2]] for i in range(len(polygon_vertices))]
                else:
                    vertices = np.array([sublist[:2]
                                        for sublist in polygon_vertices])
                    # Perform Delaunay triangulation
                    triangles = Delaunay(vertices)

                    # Extract the triangles as vertex indices
                    triangle_indices = triangles.simplices

                    # Extract the triangles as sets of vertices
                    triangles = [vertices[idx] for idx in triangle_indices]

                    triangles3d = []
                    for t in range(len(triangles)):
                        triangle3d = []
                        for vert in range(3):
                            triangles[t][vert].tolist()
                            triangle3d.append(
                                [sublist for sublist in polygon_vertices if sublist[:2] == triangles[t][vert].tolist()][0])
                        triangles3d.append(triangle3d)

                    V1 = np.array(triangles3d[0][0]) - np.array(triangles3d[0][1])
                    V2 = np.array(triangles3d[0][2]) - np.array(triangles3d[0][1])

                    normal_vector = np.cross(V1, V2)
                    normal_vector = normal_vector / np.linalg.norm(normal_vector)
                    vertical_direction = np.array([0, 0, 1])

                    # Calculate the angle between the normal vector and the vertical direction
                    slope = np.arccos(np.dot(normal_vector, vertical_direction))
                    # Convert the angle from radians to degrees
                    slope_angle_degrees = np.degrees(slope)
                    slope_angle_degrees = slope_angle_degrees

                    theta = np.arctan2(normal_vector[1], normal_vector[0])
                    theta = np.degrees(theta)
                    roof_orientation_degrees = (theta+360) % 360  # theta
                    roof_orientation_radians = np.radians(roof_orientation_degrees)

                    buffer_distance = 0.5  # meters
                    polygon_vertices = [[polygon_vertices[i][0] + buffer_distance * (np.cos(np.radians(roof_orientation_degrees)) * np.cos(np.radians(slope_angle_degrees))), polygon_vertices[i][1] + buffer_distance * (np.sin(
                        np.radians(roof_orientation_degrees)) * np.cos(np.radians(slope_angle_degrees))), polygon_vertices[i][2] + buffer_distance * np.sin(np.radians(slope_angle_degrees))] for i in range(len(polygon_vertices))]

                # Find the bounding box of the polygon.
                min_x = min(polygon_vertices, key=lambda p: p[0])[0]
                max_x = max(polygon_vertices, key=lambda p: p[0])[0]
                min_y = min(polygon_vertices, key=lambda p: p[1])[1]
                max_y = max(polygon_vertices, key=lambda p: p[1])[1]
                grid_spacing = 1
                if flat:

                    grid_points = []
                    # Create a grid of equally spaced points within the bounds of the polygon.
                    for x in np.arange(min_x, max_x + grid_spacing, grid_spacing):
                        for y in np.arange(min_y, max_y + grid_spacing, grid_spacing):
                            # Check if the point is inside the polygon using the winding number algorithm.
                            odd_nodes = False
                            j = len(polygon_vertices) - 1

                            for i in range(len(polygon_vertices)):
                                xi, yi, zi = polygon_vertices[i]
                                xj, yj, zj = polygon_vertices[j]

                                if yi < y and yj >= y or yj < y and yi >= y:
                                    if xi + (y - yi) / (yj - yi) * (xj - xi) < x:
                                        odd_nodes = not odd_nodes
                                j = i

                            if odd_nodes:
                                z = polygon_vertices[2][2]
                                grid_points.append([x, y, z])

                else:
                    vertices = np.array([sublist[:2]
                                        for sublist in polygon_vertices])
                    # Perform Delaunay triangulation
                    triangles = Delaunay(vertices)

                    # Extract the triangles as vertex indices
                    triangle_indices = triangles.simplices

                    # Extract the triangles as sets of vertices
                    triangles = [vertices[idx] for idx in triangle_indices]

                    triangles3d = []
                    for t in range(len(triangles)):
                        triangle3d = []
                        for vert in range(3):
                            triangles[t][vert].tolist()
                            triangle3d.append(
                                [sublist for sublist in polygon_vertices if sublist[:2] == triangles[t][vert].tolist()][0])
                        triangles3d.append(triangle3d)

                    grid_points = []
                    for tri3d in range(len(triangles3d)):
                        polygon_vertices = triangles3d[tri3d]

                        def distance_3d(point1, point2):
                            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

                        # Calculate the lengths of the sides of the triangle
                        side1 = distance_3d(
                            polygon_vertices[0], polygon_vertices[1])
                        side2 = distance_3d(
                            polygon_vertices[1], polygon_vertices[2])
                        side3 = distance_3d(
                            polygon_vertices[2], polygon_vertices[0])

                        segments_length = [side1, side2, side3]

                        # Determine the shortest side
                        shortest_side = segments_length.index(
                            min(segments_length))

                        if shortest_side == 0:
                            loop_vert = range(0, 1)
                        elif shortest_side == 1:
                            loop_vert = range(1, 2)
                        elif shortest_side == 2:
                            loop_vert = range(2, 3)

                        for i in loop_vert:
                            base_vertex = polygon_vertices[i]
                            if loop_vert == range(1, 2):
                                neighbor_vertex_1 = polygon_vertices[i+1]
                                neighbor_vertex_2 = polygon_vertices[i-1]
                            elif loop_vert == range(0, 1):
                                neighbor_vertex_1 = polygon_vertices[2]
                                neighbor_vertex_2 = polygon_vertices[i+1]
                            elif loop_vert == range(2, 3):
                                neighbor_vertex_1 = polygon_vertices[i-1]
                                neighbor_vertex_2 = polygon_vertices[0]

                            V1 = np.array(neighbor_vertex_1) - np.array(base_vertex)
                            V2 = np.array(neighbor_vertex_2) - np.array(base_vertex)

                        normal_vector = np.cross(V1, V2)
                        normal_vector = normal_vector / np.linalg.norm(normal_vector)
                        vertical_direction = np.array([0, 0, 1])

                        # Calculate the angle between the normal vector and the vertical direction
                        slope = np.arccos(np.dot(normal_vector, vertical_direction))
                        # Convert the angle from radians to degrees
                        slope_angle_degrees = np.degrees(slope)
                        slope_angle_degrees = slope_angle_degrees
                        

                        theta = np.arctan2(normal_vector[1], normal_vector[0])
                        theta = np.degrees(theta)
                        roof_orientation_degrees = (theta+360) % 360  # theta
                        roof_orientation_radians = np.radians(roof_orientation_degrees)

                        V1_len = np.sqrt(V1[0]**2+V1[1]**2+V1[2]**2)
                        num_points_u = math.floor(V1_len/grid_spacing)

                        V2_len = np.sqrt(V2[0]**2+V2[1]**2+V2[2]**2)
                        num_points_v = math.floor(V2_len/grid_spacing)
                        if num_points_v == 0:
                            num_points_v = 1
                        elif num_points_u == 0:
                            num_points_u = 1

                        A0 = np.array(base_vertex)
                        if num_points_u == 1 or num_points_v == 1:
                            num_points_u = 1
                            num_points_v = 1
                        else:
                            # min(num_points_u,num_points_v)
                            num_points_u = num_points_u
                            num_points_v = num_points_v

                        for u in np.linspace(0, 1, num_points_u):
                            for v in np.linspace(0, 1, num_points_v):
                                # Calculate the point on the surface using the equation
                                point = A0 + V1 * u + V2 * v

                                # Check if the point is inside the polygon using the winding number algorithm.
                                odd_nodes = False
                                j = len(polygon_vertices) - 1

                                for i in range(len(polygon_vertices)):
                                    xi, yi, zi = polygon_vertices[i]
                                    xj, yj, zj = polygon_vertices[j]

                                    if yi < point[1]+0.001 and yj >= point[1]+0.001 or yj < point[1]+0.001 and yi >= point[1]+0.001:
                                        if xi + (point[1]+0.001 - yi) / (yj - yi) * (xj - xi) < point[0]-0.001:
                                            odd_nodes = not odd_nodes

                                    j = i
                                if odd_nodes:
                                    grid_points.append(point)


                grid_points_all_roof_segments.append(grid_points)

            """ end roof points """

            observer_points = [Point(x) for x in observer_points]
            floor_boundaries = []
            floor_boundaries_2d = []
            for r in floor.values():
                floor_boundaries.append(geom.get_surface_boundaries(r))
            floor_boundaries = next(floor_boundaries[0])
            for i in range(len(floor_boundaries[0])):
                floor_boundaries_2d.append(list(floor_boundaries[0][i][:2]))
            tile_building_footprint.append(floor_boundaries_2d)

            dic.append({'orientation': all_orient, 'columns_points': all_walls_col,
                       'ground_points': all_walls_ground,
                        'wall_surface_area': all_wall_areas,
                        'windows_per_wall': all_wall_windows,
                        'all_wall_glazing_area': all_wall_glazing_area})

            dic_roof.append({'orientation': roof_segment_orientation,
                             'Roof_surface_area': roof_segment_surface_area,
                            'slope': roof_segment_slope, 'points': grid_points_all_roof_segments})

        building_grid = {'id': building_id, 'building_height': height, 'storeysAboveGround': storeysAboveGround_dataSet,
                         'GroundSurfaceArea_dataSet': GroundSurfaceArea_dataSet,  'walls': dic, 'roof': dic_roof}
        
        return building_grid


building_grid = building_observer_points().create_points(model_file, desired_building_key)



class building_characteristics():

    def generate_occupancy_profile(self, maximum_occupancy):

        occupancy_pattern_weekdays = [1, 1, 1, 1, 1, 1, 0.8, 0.6, 0.5,
                                      0.2, 0.2, 0.2, 0.2, 0.6, 0.8, 0.9, 0.9, 0.9, 0.9, 1, 1, 1, 1, 1]
        occupancy_pattern_weekdends = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1, 1, 1, 1]

        occupants_weekdays = [int(item*maximum_occupancy) for item in occupancy_pattern_weekdays]
        occupants_weekends = [int(item*maximum_occupancy) for item in occupancy_pattern_weekdends]

        """satarting the generation of an annual occupancy vector """

        year = pd.date_range(start='2022-01-01', periods=8760, freq='H')
        annual_occupancy_vector = pd.DataFrame({'datetime': year})

        # Extract the day of the week (0 = Monday, 6 = Sunday)
        annual_occupancy_vector['day_of_week'] = annual_occupancy_vector['datetime'].dt.dayofweek

        # Create a new column to label weekdays and weekends
        annual_occupancy_vector['day_type'] = annual_occupancy_vector['day_of_week'].apply(lambda x: 'weekday' if x < 5 else 'weekend')
        annual_occupancy_vector['occupancy_pattern'] = len(annual_occupancy_vector)*[0]

        # define occupancy column
        for h in range(len(annual_occupancy_vector)):
            if annual_occupancy_vector.loc[h,'day_type'] == 'weekday':
                annual_occupancy_vector.loc[h,'occupancy_pattern'] = occupants_weekdays[annual_occupancy_vector.loc[h, 'datetime'].hour]

            elif annual_occupancy_vector.loc[h,'day_type'] == 'weekend':
                annual_occupancy_vector.loc[h,'occupancy_pattern'] = occupants_weekends[annual_occupancy_vector.loc[h, 'datetime'].hour]

        return annual_occupancy_vector
    

    def get_outside_temperature_data(self, path_to_temp, tile_center):
        source_epsg = 4326
        target_epsg = 25832

        transformer = pyproj.Transformer.from_crs(source_epsg, target_epsg, always_xy=True)
        files = [f for f in os.listdir(path_to_temp)]


        file_detials = []
        for i in range(len(files)):
            filepath = os.path.join(path_to_temp, files[i])
            filename = os.path.basename(filepath)
            year = int(filename[3:3+4])
            latitude = float(filename[8:8+6])/10000
            longitude = float(filename[14:14+6])/10000
            utm_easting, utm_northing = transformer.transform(longitude, latitude)
            file_detials.append({'filename': filename, 'latitude': utm_easting, 'longitude': utm_northing})

        temperture_files = pd.DataFrame(file_detials)
        temperture_files['Distance'] = temperture_files.apply(lambda row: distance.euclidean(
            (tile_center.x, tile_center.y), (row['latitude'], row['longitude'])), axis=1)
        closest_point = temperture_files.loc[temperture_files['Distance'].idxmin()]
        filepath = os.path.join(path_to_temp, closest_point['filename'])
        data = pd.read_csv(filepath, sep=r"\s+",skiprows=list(range(0, 31))+[33])
        data.index = pd.date_range(f"{year}-01-01 00:30:00", periods=8760, freq="H", tz="Europe/Berlin")

        return data

    def average_Uvalue(self, YearClass, SurfaceType):
        if YearClass <= 1978:
            if SurfaceType == 'Roof':
                U = 0.77
            elif SurfaceType == 'Wall':
                U = 1.15
            elif SurfaceType == 'Window':
                U = 2.64
            elif SurfaceType == 'Floor':
                U = 1.05
        elif YearClass >= 1979 and YearClass <= 1994:
            if SurfaceType == 'Roof':
                U = 0.40
            elif SurfaceType == 'Wall':
                U = 0.64
            elif SurfaceType == 'Window':
                U = 2.37
            elif SurfaceType == 'Floor':
                U = 0.71

        elif YearClass >= 1995:
            if SurfaceType == 'Roof':
                U = 0.23
            elif SurfaceType == 'Wall':   
                U = 0.28
            elif SurfaceType == 'Window':
                U = 1.28
            elif SurfaceType == 'Floor':
                U = 0.36

        return U


tile_center = tile_footprints.geometry.centroid.iloc[0]
outside_temperature = building_characteristics().get_outside_temperature_data(climate_data, tile_center)


########################################################################################################

""" Mapping calculated hours to their similar hour where sun position angle differs only withing +/- 3 degree """

listing=pd.DataFrame(columns=['hour','similar_hours_list'])

for h in range(len(hours_to_be_calculated['hour'].tolist())):
    listing.loc[h,'hour']=hours_to_be_calculated['hour'].tolist()[h]
    listing.loc[h,'similar_hours_list']=hours_deleted[h]['hour'].tolist()


hours_df = pd.DataFrame(columns=['hours','calculated_hours','similar_hours'])
hours_df['hours']=sun_position['hour']
hours_df=hours_df.reset_index(drop=True)
hours_df['calculated_hours']=hours_df['hours'].isin(hours_to_be_calculated['hour'])

for h in range(len(hours_df)):
    if hours_df.loc[h,'calculated_hours']==True:
        calcualted_hour=hours_df.loc[h,'hours']
        similar_hours_list=listing.loc[listing['hour'] == calcualted_hour]['similar_hours_list'].iloc[0]
        hours_df.loc[hours_df['hours'].isin(similar_hours_list), 'similar_hours'] = calcualted_hour



"""Identidy an hour for generating examplery plots"""
# examplery_hour=6512
try: 
    idx_loc=hours_to_be_calculated.index.get_loc(examplery_hour)
except KeyError: 
    equivalent_calculated_hour=int(hours_df.loc[hours_df['hours'] == examplery_hour].iloc[0]['similar_hours'])
    idx_loc=hours_to_be_calculated.index.get_loc(hours_to_be_calculated.index[hours_to_be_calculated["hour"] == equivalent_calculated_hour].tolist()[0])


""" calculate shadow height and shadow factor for points """
number_of_building = len(building_grid['id'])
save_grid = []
calc_for_sun_positions = []

for sp in range(len(hours_to_be_calculated)): # idx_loc,idx_loc+1
    calc_for_sun_position_i = {}
    updated_building_grid = copy.deepcopy(building_grid)
    shade_line_wall=[]
    orientation_line_wall=[]
    shade_line_roof = []
    orientation_line_roof=[]
    solar_radiation_all_buildings = []
    roof_solar_radiation_all_buildings = []

    """get sun position """
    sun_elevation = hours_to_be_calculated['altitude'].iloc[sp]
    Sun_azimuth_angle = hours_to_be_calculated['azimuth'].iloc[sp]
    Sun_azimuth_angle_due_north=(180+Sun_azimuth_angle) % 360
    
    """ get outside temperature"""
    hour_of_the_year = int(hours_to_be_calculated['hour'].iloc[sp])
    T_o = outside_temperature['t'].iloc[hour_of_the_year-1]

    for b in range(len(updated_building_grid['id'])):

        number_of_Walls_for_building_i = len(building_grid['walls'][b]['orientation'])
        number_of_roof_segments_building_i = len(building_grid['roof'][b]['orientation'])
        building_height=building_grid['building_height'][b]

        """ get U values"""
        YC = tile_footprints[tile_footprints['id'] == desired_building_key]['year_class'].iloc[0]

        U_value_wall = building_characteristics().average_Uvalue(int(YC[5:9]), 'Wall')
        U_value_roof = building_characteristics().average_Uvalue(int(YC[5:9]), 'Roof')
        U_value_window = building_characteristics().average_Uvalue(int(YC[5:9]), 'Window')
        U_value_floor = building_characteristics().average_Uvalue(int(YC[5:9]), 'Floor')

        total_solar_radiation_all_walls = []
        solar_radiation_all_walls = []
        shadow_height_all_walls=[]
        shade_line = []

        for j in range(number_of_Walls_for_building_i):
            """get wall orientation """
            wall_orientation = building_grid['walls'][b]['orientation'][j]
            wall_surface_area = building_grid['walls'][b]['wall_surface_area'][j]
            number_of_cols_per_wall = len(building_grid['walls'][b]['columns_points'][j])
            number_of_points_at_each_col = [len(nested_list) for nested_list in building_grid['walls'][b]['columns_points'][j]]
            number_of_ground_points_per_wall = len(building_grid['walls'][b]['ground_points'][j])

            if number_of_ground_points_per_wall == 0:
                solar_radiation_all_columns_per_wall = []
                shadow_heights_all_columns_per_wall=[]
                total_solar_radiation_all_walls.append(0)
                solar_radiation_all_walls.append([])
                continue

            total_direct_radiation_at_the_wall = 0            
            Surface_orientation_due_north_deg=(90 - wall_orientation)%360 

            if Surface_orientation_due_north_deg > (Sun_azimuth_angle_due_north) + 90 or Surface_orientation_due_north_deg < (Sun_azimuth_angle_due_north) - 90:

                """
                Why No calculation Necessary? 

                Wall Orientation Greater than Sun Azimuth + 90 Degrees: If the wall is facing away from the general direction of 
                the sun (azimuth ± 90 degrees), it means the wall's normal vector is pointing almost perpendicular to the sun's rays.
                In this case, the angle of incidence would be very high, and the wall would not receive direct solar radiation. 
                Instead, it would likely be in the shadow cast by the wall itself.

                Wall Orientation Less than Sun Azimuth - 90 Degrees: Similarly, if the wall's orientation is significantly less than 
                the sun azimuth minus 90 degrees, the wall's normal vector is again pointing away from the sun, and the wall is likely 
                to be in the shadow.

                """
                solar_radiation_all_columns_per_wall = [pts_per_col*[0] for pts_per_col in number_of_points_at_each_col]                
                shadow_heights_all_columns_per_wall = [pts_per_col*[building_height] for pts_per_col in number_of_points_at_each_col]

            else:
                """ create line of sight for each wall """
                line_length = hours_to_be_calculated['maximum_shading_line'].iloc[sp]
                line_angle = np.radians(hours_to_be_calculated['azimuth'].iloc[sp])
                orientation_angle =np.radians(building_grid['walls'][b]['orientation'][j])
                line_angle = (1.5*np.pi - line_angle)
                
                
                count = 0
                solar_radiation_all_columns_per_wall = []
                shadow_heights_all_columns_per_wall=[]
                for p in range(len(building_grid['walls'][b]['ground_points'][j])):
                    if len(building_grid['walls'][b]['ground_points'][j][p]) == 0:
                        continue

                    ground_point = building_grid['walls'][b]['ground_points'][j][p][0]

                    shade_line_end_point = [ground_point[0] + line_length * np.cos(
                        line_angle), ground_point[1] + line_length * np.sin(line_angle), ground_point[2]]
                    orinetation_line_end_point= [ground_point[0] + line_length * np.cos(
                        orientation_angle), ground_point[1] + line_length * np.sin(orientation_angle), ground_point[2]]

                    shade_line.append(LineString([Point(ground_point), Point(shade_line_end_point)]))
                    shade_line_wall.append(LineString( [Point(ground_point), Point(shade_line_end_point)]))
                    orientation_line_wall.append(LineString( [Point(ground_point), Point(orinetation_line_end_point)]))

                    """check intersection"""

                    try:
                        check_intersection = (tile_footprints['geometry'].intersects(
                            (LineString([Point(ground_point[:2]), Point(shade_line_end_point[:2])]))))
                        matched = check_intersection[check_intersection].index

                    except:
                        matched = []
                        count = count+1

                    shadow_heights = []
                    shadow_heights_col_point=[]
                    intersection_distances = []

                    if len(matched) != 0:
                        for m in range(len(matched)):
                            indx = check_intersection[check_intersection].index[m]
                            intersection_distance = tile_footprints.iloc[indx]['geometry'].boundary.distance(Point(ground_point))
                            intersection_distances.append(intersection_distance)

                            intersected_building_height = float(buildings_cityjson[tile_footprints.iloc[indx]['id']].attributes['measuredHeight'])
                            intersected_building_elevation = float( buildings_cityjson[tile_footprints.iloc[indx]['id']].attributes['HoeheDach'])


                            shadow_height = ((intersected_building_elevation-ground_point[2])/np.tan(np.radians(sun_elevation)) - intersection_distance) * np.tan(np.radians(sun_elevation))
                            shadow_heights.append(shadow_height)

                        max_shadow_height = max(shadow_heights)
                        max_shadow_height_index = shadow_heights.index(max_shadow_height)

                        """do the fixing here """

                        if max_shadow_height > 0:
                            """
                            ground point is in shadow and shodow on columns points must be checked 
                            Calculation has to be performed for column points too """
                            max_shadow_height_ind = []

                            number_of_points_per_col = len(building_grid['walls'][b]['columns_points'][j][p])
                            col_points = [item[2] for item in building_grid['walls'][b]['columns_points'][j][p]]
                            is_ascending = all(col_points[i] <= col_points[i+1] for i in range(len(col_points)-1))
                            is_dec = all(col_points[i] >= col_points[i+1] for i in range(len(col_points)-1))

                            if is_ascending == True:
                                looping_range = range(number_of_points_per_col)

                            elif is_dec == True:
                                looping_range = range(number_of_points_per_col-1, -1, -1)

                            else:
                                """ In case the col. had only one point """
                                looping_range = range(number_of_points_per_col)

                            solar_radiation_at_point = []
                            
                            for clp in looping_range:  # number_of_points_per_col
                                column_point = building_grid['walls'][b]['columns_points'][j][p][clp]
                                shade_line_end_point = [column_point[0] + line_length * np.cos(
                                    line_angle), column_point[1] + line_length * np.sin(line_angle), column_point[2]]
                                
                                shadow_heights_col = []
                                for mc in range(len(matched)):
                                    indx = check_intersection[check_intersection].index[mc]
                                    intersection_distance = tile_footprints.iloc[indx]['geometry'].boundary.distance(Point(ground_point))
                                    id_interested_foot_print = tile_footprints['id'][matched[mc]]
                                    intersected_building_elevation = float(buildings_cityjson[id_interested_foot_print].attributes['HoeheDach'])
                                    if intersected_building_elevation > column_point[2]:
                                        """ intersection occured """
                                        shadow_height = ((intersected_building_elevation-column_point[2])/np.tan(np.radians(sun_elevation)) - intersection_distance) * np.tan(np.radians(sun_elevation))
                                        shadow_heights_col.append(shadow_height)
                                        
                                    else:
                                        shadow_heights_col.append(0)
                                max_shadow_height_ind.append(shadow_heights_col.index(max(shadow_heights_col)))

                                if max(shadow_heights_col) > 0:

                                    shadow_heights_col_point.append(max(shadow_heights_col))
                                    solar_radiation_at_point.append(0)
                                else:
                                    shadow_heights_col_point.append(0)

                                    """ Calculation is based on: 
                                        M. Horvath, "Analysis of buildings solar heat gain," 2013 4th International Youth Conference on Energy (IYCE), Siófok, Hungary, 2013, pp. 1-5, doi: 10.1109/IYCE.2013.6604196.
                                    """

                                    """ Steps1: 
                                        Define variables required for calculating solar radiation and 
                                        determine angle of incidence
                                        Reamrk: The angle of incidence is generally measured in radians and is typically
                                        bound within the range of 0 to π/2 (0 to 90 degrees)
                                    """

                                    Surface_orientation = np.radians( wall_orientation)  # value is stored in degrees
                                    Surface_orientation_due_north=np.radians((90 - wall_orientation)%360 )
                                    Surface_slope = np.radians(90)

                                    # Sun Altitude and Sun Elevation are terms used interchangeably to describe the same angle.

                                    Sun_altitude = np.radians(sun_elevation)
                                    Sun_azimuth = np.radians(Sun_azimuth_angle_due_north)

                                    angle_of_incidence = np.arccos((np.sin(Sun_altitude) * np.cos(Surface_slope)) + (np.cos(Sun_altitude) * np.sin(Surface_slope)*np.cos(Sun_azimuth-Surface_orientation_due_north)))
                                    angle_of_incidence_deg = np.degrees(angle_of_incidence)

                                    """ Steps2: 
                                        calculating air mass 
                                    """
                                    air_mass = (np.cos((np.pi/2)-Sun_altitude) + 0.50572 * (6.07995 + Sun_altitude)**(-1.6364))**(-1)

                                    """ Steps3: 
                                        calculating direct solar radiation
                                    """
                                    a = 0.14  # empirical constant
                                    surface_altitude_km = column_point[2]/1000
                                    direct_solar_radiation = 1.353 * (((1-a*surface_altitude_km)*(0.7**(air_mass**0.678)))+a*surface_altitude_km)          # in kW/m2

                                    """ Step4 (Last):
                                        Evaluate the direct radiation on the vertical surface .. 
                                    """

                                    direct_radiation_tilted = direct_solar_radiation * np.cos(angle_of_incidence)
                                    solar_radiation_at_point.append(direct_radiation_tilted)

                            if is_dec == True:
                                solar_radiation_all_columns_per_wall.append(solar_radiation_at_point[::-1])
                                shadow_heights_all_columns_per_wall.append(shadow_heights_col_point[::-1])

                            elif is_ascending == True:
                                solar_radiation_all_columns_per_wall.append(solar_radiation_at_point)
                                shadow_heights_all_columns_per_wall.append(shadow_heights_col_point)

                            point_area = wall_surface_area /sum([len(item)+1 for item in solar_radiation_all_columns_per_wall])
                        else:

                            """
                            there were building interseting with the ground point 
                            however, shadow height is negative as the distance is too large 
                            Therefore, it's certain that all points at the recpective column are not obstructed by shadow of the surrounding buildings 
                            shadow height is set to zero
                            and solar radiation at each point must be calculated
                            """

                            shadow_heights.append(0)
                            number_of_points_per_col = len(building_grid['walls'][b]['columns_points'][j][p])
                            shadow_heights_col_point=number_of_points_per_col*[0]   #"""here trial"""
                            col_points = [item[2] for item in building_grid['walls'][b]['columns_points'][j][p]]
                            is_ascending = all(col_points[i] <= col_points[i+1] for i in range(len(col_points)-1))

                            is_dec = all(col_points[i] >= col_points[i+1] for i in range(len(col_points)-1))

                            if is_ascending == True:
                                looping_range = range(number_of_points_per_col)
                            elif is_dec == True:
                                looping_range = range(number_of_points_per_col-1, -1, -1)
                            else:
                                """in case the col. had only one point """
                                looping_range = range(number_of_points_per_col)

                            solar_radiation_at_point = []
                            for clp in looping_range:  # number_of_points_per_col
                                column_point = building_grid['walls'][b]['columns_points'][j][p][clp]

                                """ Calculation is based on: 
                                    M. Horvath, "Analysis of buildings solar heat gain," 2013 4th International Youth Conference on Energy (IYCE), Siófok, Hungary, 2013, pp. 1-5, doi: 10.1109/IYCE.2013.6604196.
                                """

                                """ Steps1: 
                                    Define variables required for calculating solar radiation and 
                                    determine angle of incidence
                                    Reamrk: The angle of incidence is generally measured in radians and is typically
                                    bound within the range of 0 to π/2 (0 to 90 degrees)
                                """

                                Surface_orientation = np.radians( wall_orientation)  # value is stored in degrees
                                Surface_orientation_due_north=np.radians((90 - wall_orientation)%360 )
                                Surface_slope = np.radians(90)

                                # Sun Altitude and Sun Elevation are terms used interchangeably to describe the same angle.
                                Sun_altitude = np.radians(sun_elevation)
                                Sun_azimuth = np.radians(Sun_azimuth_angle_due_north)

                                angle_of_incidence = np.arccos((np.sin(Sun_altitude) * np.cos(Surface_slope)) + (np.cos(Sun_altitude) * np.sin(Surface_slope)*np.cos(Sun_azimuth-Surface_orientation_due_north)))
                                angle_of_incidence_deg = np.degrees( angle_of_incidence)
                                
                                

                                """ Steps2: 
                                    calculating air mass 
                                """

                                air_mass = (np.cos((np.pi/2)-Sun_altitude) + 0.50572 * (6.07995 + Sun_altitude)**(-1.6364))**(-1)

                                """ Steps3: 
                                    calculating direct solar radiation
                                """

                                a = 0.14  # empirical constant
                                surface_altitude_km = column_point[2]/1000
                                direct_solar_radiation = 1.353 * (((1-a*surface_altitude_km)*(0.7**(air_mass**0.678)))+a*surface_altitude_km)      # in kW/m2

                                """ Step4 (Last):
                                    Evaluate the direct radiation on the vertical surface .. 
                                """
                                direct_radiation_tilted = direct_solar_radiation * np.cos(angle_of_incidence)
                                solar_radiation_at_point.append(direct_radiation_tilted)

                            if is_dec == True:
                                solar_radiation_all_columns_per_wall.append(solar_radiation_at_point[::-1])
                                shadow_heights_all_columns_per_wall.append(shadow_heights_col_point[::-1])

                            elif is_ascending == True:
                                solar_radiation_all_columns_per_wall.append(solar_radiation_at_point)
                                shadow_heights_all_columns_per_wall.append(shadow_heights_col_point)

                    else:

                        """
                        No buildings obstructing at the ground point 
                        Therefore, it's certain that all points at the recpective column are not obstructed by shadow of the surrounding buildings 
                        shadow height is set to zero
                        and solar radiation at each point must be calculated
                        """

                        shadow_heights.append(0)
                        shadow_heights_col_point.append(0)
                        number_of_points_per_col = len(
                            building_grid['walls'][b]['columns_points'][j][p])
                        col_points = [
                            item[2] for item in building_grid['walls'][b]['columns_points'][j][p]]
                        is_ascending = all(
                            col_points[i] <= col_points[i+1] for i in range(len(col_points)-1))
                        is_dec = all(col_points[i] >= col_points[i+1]
                                      for i in range(len(col_points)-1))

                        if is_ascending == True:
                            looping_range = range(number_of_points_per_col)
                        elif is_dec == True:
                            looping_range = range(
                                number_of_points_per_col-1, -1, -1)

                        else:
                            """in case the col. had only one point """
                            looping_range = range(number_of_points_per_col)

                        solar_radiation_at_point = []
                        for clp in looping_range:  # number_of_points_per_col
                            column_point = building_grid['walls'][b]['columns_points'][j][p][clp]

                            """ Calculation is based on: 
                                M. Horvath, "Analysis of buildings solar heat gain," 2013 4th International Youth Conference on Energy (IYCE), Siófok, Hungary, 2013, pp. 1-5, doi: 10.1109/IYCE.2013.6604196.
                            """

                            """ Steps1: 
                                Define variables required for calculating solar radiation and 
                                determine angle of incidence
                                Remark: The angle of incidence is generally measured in radians and is typically
                                bound within the range of 0 to π/2 (0 to 90 degrees)
                            """

                            Surface_orientation = np.radians(wall_orientation)  # value is stored in degrees
                            Surface_orientation_due_north=np.radians((90 - wall_orientation)%360 )
                            Surface_slope = np.radians(90)

                            # Sun Altitude and Sun Elevation are terms used interchangeably to describe the same angle.
                            Sun_altitude = np.radians(sun_elevation)
                            Sun_azimuth = np.radians(Sun_azimuth_angle_due_north)

                            angle_of_incidence = np.arccos((np.sin(Sun_altitude) * np.cos(Surface_slope)) + (np.cos(Sun_altitude) * np.sin(Surface_slope)*np.cos(Sun_azimuth-Surface_orientation_due_north)))
                            angle_of_incidence_deg = np.degrees(angle_of_incidence)

                            """ Steps2: 
                                calculating air mass                         
                                Kasten, F., & Young, A. T. (1989). Revised optical air mass tables and approximation formula. Applied Optics, 28(22), 47354738. https://doi.org/10.1364/AO.28.004735 
                            """

                            air_mass = (np.cos((np.pi/2)-Sun_altitude) + 0.50572 * (6.07995 + Sun_altitude)**(-1.6364))**(-1)

                            """ Steps3: 
                                calculating direct solar radiation
                            """

                            a = 0.14  # empirical constant
                            surface_altitude_km = column_point[2]/1000
                            direct_solar_radiation = 1.353 *(((1-a*surface_altitude_km)*(0.7**(air_mass**0.678)))+a*surface_altitude_km)          # in kW/m2

                            """ Step4 (Last):
                                Evaluate the direct radiation on the vertical surface .. 
                            """

                            direct_radiation_tilted = direct_solar_radiation * np.cos(angle_of_incidence)
                            solar_radiation_at_point.append(direct_radiation_tilted) #direct_solar_radiation
                            shadow_heights_col_point.append(0)
                            
                        if is_dec == True:
                            solar_radiation_all_columns_per_wall.append(solar_radiation_at_point[::-1])
                            shadow_heights_all_columns_per_wall.append(shadow_heights_col_point[::-1])

                        elif is_ascending == True:
                            solar_radiation_all_columns_per_wall.append(solar_radiation_at_point)
                            shadow_heights_all_columns_per_wall.append(shadow_heights_col_point)

                    point_area = wall_surface_area /sum([len(item) for item in solar_radiation_all_columns_per_wall])

                    total_solar_radiation_at_the_wall = []
                    for i in range(len(solar_radiation_all_columns_per_wall)):
                        total_solar_radiation_at_the_wall.append(
                            sum([item*point_area for item in solar_radiation_all_columns_per_wall[i]]))
                    total_direct_radiation_at_the_wall = sum(total_solar_radiation_at_the_wall)

            shadow_height_all_walls.append(shadow_heights_all_columns_per_wall)

            total_solar_radiation_all_walls.append(total_direct_radiation_at_the_wall)
            solar_radiation_all_walls.append(solar_radiation_all_columns_per_wall)
            
        updated_building_grid['walls'][b]['solar_radiation_at_col'] = solar_radiation_all_walls
        updated_building_grid['walls'][b]['shadow_height_at_col'] = shadow_height_all_walls
        updated_building_grid['walls'][b]['total_direct_radiation_all_walls'] = total_solar_radiation_all_walls
        updated_building_grid['walls'][b]['total_direct_radiation_building'] = sum(total_solar_radiation_all_walls)


        solar_heat_gain_building = []     
        all_walls_window_points=[]
        solar_radiation_of_all_window_points=[]
        glazing_area_per_building = []
        for w in range(number_of_Walls_for_building_i):
            glazing_area_per_wall = building_grid['walls'][b]['all_wall_glazing_area'][w]
            glazing_area_per_building.append(glazing_area_per_wall)
            columns_per_wall = len(building_grid['walls'][b]['columns_points'])
            windows_per_wall = building_grid['walls'][b]['windows_per_wall'][w]

            solar_radiation_of_point = []
            points_in_rectangle = []
            """ Solar heat gains by windows """
            for f in range(len(windows_per_wall)):

                all_windows_per_floor = building_grid['walls'][b]['windows_per_wall'][w][f]
                for win in range(len(all_windows_per_floor)):
                    """all_windows_per_floor"""
                    single_window_vertices = all_windows_per_floor[win]
                    x1, y1, z1 = single_window_vertices[0]
                    x2, y2, z2 = single_window_vertices[1]
                    x3, y3, z3 = single_window_vertices[2]
                    x4, y4, z4 = single_window_vertices[3]

                    for col_w in range(columns_per_wall):
                        num_points_per_col = len(
                            building_grid['walls'][b]['columns_points'][col_w])
                        for poin_in_col_w in range(num_points_per_col):
                            n = len(
                                building_grid['walls'][b]['columns_points'][col_w][poin_in_col_w])
                            for point in range(n):
                                pt = building_grid['walls'][b]['columns_points'][col_w][poin_in_col_w][point]
                                px, py, pz = building_grid['walls'][b]['columns_points'][col_w][poin_in_col_w][point]
                                if (min(x1, x2, x3, x4) <= px <= max(x1, x2, x3, x4) and
                                    min(y1, y2, y3, y4) <= py <= max(y1, y2, y3, y4) and
                                        min(z1, z2, z3, z4) <= pz <= max(z1, z2, z3, z4)):
                                    points_in_rectangle.append(pt)
                                    solar_radiation_of_point.append(updated_building_grid['walls'][b]['solar_radiation_at_col'][col_w][poin_in_col_w][point])

            all_walls_window_points=all_walls_window_points+points_in_rectangle
            points_in_rectangle_points=[Point(pt_w) for pt_w in all_walls_window_points]
            solar_radiation_of_all_window_points=solar_radiation_of_all_window_points+solar_radiation_of_point
            
            window_grid_check = {'geometry': points_in_rectangle_points,'solar_radiation_list': solar_radiation_of_all_window_points}

            solar_heat_gain_by_wall = sum([(glazing_area_per_wall/len(points_in_rectangle))*item for item in solar_radiation_of_point])
            solar_heat_gain_building.append(solar_heat_gain_by_wall)

            updated_building_grid['walls'][b]['solar_heat_gain_building'] = solar_heat_gain_building


        total_absolute_solar_radiation_all_roofs = []
        solar_radiation_all_roof_segments = []
        shadow_height_all_roof_segments = []
        for r in range(number_of_roof_segments_building_i):
            roof_segment_orientation = building_grid['roof'][b]['orientation'][r]
            roof_segment_slope = 90-building_grid['roof'][b]['slope'][r]

            roof_segment_orientation1=(90-roof_segment_orientation)%360
            roof_segment_area = building_grid['roof'][b]['Roof_surface_area'][r]
            line_length = hours_to_be_calculated['maximum_shading_line'].iloc[sp]
            line_angle = np.radians(hours_to_be_calculated['azimuth'].iloc[sp])
            line_angle = (1.5*np.pi - line_angle)
            line_slope = np.radians(roof_segment_slope)

            solar_radiation_roof_segment = []
            shadow_height_roof_segment = []
            for p in range(len(building_grid['roof'][b]['points'][r])):

                point = building_grid['roof'][b]['points'][r][p]
                shade_line_end_point = [point[0] + line_length * np.cos(line_angle) * np.cos(line_slope), point[1] + line_length * np.sin(line_angle) * np.cos(line_slope), point[2] + line_length * np.sin(line_slope)]
                shade_line_end_point = [point[0] + line_length * np.cos(line_angle), point[1] + line_length * np.sin(line_angle), point[2]]
               
                
               
                orinetation_line_end_point_roof= [point[0] + line_length * np.cos((np.radians(roof_segment_orientation1))), point[1] + line_length * np.sin((np.radians(roof_segment_orientation1))), point[2]]
                A = shade_line_end_point[0] - point[0]
                B = shade_line_end_point[1] - point[1]
                C = shade_line_end_point[2] - point[2]
                D = A * point[0] + B * point[1] + C * point[2]

                """
                parametric equation of 3d line: using it to get the elevation at the intersection point with the footprint
                x= point[0]+ A*t  
                y= point[1]+ B*t  
                z= point[2]+ C*t  
                
                """

                shade_line_roof.append((LineString([Point(point), Point(shade_line_end_point)])))
                shade_line = LineString([Point(point), Point(shade_line_end_point)])
                
                    
                orientation_line_roof.append((LineString([Point(point), Point(orinetation_line_end_point_roof)])))

                try:
                    check_intersection = (tile_footprints['geometry'].intersects((LineString([Point(point[:2]), Point(shade_line_end_point[:2])]))))
                    matched = check_intersection[check_intersection].index
                    building_id_all_intersections = [tile_footprints['id'][matched[mc]] for mc in range(len(matched))]

                    intersected_polygons = [tile_footprints[tile_footprints['id'] == building_id].geometry for building_id in building_id_all_intersections]
                    inter = [shade_line.intersection(poly).bounds[:2] for poly in intersected_polygons]
                    intersection_point2d = [[inter[i]['minx'].iloc[0], inter[i]['miny'].iloc[0]] for i in range(len(inter))]

                    t = [(intersection_point2d[i][0]-point[0]) /A for i in range(len(intersection_point2d))]
                    intersection_point_z_component = [point[2] + C*t[i] for i in range(len(intersection_point2d))]
                    intersection_point_3d = [[intersection_point2d[i][0], intersection_point2d[i][1],intersection_point_z_component[i]] for i in range(len(intersection_point2d))]
                except:
                    matched = []
                    count = count+1

                """ calculating shadow height for each of the intersected buildings 
                negative shadow heights mean that building is not shadowed """

                if len(matched) == 0:
                    shadow_heights_roof_point = [0]
                else:
                    shadow_heights_roof_point = []

                for mc in range(len(matched)):
                    indx = check_intersection[check_intersection].index[mc]
                    intersection_distance = tile_footprints.iloc[indx]['geometry'].boundary.distance(Point(point[:2]))
                    id_interested_foot_print = tile_footprints['id'][matched[mc]]
                    intersected_building_elevation = float(buildings_cityjson[id_interested_foot_print].attributes['HoeheDach'])
                    
                    if intersected_building_elevation > intersection_point_3d[mc][2] or intersected_building_elevation > intersection_point_3d[mc][2]:
                        if intersected_building_elevation >= point[2] and id_interested_foot_print != building_grid['id'][b]:

                            """ intersection occured """
                            shadow_height = ((intersected_building_elevation-point[2])/np.tan(np.radians(sun_elevation)) - intersection_distance) * np.tan(np.radians(sun_elevation))
                            shadow_heights_roof_point.append(shadow_height)
                        else:
                            shadow_heights_roof_point.append(0)

                    else:
                        shadow_heights_roof_point.append(0)

                if max(shadow_heights_roof_point) > 0 and len(matched) != 0:
                    shadow_heights_roof_point.append(max(shadow_heights_roof_point))
                    shadow_height_roof_segment.append(max(shadow_heights_roof_point))
                    solar_radiation_roof_point = 0
                    solar_radiation_roof_segment.append(solar_radiation_roof_point)
                else:
                    shadow_heights_roof_point.append(0)

                    """ Calculation is based on: 
                        M. Horvath, "Analysis of buildings solar heat gain," 2013 4th International Youth Conference on Energy (IYCE), Siófok, Hungary, 2013, pp. 1-5, doi: 10.1109/IYCE.2013.6604196.
                    """

                    """ Steps1: 
                        Define variables required for calculating solar radiation and 
                        determine angle of incidence
                        Remark: The angle of incidence is generally measured in radians and is typically
                        bound within the range of 0 to π/2 (0 to 90 degrees)
                    """

                    Surface_orientation = np.radians(roof_segment_orientation)  # value is stored in degrees
                    Surface_orientation_due_north= np.radians((90-roof_segment_orientation)%360)
                    Surface_orientation_due_north= np.radians(roof_segment_orientation)
                    Surface_slope = np.radians(roof_segment_slope)
                    # Sun Altitude and Sun Elevation are terms used interchangeably to describe the same angle.
                    Sun_altitude = np.radians(sun_elevation)
                    Sun_azimuth = np.radians(Sun_azimuth_angle_due_north)

                    angle_of_incidence = np.arccos((np.sin(Sun_altitude) * np.cos(Surface_slope)) + (np.cos(Sun_altitude) * np.sin(Surface_slope)*np.cos(Sun_azimuth-Surface_orientation_due_north)))
                    angle_of_incidence_deg = np.degrees(angle_of_incidence)

                    """ Steps2: 
                        calculating air mass 
                    """
                    air_mass = (np.cos((np.pi/2)-Sun_altitude) + 0.50572 *(6.07995 + Sun_altitude)**(-1.6364))**(-1)

                    """ Steps3: calculating direct solar radiation
                    """
                    a = 0.14  # empirical constant

                    # extra-terrestrial solar radiation density of 1353 W/m2 or 1.353 kW/m2
                    # 0.7 factor represents the approximately 70% of the incoming radiation that will pass through the atmosphere.
                    # the 0.678 factor is the air mass correction factor proposed by Meinel et al:  A. B. Meinel, M. P. Meinel, reportApplied Solar Energy: an Introduction, NASA STI/Recon Technical Report A 77.
                    # Ref: https://www.pveducation.org/pvcdrom/properties-of-sunlight/air-mass#footnote1_4rb1p3b

                    surface_altitude_km = point[2]/1000
                    direct_solar_radiation = 1.353 * (((1-a*surface_altitude_km)*(0.7**(air_mass**0.678))) + a*surface_altitude_km)          # in kW/m2

                    """ Step4 (Last):
                        Evaluate the direct radiation on the roof surface .. 
                    """

                    direct_radiation_tilted = direct_solar_radiation * np.cos(angle_of_incidence)
                    solar_radiation_roof_point = direct_radiation_tilted
                    solar_radiation_roof_segment.append(solar_radiation_roof_point)
                    shadow_height_roof_segment.append(0)
                    roof_point_area = roof_segment_area / len(solar_radiation_roof_segment)

                    total_absolute_solar_radiation_at_roof = sum(
                        [roof_point_area*solar_radiation_roof_segment[roof_solar_point] for roof_solar_point in range(len(solar_radiation_roof_segment))])  # value in kW

            solar_radiation_all_roof_segments.append( solar_radiation_roof_segment)
            shadow_height_all_roof_segments.append(shadow_height_roof_segment)
            total_absolute_solar_radiation_all_roofs.append(total_absolute_solar_radiation_at_roof)

        updated_building_grid['roof'][b]['solar_radiation_at_roof_segment'] = solar_radiation_all_roof_segments
        updated_building_grid['roof'][b]['absolute_solar_radiation_roof'] = total_absolute_solar_radiation_all_roofs
        updated_building_grid['roof'][b]['shadow_heights_roof_point'] = shadow_height_all_roof_segments

        solar_radiation_all_buildings.append(solar_radiation_all_walls)
        roof_solar_radiation_all_buildings.append(solar_radiation_all_roof_segments)

    calc_for_sun_position_i['sun_position'] = sp
    calc_for_sun_position_i['total_direct_radiation_all_walls'] = updated_building_grid['walls'][b]['total_direct_radiation_all_walls']
    calc_for_sun_position_i['total_direct_radiation_building'] = updated_building_grid['walls'][b]['total_direct_radiation_building']
    calc_for_sun_positions.append(calc_for_sun_position_i)


    if sp == idx_loc:
        ShadeLines_walls = { 'geometry': shade_line_wall}
        gdf_shade_lines = gpd.GeoDataFrame(ShadeLines_walls, crs="EPSG:25832")
        gdf_shade_lines.to_file(project_dir/"output/saved_shp_files"/f"shade_lines_walls_hour_of_the_year_{examplery_hour}.shp")
        
        orientation_lines_walls = { 'geometry': orientation_line_wall}
        gdf_orientation_lines = gpd.GeoDataFrame(orientation_lines_walls, crs="EPSG:25832")
        gdf_orientation_lines.to_file(project_dir/"output/saved_shp_files"/"orientation_line_walls.shp")
        
        gdf_window_grid_check = gpd.GeoDataFrame(window_grid_check, crs="EPSG:25832")
        gdf_window_grid_check.to_file(project_dir/"output/saved_shp_files"/f"window_solar_radiation_check_hour_of_the_year_{examplery_hour}.shp")
        
        
        """ Reformating and storig roof and wall calculation together """
        points_list = []
        solar_radiation_list = []
        shadow_list=[]
        
        for i in range(len(updated_building_grid['walls'])): 
            if len(updated_building_grid['walls'][i]) == 3:
                continue
            
            # list of point coordinates for each wall 
            my_list = updated_building_grid['walls'][i]['columns_points']
            points = list(itertools.chain.from_iterable(itertools.chain.from_iterable(my_list)))
            points = [Point(item) for item in points]  
            # list of correspponding calculated solar radiation (walls)
            solar_radiation = list(itertools.chain.from_iterable(itertools.chain.from_iterable(updated_building_grid['walls'][i]['solar_radiation_at_col']))) 
            # list of correspponding calculated shadow height (walls)
            shadow_heights_grid = list(itertools.chain.from_iterable(itertools.chain.from_iterable(updated_building_grid['walls'][i]['shadow_height_at_col']))) 
            
            points_roof_list = []
            solar_radiation_roof_list = []
            shadow_heights_roof_list = [] 
            for i in range(len(updated_building_grid['roof'])):
                
                # list of point coordinates for each roof 
                my_list_roof = updated_building_grid['roof'][i]['points']
                points_roof = list(itertools.chain.from_iterable(itertools.chain.from_iterable([my_list_roof])))
                points_roof = [Point(item) for item in points_roof]
                
                # list of correspponding calculated solar radiation (roof i)
                solar_radiation_roof = list(itertools.chain.from_iterable(itertools.chain.from_iterable([updated_building_grid['roof'][i]['solar_radiation_at_roof_segment']])))
                
                # list of correspponding calculated solar radiation (roof i)
                shadow_heights_roof = list(itertools.chain.from_iterable(itertools.chain.from_iterable([updated_building_grid['roof'][i]['shadow_heights_roof_point']])))        
                
                # storing data for all the roof surfaces in one  list 
                points_roof_list = points_roof_list +  points_roof
                solar_radiation_roof_list = solar_radiation_roof_list + solar_radiation_roof
                shadow_heights_roof_list = shadow_heights_roof_list + shadow_heights_roof
            
            # combining together roof and wall data 
            points_list = points_list + points + points_roof_list
            solar_radiation_list = solar_radiation_list + solar_radiation+solar_radiation_roof_list
            shadow_list = shadow_list + shadow_heights_grid + shadow_heights_roof_list
        
        """save in shapefile"""
        building_grid_check = {'geometry': points_list, 'solar_radiation_list': solar_radiation_list, 'shadow_heights':shadow_list}
        solar_radiation_and_shadow_height_geodata = gpd.GeoDataFrame(building_grid_check, crs="EPSG:25832")
        solar_radiation_and_shadow_height_geodata.to_file(project_dir/"output/saved_shp_files"/f"solar_radiation_and_shadow_height_geodata_hour_of_the_year_{examplery_hour}.shp")
        

    

    """Saving solar calculations for future runs  """
    with open(project_dir/"output/hourly_solar_radiation"/f"grid_{sp}.pickle", "wb") as handle:
        pickle.dump(updated_building_grid, handle,protocol=pickle.HIGHEST_PROTOCOL)
    save_grid.append(updated_building_grid)
    
    
########################################################################################################


all_selected_hours_df=pd.DataFrame(columns=['ID','DataFrames'])
for calculated_sp in range(len(hours_to_be_calculated)):
    with open(project_dir/"output/hourly_solar_radiation"/f"grid_{calculated_sp}.pickle", "rb") as handle:
        solar_radiation_calculation = pickle.load(handle)
        Iinc_walls=pd.DataFrame(columns=['solar_radiation'])
        for b in range(len(solar_radiation_calculation['walls'])):
            Iinc_walls.loc[b,'solar_radiation']=solar_radiation_calculation['walls'][b]['solar_heat_gain_building']

    all_selected_hours_df.loc[calculated_sp,'ID']=hours_to_be_calculated.iloc[calculated_sp]['hour']
    all_selected_hours_df.at[calculated_sp,'DataFrames']=Iinc_walls
    

class calculate_demand():
    def transmission_heat_transfer(self,timestep,T_building_previous_timestep='None'):
        number_of_building = len(building_grid['id'])


        outside_temperature = building_characteristics().get_outside_temperature_data(climate_data, tile_center)[['t']]
        outside_temperature = outside_temperature.reset_index()
        outside_temperature=outside_temperature.rename({"index": "datetime"}, axis='columns')

        H_tr_all_buildings=[]
        for i in range(number_of_building):

            number_of_Walls_for_building_i = len(building_grid['walls'][i]['orientation'])

            H_tr = pd.DataFrame(columns=['H_tr_windows','H_tr_opaque_wall'],index=range(0,1))
            for h in range(1): 
                inside_temperature=T_building_previous_timestep
                H_tr_windows_all_walls=[]
                H_tr_opaque_wall_all_walls=[]

                for w in range (number_of_Walls_for_building_i):
                    wall_surface_area = building_grid['walls'][i]['wall_surface_area'][w]
                    all_wall_glazing_area=building_grid['walls'][i]['all_wall_glazing_area'][w]
                    try:
                        year_class=int(tile_footprints[tile_footprints['id']==building_grid['id'][i]]['year_class'].iloc[0][5:9])
                    except: 
                        year_class=2016

                    H_tr_windows_all_walls.append(all_wall_glazing_area*building_characteristics().average_Uvalue(YearClass=year_class, SurfaceType='Window') * (outside_temperature.loc[timestep, 't']+273.15-inside_temperature))
                    H_tr_opaque_wall_all_walls.append((wall_surface_area-all_wall_glazing_area)*building_characteristics().average_Uvalue(YearClass=year_class, SurfaceType='Wall') * (outside_temperature.loc[timestep, 't']+273.15-inside_temperature))
                else:
                    H_tr_windows_all_walls.append(0)
                    H_tr_opaque_wall_all_walls.append(0)

                H_tr.loc[h,'H_tr_windows']=H_tr_windows_all_walls
                H_tr.loc[h,'H_tr_opaque_wall']=H_tr_opaque_wall_all_walls

                total_roof_area=sum(building_grid['roof'][i]['Roof_surface_area'])
                H_tr_roof=total_roof_area*building_characteristics().average_Uvalue(YearClass=year_class, SurfaceType='Roof') * (outside_temperature.loc[timestep, 't']+273.15-inside_temperature)

                H_tr.loc[h,'H_tr_roof']=H_tr_roof
            H_tr_all_buildings.append(H_tr)

        return H_tr_all_buildings


    def solar_gain(self,grid):
        
        building_grid=grid
        number_of_building = len(building_grid['id'])

        H_solar = pd.DataFrame(columns=['H_solar'],index=range(1))

        for i in range(number_of_building):
            H_solar.loc[i,'H_solar']=sum(building_grid['walls'][i]['solar_heat_gain_building'])
        return H_solar   

    def internal_heat_gain(self,building_grid):

        number_of_building = len(building_grid['id'])

        all_building_internal_gains=[]

        """ Internal heat gain """
        for i in range(number_of_building):
            building_floor_area=building_grid['GroundSurfaceArea_dataSet'][i]*building_grid['storeysAboveGround'][i]
            space_area_per_person = 47.7 #sq meteres per person
            maximum_occupancy=int(building_floor_area/space_area_per_person)
            occupancy=building_characteristics().generate_occupancy_profile(maximum_occupancy)

            occupancy['internal_gains']=len(occupancy)*[0]

            for h in range (len(occupancy)):
                if occupancy.loc[h,'datetime'].hour <= 6:
                    heat_per_occupants=80    # in Watts
                    occupancy.loc[h,'internal_gains']=occupancy.loc[h,'occupancy_pattern']*heat_per_occupants
                else:
                    heat_per_occupants=random.randint(100,125)    # in Watts
                    occupancy.loc[h,'internal_gains']=occupancy.loc[h,'occupancy_pattern']*heat_per_occupants

            all_building_internal_gains.append(occupancy['internal_gains'])

        return all_building_internal_gains


one_year_internal_gain_hourly_timestep=calculate_demand().internal_heat_gain(building_grid)


    
def calculate_timestep_single_building_demand(timestep,T_building_previous_timestep):

    dt_seconds=1 * 60 * 60
    specific_heat_capacity= 396E3 #kJ/(m2·K)
    T_buildings_timestep=[]
    space_heating_power_timestep=[]
    solar_gains_timestep=[]
    solar_gains_indvidual_wall_timestep=[]
    internal_heat_gains_timestep=[]
    heat_transfer_by_transmission_timestep=[]

    for building in range (len(building_grid['id'])):

        reference_floor_area=sum(building_grid['walls'][building]['wall_surface_area'])
        heated_living_area= building_grid['GroundSurfaceArea_dataSet'][building] #*building_grid['storeysAboveGround'][building]
        heat_capacity=specific_heat_capacity*reference_floor_area  #heated_living_area #
        internal_heat_gains=one_year_internal_gain_hourly_timestep[building].loc[timestep]

        one_year_Htr_hourly_timestep=calculate_demand().transmission_heat_transfer(timestep,T_building_previous_timestep[building])
        
        one_year_Htr_hourly_timestep_cehck=one_year_Htr_hourly_timestep[0]

        heat_transfer_by_transmission=sum(one_year_Htr_hourly_timestep[building].loc[0,'H_tr_windows'])+sum(one_year_Htr_hourly_timestep[building].loc[0,'H_tr_opaque_wall'])+one_year_Htr_hourly_timestep[building].loc[0,'H_tr_roof']

        if timestep in sun_position['hour'].values:
            # print('yes')
            calculated_hour= hours_df.loc[hours_df['hours'] == timestep].iloc[0]['similar_hours']
            calculated_hour_index=all_selected_hours_df[all_selected_hours_df['ID'] == calculated_hour].index.tolist()[0]
            solar_heat_gains=sum(all_selected_hours_df.loc[all_selected_hours_df['ID'] == calculated_hour].iloc[0]['DataFrames'].loc[building].iloc[0])
            solar_heat_gains_indvidual_walls=all_selected_hours_df.loc[all_selected_hours_df['ID'] == calculated_hour].iloc[0]['DataFrames'].loc[building].iloc[0]
            solar_heat_gains=solar_heat_gains*1000

        else:
            solar_heat_gains=0
            number_of_walls=len(building_grid['walls'][building]['orientation'])
            solar_heat_gains_indvidual_walls=number_of_walls*[0]
        
        solar_gains_timestep.append(solar_heat_gains)
        solar_gains_indvidual_wall_timestep.append(solar_heat_gains_indvidual_walls)
        heat_transfer_by_transmission_timestep.append(heat_transfer_by_transmission)
        internal_heat_gains_timestep.append(internal_heat_gains)
        
        
        if T_building_previous_timestep[building] < 21+273.15-0.5:
            Tset=21+273.15 
            thermalllll= specific_heat_capacity*((Tset - T_building_previous_timestep[building]) / dt_seconds)
            space_heating_power=thermalllll-heat_transfer_by_transmission/reference_floor_area-internal_heat_gains/heated_living_area-solar_heat_gains/reference_floor_area
            T_building_timestep = ((heat_transfer_by_transmission/reference_floor_area + space_heating_power + solar_heat_gains/reference_floor_area + internal_heat_gains/heated_living_area)* (1/(specific_heat_capacity)) * dt_seconds )+T_building_previous_timestep[building]
        
        elif T_building_previous_timestep[building] > 24+273.15+0.5:
            Tset=24+273.15
            thermalllll= specific_heat_capacity*((Tset - T_building_previous_timestep[building]) / dt_seconds)
            space_heating_power=thermalllll-heat_transfer_by_transmission-internal_heat_gains-solar_heat_gains
            space_heating_power=thermalllll-heat_transfer_by_transmission/reference_floor_area-internal_heat_gains/heated_living_area-solar_heat_gains/reference_floor_area
            T_building_timestep = ((heat_transfer_by_transmission/reference_floor_area + space_heating_power + solar_heat_gains/reference_floor_area + internal_heat_gains/heated_living_area)* (1/(specific_heat_capacity)) * dt_seconds )+T_building_previous_timestep[building]
        
        else:
           thermalllll=0
           space_heating_power=0
           T_building_timestep = ((heat_transfer_by_transmission/reference_floor_area + space_heating_power + solar_heat_gains/reference_floor_area + internal_heat_gains/heated_living_area)* (1/(specific_heat_capacity)) * dt_seconds )+T_building_previous_timestep[building]   

        space_heating_power_timestep.append(space_heating_power)
        T_buildings_timestep.append(T_building_timestep)


    """ sum(one_year_Htr_hourly_timestep[building].loc[timestep,'H_tr_windows'])+sum(one_year_Htr_hourly_timestep[building].loc[timestep,'H_tr_opaque_wall']) """

    return T_buildings_timestep,space_heating_power_timestep,solar_gains_timestep,solar_gains_indvidual_wall_timestep,heat_transfer_by_transmission_timestep,internal_heat_gains_timestep


"""defining initial temperature at timestep zero"""
T_building_previous_timestep=len(building_grid['id'])*[outside_temperature['t'].iloc[0]+273.15]


""" heating and cooling demand claculation for a year """
annual_space_heating_demand=[]
T_building_timeseries=[]
solar_gains_windows=[]
solar_gains_windows_indv_walls=[]
transmission_gains=[]
occupant_gains=[]
bar = progressbar.ProgressBar(maxval=8760).start()
for idx, timestep in enumerate(range(8760)):
    T_building,P_heating,solar_heat_gains_windows,solar_gains_indvidual_wall_timestep,heat_transfer_by_transmission_calculated,internal_heat_gains_calculated=calculate_timestep_single_building_demand(timestep,T_building_previous_timestep)
    T_building_timeseries.append(T_building)
    annual_space_heating_demand.append(P_heating)
    T_building_previous_timestep=T_building
    solar_gains_windows_indv_walls.append(solar_gains_indvidual_wall_timestep)
    solar_gains_windows.append(solar_heat_gains_windows)
    transmission_gains.append(heat_transfer_by_transmission_calculated)
    occupant_gains.append(internal_heat_gains_calculated)

    bar.update(idx)


""" exporting caluclated data to output file """
with open(project_dir/"output/heating_cooling_demand"/"T_building_timeseries.pickle", "wb") as handle:
    pickle.dump(T_building_timeseries, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(project_dir/"output/heating_cooling_demand"/"annual_space_heating_demand.pickle", "wb") as handle:
    pickle.dump(annual_space_heating_demand, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(project_dir/"output/heating_cooling_demand"/"solar_gains_windows.pickle", "wb") as handle:
    pickle.dump(solar_gains_windows, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(project_dir/"output/heating_cooling_demand"/"transmission_gains.pickle", "wb") as handle:
    pickle.dump(transmission_gains, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(project_dir/"output/heating_cooling_demand"/"occupant_gains.pickle", "wb") as handle:
    pickle.dump(occupant_gains, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(project_dir/"output/heating_cooling_demand"/"solar_gains_windows_indv_walls.pickle", "wb") as handle:
    pickle.dump(solar_gains_windows_indv_walls, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Defining the plot time interval 
doy=271
start_hour_plotting=doy*24+24
end_hour_plotting=start_hour_plotting+24

""" Plotting selected hours heating cooling demand """
# get variables needed for demand plot 
annual_HC_demand_plotting=[]
annual_solar_gains_plotting=[]
annual_inside_temperature=[]
for d in range(len(solar_gains_windows)):
    annual_solar_gains_plotting.append(solar_gains_windows[d][0]/1000)
    annual_HC_demand_plotting.append(annual_space_heating_demand[d][0])
    annual_inside_temperature.append(T_building_timeseries[d][0]-273.15)

# define x axis --> needed if you want to do fill_between 
x = np.arange(0, (end_hour_plotting-start_hour_plotting),int((end_hour_plotting-start_hour_plotting)/24))

fig, ax = plt.subplots(figsize=(8, 5))
solar_gains=ax.fill_between(x, annual_solar_gains_plotting[start_hour_plotting:end_hour_plotting], alpha=0.3, label='Solar heat gains',color='#FF6701')
hc_demand=ax.fill_between(x, [-1*element for element in annual_HC_demand_plotting[start_hour_plotting:end_hour_plotting]], alpha=0.8, label=' H/C demand', color='#A7C7E7')
ax.set_ylabel('Solar gain [kWh] /H/C demand[Wh/m$^{2}$]',fontname='cmr10',fontsize=12)
ax.set_xlabel('time [h]',fontname='cmr10',fontsize=12)
ax.set_title("Space Cooling demand for September, 29$^{th}$",fontname='cmr10')
ax2 = ax.twinx()
thermal_comfort=ax2.fill_between(x, 21, 24, color='gray', alpha=0.5, label='T$_{comfort}$')
T_out=ax2.plot(x,outside_temperature[start_hour_plotting:end_hour_plotting]['t'],color='red',label='Outside Temperature')
T_in=ax2.plot(x,annual_inside_temperature[start_hour_plotting:end_hour_plotting],label='Inside Temperature')
ax2.set_ylabel('Temperature [$^\circ$C]',fontname='cmr10',fontsize=12)  
plt.ylim(0, 30)
plt.xticks(x)
plt.xlim(0, (end_hour_plotting-start_hour_plotting)-1)

# Combine the legends from both axes 
handles1, labels1 = ax.get_legend_handles_labels()  # Legends from ax1 (sin(x))
handles2, labels2 = ax2.get_legend_handles_labels()  # Legends from ax2 (cos(x))

# Combine the handles and labels and create a single legend
ax.legend(handles=handles1 + handles2, labels=labels1 + labels2, loc='lower left',fontsize=6,prop={'family':'cmr10'})


plt.savefig(project_dir/"output/heating_cooling_demand"/f"space_heating_demand_plotting_{end_hour_plotting-start_hour_plotting}_hours.png")






"""Plotting cumulative solar radiation and showing contribution of each wall """

# Generate a gradient color by interpolating between two colors
def generate_color_gradients(n_colors=14):
    colors = []
    for i in range(n_colors):
        
        r = i / (n_colors - 1)
        color = plt.cm.OrRd(r)
        colors.append(color)
    return colors

# Generate the color gradients
color_gradients = generate_color_gradients()


# reformat data for plotting 
all_walls_time_series=[]
for wall in range(len(building_grid['walls'][0]['orientation'])):
    wall_time_series=[]
    for timestep in range(8760):
        wall_time_series.append(solar_gains_windows_indv_walls[timestep][0][wall])
    all_walls_time_series.append(wall_time_series)

# plotting 
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_ylabel('Solar gain [kWh]',fontname='cmr10',fontsize=12)
ax.set_title("Solar Heat Gain for September, 29$^{th}$",fontname='cmr10')
axw = int(end_hour_plotting-start_hour_plotting)*[0]
plots_ax=[]
for w in range(len(building_grid['walls'][0]['orientation'])):  # loogping over the number of walls in the selected building 
    axw_old = axw
    axw = [sum(x) for x in zip(axw, all_walls_time_series[w][start_hour_plotting:end_hour_plotting])]
    axw_diff = [sum(x) for x in zip(axw, [-x for x in axw_old])]
    wall_orientation = (90-(building_grid['walls'][0]['orientation'][w])) % 360
    if max(axw_diff)>0:
        t_annotate=axw_diff.index(max(axw_diff))
        ax.plot(axw, linestyle='--', color='k',linewidth=0.5)
        plot=ax.fill_between(np.arange(0, 24, 1), axw_old, axw, color=color_gradients[w],label=str("%.2f" % wall_orientation)+'$^\circ$')
        plots_ax.append(plot)
        plt.xticks(np.arange(0, 25, 2.0))
empty_line = plt.Line2D([], [], linestyle='None', color='none',label='') 
ax2 = ax.twinx()
plot1_ax2=ax2.plot(sun_alt_azimuth['altitude'][start_hour_plotting:end_hour_plotting].to_list(), linestyle='dotted',label='Sun Altitude')  
plot2_ax2=ax2.plot([(element+180)%360 for element in sun_alt_azimuth['azimuth'][start_hour_plotting:end_hour_plotting].to_list()],label='Sun Azimuth')      
ax2.set_ylabel('Angle [deg]',fontname='cmr10',fontsize=12)  
all_axis=plot1_ax2+plot2_ax2+[empty_line]+plots_ax
labs = [l.get_label() for l in all_axis]
leg=ax.legend(all_axis, labs, loc='upper right',fontsize=6,prop={'family':'cmr10'}, frameon=True,)
plt.savefig(project_dir/"output/heating_cooling_demand"/f"solar_radiation_{end_hour_plotting-start_hour_plotting}_hours.png")
