# resources: acceleration deceleration safe distance and speed calculations:
# https://copradar.com/chapts/references/acceleration.html
# https://arconforensics.com/system/ckeditor/attachment_files/184/the_dangers_of_speeding.pdf
# some examples: https://tasks.illustrativemathematics.org/content-standards/HSA/REI/B/4/tasks/586


from l5kit.data.filter import filter_tl_faces_by_status, filter_agents_by_distance
from l5kit.data.map_api import InterpolationMethod, MapAPI, TLFacesColors
from l5kit.geometry import rotation33_as_yaw, transform_points, geodetic_to_ecef
from l5kit.rasterization.box_rasterizer import get_box_world_coords
from enum import IntEnum
import numpy as np
from shapely.geometry import Polygon, Point
import torch
import math

INTERPOLATION_POINTS = 30
STEP_INTERPOLATION = INTERPOLATION_POINTS
MAX_POINTS_LANES = INTERPOLATION_POINTS
MAX_LANE_DISTANCE = 35  # maximum distance from the target agent to a lane
INTERP_METHOD = InterpolationMethod.INTER_ENSURE_LEN
MAX_CROSSWALKS = 20
MAX_POINTS_CW = 20
MAX_LANE_DISTANCE = 35
class RasterEls(IntEnum):  # map elements
    LANE_NOTL = 0
    ROAD = 1
    CROSSWALK = 2


class RewardClass():
    def __init__(self, mapAPI, train, init_target_agent_features):
        self.track_next_intersection = []   # empty list to keep track of the next intersection/node id so that in case of bidirectional lane this information can be used
        self.next_node_flag = True  # will be false if a node is not found in an episode
        self.init_target_agent_features = init_target_agent_features
        self.record_previous_target_speed = init_target_agent_features['speed']
        self.max_agents_distance = 35 # use configuration file(cfg) or retrieve from env
        self.mapAPI = mapAPI
        self.total_distance = 0  # for evaluation; to check the distance to the neighboring agent
        self.train = train
        self.safe_distance = 1
        self.prev_lane = init_target_agent_features['target_agent_lane']
        self.prev_lane_coord = self.mapAPI.get_lane_as_interpolation(
                    self.prev_lane, INTERPOLATION_POINTS, InterpolationMethod.INTER_ENSURE_LEN)
        #print('target agent track id:', init_target_agent_features['track_id'], 'target agent centroid:', init_target_agent_features['centroid'])
    def indices_in_bounds(self, center: np.ndarray, bounds: np.ndarray, half_extent: float) -> np.ndarray:
        """
        Get indices of elements for which the bounding box described by bounds intersects the one defined around
        center (square with side 2*half_side)

        Args:
            center (float): XY of the center
            bounds (np.ndarray): array of shape Nx2x2 [[x_min,y_min],[x_max, y_max]]
            half_extent (float): half the side of the bounding box centered around center

        Returns:
            np.ndarray: indices of elements inside radius from center
        """
        x_center, y_center = center[0]

        x_min_in = x_center > bounds[:, 0, 0] - half_extent
        y_min_in = y_center > bounds[:, 0, 1] - half_extent
        x_max_in = x_center < bounds[:, 1, 0]
        y_max_in = y_center < bounds[:, 1, 1] + half_extent
        return np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]


    def calculate_distance(self, d1, d2):
        dist = np.linalg.norm(d1, d2, axis = -1)
        return dist

    def calc_Ts(self, u:float, v: float, d:float)-> float:
        '''
        calculates the target speed given the distance between the agent and the stopping node/stop sign.
        :param u: current speed of the vehicle (speed in the frame)
        :param v: final speed = 0 as we consider the agent is going to stop at the stop sign/red light
        :param d: distance between the agent and the end node or stop sign
        :return Ts: target speed for the concerned frame
        '''
        '''
        procedure: assuming uniform deceleration after braking aiming at stopping the agent at a specific distance
        (stop sign/ node/intersection). first, determine the acceleration by using the formula sqr(v)= sqr(u)+2as
        and then find v at any time by using v=u+at
        '''
        sqr_v = v*v
        sqr_u = u*u
        a = (sqr_v - sqr_u)/(2*d)
        #a = 6.096 #m/s2
        #Ts = np.sqrt(2*a*d)
        # target seed for current frame
        Ts = u + a*0.1 # time interval for each fram is 0.1s

        return Ts

    def filter_agents_in_frame(self):
        '''
        requires current frame index and scene dataset to get the agents in the frame
        :return: all agents in the frame
        '''

        agents_interval = self.simulated_dataset.dataset.frames[self.frame_index]['agent_index_interval'] #_could also use recorded dataset
        agents_in_current_frame = self.simulated_dataset.dataset.agents[agents_interval[0]:agents_interval[1]]
        return agents_in_current_frame

    def filter_agents_within_threshold_distance(self):
        '''
        :returns: this function filters agents that are located ahead of the target agent and
        that are located within a maximum distance
        '''

        cur_agents = self.filter_agents_in_frame() # agents in current/observation frame
        #init_agent_centroid = self.init_target_agent_features['centroid']
        agent_centroid = self.curr_ego_state['centroid']
        all_centroids = cur_agents['centroid']
        #check = (all_centroids == init_agent_centroid).all(axis=1)
        # if check.any():
        #     print("The check_array exists in the main_array.")
        # else:
        #     print("The check_array does not exist in the main_array.")
        a = np.where(np.all(np.isin(all_centroids, agent_centroid), axis=1))  # index where the target agent is in the list of neighbors
        cur_agents = np.delete(cur_agents, a)   # remove target agent from the neighbor list
        agents_in_distance = filter_agents_by_distance(cur_agents, agent_centroid, self.max_agents_distance) # agents sorted

        return agents_in_distance

    def find_lane_coordinates(self, lane_id):
        if isinstance(lane_id, np.ndarray):
            lane_id = lane_id[0]
        lane_coords = self.mapAPI.get_lane_as_interpolation(
            lane_id, INTERPOLATION_POINTS, InterpolationMethod.INTER_ENSURE_LEN
        )
        agent_from_world = np.squeeze(self.curr_ego_state['agent_from_world'])
        xy_left = lane_coords["xyz_left"][:MAX_POINTS_LANES, :2]
        xy_right = lane_coords["xyz_right"][:MAX_POINTS_LANES, :2]
        xy_left = transform_points(xy_left, agent_from_world)
        xy_right = transform_points(xy_right, agent_from_world)
        lane_coords = np.concatenate((xy_left, xy_right))
        return lane_coords
    def distance_to_crosswalk(self):

        # first find all the crosswalks in the frame, then find the one that intersects agent's lane and lanes ahead
        ego_centroid = self.curr_ego_state['centroid']
        ta_lane = self.curr_ego_state['target_agent_lane']
        ta_lane_coord = self.find_lane_coordinates(ta_lane)
        ta_lane_ahead = self.mapAPI.__getitem__(ta_lane[0]).element.lane.lanes_ahead
        num_lane_ahead = len(ta_lane_ahead)
        if num_lane_ahead>0:
            lane_ahead_coords = np.zeros((num_lane_ahead,INTERPOLATION_POINTS*2,2))
            for i in range (num_lane_ahead):
                lane_ahead_coords[i] = self.find_lane_coordinates(ta_lane_ahead[i].id)
        crosswalks_bounds = self.mapAPI.bounds_info["crosswalks"]["bounds"]
        crosswalks_indices = self.indices_in_bounds(ego_centroid, crosswalks_bounds, MAX_LANE_DISTANCE)
        crosswalks_points = np.zeros((MAX_CROSSWALKS, MAX_POINTS_CW, 2), dtype=np.float32)
        crosswalks_availabilities = np.zeros_like(crosswalks_points[..., 0])
        dist_to_cw = np.zeros(len(crosswalks_indices))
        for i, xw_idx in enumerate(crosswalks_indices[:MAX_CROSSWALKS]):
            xw_id = self.mapAPI.bounds_info["crosswalks"]["ids"][xw_idx]
            points = self.mapAPI.get_crosswalk_coords(xw_id)["xyz"]
            # np.squeeze is used to reduce a dimension in case of ego agent_from_world
            points = transform_points(points[:MAX_POINTS_CW, :2], np.squeeze(self.curr_ego_state['agent_from_world']))
            n = len(points)
            crosswalks_points[i, :n] = points
            # find if this crosswalk is ahead of the target agent
            if points[0][0]>0:  # crosswalk is ahead of the agent
                # find if this crosswalk is on the ta_lane
                is_cw_on_lane = self.is_object_on_lane(points,ta_lane_coord)
                if is_cw_on_lane:
                    #find distance from target agent
                    dist_to_cw [i] = self.min_distance_to_polygon((0,0), points)

                else:
                    #check if the the cw is on the lanes ahead

                    for j in range(num_lane_ahead):
                        lane_coords = lane_ahead_coords[j]
                        lane_coords = np.squeeze(lane_ahead_coords[j])
                        is_cw_on_lane = self.is_object_on_lane(points, lane_coords)
                        if is_cw_on_lane:
                            dist_to_cw[i] = self.min_distance_to_polygon((0,0), points)
        # there might be multiple cw that are on the lanes ahead. taking the minimum of them as the stopping distance
        dist_to_cw_on_lane = dist_to_cw[dist_to_cw !=0]
        if dist_to_cw_on_lane != []:
            dist_to_cw_on_lane = min(dist_to_cw_on_lane)
        else:
            dist_to_cw_on_lane = 100   # meaning no cw on the ta lane or lanes ahead

        return dist_to_cw_on_lane
        #find the indices of the crosswalks within the area of the frame
        # nearest_crosswalk = 100 # setting a distance greater than the threshold distance
        # mindist = []
        # agent_centroid_x = agent['centroid'][0]
        #
        # for idx in self.indices_in_bounds(center_in_world, self.mapAPI.bounds_info["crosswalks"]["bounds"], MAX_LANE_DISTANCE):
        #     crosswalk = self.mapAPI.get_crosswalk_coords(self.mapAPI.bounds_info["crosswalks"]["ids"][idx])
        #     edge_points = crosswalk['xyz'][[0, -1], :1] #crosswalk has multiple points for the crosswalk line along X and Y coordinates. get only the first and last ones
        #     dist1 = np.abs(agent_centroid_x - edge_points[0])
        #     dist2 = np.abs(agent_centroid_x - edge_points[1])
        #     mindist.append(min(dist1,dist2))
        #
        # mindist = np.array(mindist)
        #
        # if len(mindist) == 0:
        #     return np.array(nearest_crosswalk)
        # else:
        #     nearest_crosswalk = min(mindist)
        #     return nearest_crosswalk
    def is_object_on_lane(self, object_coords, lane_coords):
        object_polygon = Polygon(object_coords)
        lane_polygon = Polygon(lane_coords)

        return object_polygon.intersects(lane_polygon)

    def min_distance_to_polygon(self, ref_point, polygon_coords):
        reference_point = Point(ref_point)
        polygon = Polygon(polygon_coords)
        distance = reference_point.distance(polygon)
        return distance

    def distance_to_intersection (self, agent_centroid, agent_lane):
        '''

        :param agent_centroid:
        :param agent_lane:
        :return: distance to next intersection
        '''
        dist_intersection = np.array([100]) #setting high value to ignore
        parent_seg_or_junc = self.mapAPI.__getitem__(agent_lane).element.lane.parent_segment_or_junction.id
        junction = self.mapAPI.__getitem__(parent_seg_or_junc).element.HasField('junction')
        segment = self.mapAPI.__getitem__(parent_seg_or_junc).element.HasField('segment')

        if segment:
            #print('segment')
            if self.mapAPI.__getitem__(agent_lane).element.lane.orientation_in_parent_segment == 2: #if lane direction in the segment is forward
                node_id = self.mapAPI.__getitem__(parent_seg_or_junc).element.segment.end_node.id
            elif self.mapAPI.__getitem__(agent_lane).element.lane.orientation_in_parent_segment == 3: # if lane direction in the segment is backward
                node_id = self.mapAPI.__getitem__(parent_seg_or_junc).element.segment.start_node.id
            elif self.mapAPI.__getitem__(agent_lane).element.lane.orientation_in_parent_segment == 1 or 4: # lane direction is two way
                return dist_intersection    #not appropriate
            node_loc = self.mapAPI.__getitem__(node_id).element.node.location  # get the location of the node related to the junction
            node_lat, node_lng = self.mapAPI._undo_e7(node_loc.lat_e7), self.mapAPI._undo_e7(node_loc.lng_e7)
            node_xyz = self.mapAPI.geo2world(node_lat, node_lng, 0) #ignore z by putting 0
            node_coord = node_xyz[0][:2]
            dist_intersection = np.linalg.norm(agent_centroid - node_coord)  # distance between agent and the end node of the segment

        if junction:
            #print('junction')
            loc = self.mapAPI.__getitem__(
                parent_seg_or_junc).bounding_box.south_west  # get one corner (lat long) of bounding box of the junction
            junc_lat, junc_lng = self.mapAPI._undo_e7(loc.lat_e7), self.mapAPI._undo_e7(loc.lng_e7)
            # print('loc:', loc)
            # print('junc_lat:',junc_lat)
            # print('junc_lat', junc_lng)
            xyz= self.mapAPI.geo2world(junc_lat, junc_lng, 0)  # set alt= 0; ignore and remove z coordinate
            node_coord = xyz[0][:2]
            dist_intersection = np.linalg.norm(agent_centroid - node_coord)
            #print('dist_int in junction:', dist_intersection)
        return dist_intersection

    def find_length_of_lane(self, lane_id, agent):
            lane = self.mapAPI.get_lane_as_interpolation(lane_id, STEP_INTERPOLATION, INTERP_METHOD)
            xy_left = lane["xyz_left"][:MAX_POINTS_LANES, :2]
            xy_right = lane["xyz_right"][:MAX_POINTS_LANES, :2]
            # convert coordinates into local space
            xy_left = transform_points(xy_left, agent['agent_from_world'])
            xy_right = transform_points(xy_right, agent['agent_from_world'])
            x = xy_left
            lane_length = np.amax(x) - np.amin(x)

            return lane_length

    # we may not need raster_from_world array

    #############################################################
    def distance_to_agent_front_back (self):
        '''

        :param target_agent_lane:
        :param center_in_world:
        :param agent:
        :param tl_faces:
        :return: distance to the agents in the same lane or lanes ahead

        fist find all the neighbors, then find if each of the neighbors are located in the lanes ahead
        '''
        # fist find ta lane coordinates and lane ahead coordinates
        ta_centroid = self.curr_ego_state['centroid']
        ta_lane = self.curr_ego_state['target_agent_lane']
        if isinstance(ta_lane, np.ndarray):
            ta_lane = ta_lane[0]
        ta_lane_coord = self.find_lane_coordinates(ta_lane)
        ta_lane_ahead = self.mapAPI.__getitem__(ta_lane).element.lane.lanes_ahead
        num_lane_ahead = len(ta_lane_ahead)
        if num_lane_ahead > 0:
            lane_ahead_coords = np.zeros((num_lane_ahead, INTERPOLATION_POINTS * 2, 2))
            for i in range(num_lane_ahead):
                lane_ahead_coords[i] = self.find_lane_coordinates(ta_lane_ahead[i].id)

        neighboring_agents = self.filter_agents_within_threshold_distance()
        neighbors_centroid = neighboring_agents['centroid']
        # find neighbors within safe distance
        dist = np.linalg.norm(neighbors_centroid - self.curr_ego_state['centroid'], axis=-1)
        # ignore the neighbor that has a distance less than 0.5. this is a hack. such a neighbor is appearing probably because
        # of error difference while matrix transformation.
        mask = dist >= 1
        # print('distance to the neighbors:', dist)
        # print('safe distance:', self.safe_distance)
        neighboring_agents = neighboring_agents[mask]
        nbr_agents_coord = get_box_world_coords(neighboring_agents)
        nbr_agent_coord_relative = transform_points(nbr_agents_coord, self.curr_ego_state['agent_from_world'])

        dist_to_agents = np.zeros(len(neighboring_agents))
        for i in range (len(neighboring_agents)):
            points = nbr_agent_coord_relative[i]

            # find if this agent is on the ta_lane
            is_agent_on_lane = self.is_object_on_lane(points, ta_lane_coord)
            if is_agent_on_lane:
                if points[0][0]<0:
                    agent_location = -1
                else:
                    agent_location = 1
                # find distance from target agent
                dist_to_agents[i] = self.min_distance_to_polygon((0, 0), points) * agent_location   # for precision add bounding box coord for target agent

            else:
                # check if the the cw is on the lanes ahead
                for j in range(num_lane_ahead):
                    lane_coords = lane_ahead_coords[j]
                    lane_coords = np.squeeze(lane_ahead_coords[j])
                    is_agent_on_lane = self.is_object_on_lane(points, lane_coords)
                    if is_agent_on_lane:
                        if points[0][0] < 0:
                            agent_location = -1
                        else:
                            agent_location = 1
                        dist_to_agents[i] = self.min_distance_to_polygon((0, 0), points) * agent_location
        has_agent_ahead = np.any(dist_to_agents >0)
        has_agent_behind = np.any(dist_to_agents <0)
        if has_agent_ahead:
            dist_to_agent_ahead = min(dist_to_agents[dist_to_agents>0])
        else: dist_to_agent_ahead = 100
        if has_agent_behind:
            dist_to_agent_behind = max(dist_to_agents[dist_to_agents<0])
        else: dist_to_agent_behind = 100

        return dist_to_agent_ahead, dist_to_agent_behind



    def find_target_speed(self):
        '''
        :param center_in_world: coordinate of the center of the frame to be predicted
        :param raster_from_world: for the frame to predict (next frame)
        :param tl_faces:
        :param agent: agent_to_predict_dict
        :return: target speed for the agent
        '''


        ego_lane_id = self.curr_ego_state['target_agent_lane']
        if isinstance(ego_lane_id, np.ndarray):
            ego_lane_id = ego_lane_id[0]
        # extract agent_lane details
        #print('agent_lane_id: ', agent_lane_id)
        parent_seg_or_junc = self.mapAPI.__getitem__(ego_lane_id).element.lane.parent_segment_or_junction.id

        # is it a jucntion? consider writing it in a separate function def
        junction = self.mapAPI.__getitem__(parent_seg_or_junc).element.HasField('junction')
        segment = self.mapAPI.__getitem__(parent_seg_or_junc).element.HasField('segment')
        ego_lane_detail = self.mapAPI.__getitem__(ego_lane_id).element

        target_speed = None
        # if it is a junction
        if junction:
            #print('target speed from junction')
            target_speed = 6   # this is hack. taking the ground truth speed from recorded dataset. beause junction does not provide any target speed
            # print('lane belongs to a junction')
            # print('target speed from junction')
            #target_speed = 4.17 # = 20 Km/h according to safe driving manual
            #print('target speed from junction', target_speed)
            # print('___INSIDE JUNCTION___')
            # loc = self.mapAPI.__getitem__(
            #     parent_seg_or_junc).bounding_box.south_west  # get one corner (lat long) of bounding box of the junction
            # junc_lat, junc_lng = self.mapAPI._undo_e7(loc.lat_e7), self.mapAPI._undo_e7(loc.lng_e7)
            # xyz = self.mapAPI.geo2world(junc_lat, junc_lng, 0)  # set alt= 0; ignore and remove z coordinate
            # target speed determination
            # ????what if the lane can also be used for turning as well as going through? -- Answer: the junction lane is a
            #lane in the middle of intersection. an agent on such a lane should move following the lane
            # if self.target_agent_dict['target_agent_lane_tl_status'] == 3: #and self.mapAPI.__getitem__(agent_lane_id).element.lane.turn_type_in_parent_junction == 1 or 2:  # 1 represents the agent going THROUGH because the lane is not for turning?
            #     target_speed = 0    # Red light, so stop
            #     print('first')
            # elif self.target_agent_dict['target_agent_lane_tl_status'] != 3 and self.mapAPI.__getitem__(agent_lane_id).element.lane.turn_type_in_parent_junction == 1:
            #     target_speed = self.record_previous_target_speed  # need to determine: speed limit of previous the lane of previous segment
            #     print('second')
            # elif (self.mapAPI.__getitem__(
            #         agent_lane_id).element.lane.turn_type_in_parent_junction == 2 or 3 or 4 or 5 or 6): # turn left/right/u turn
            #     target_speed = 4.17  # m/s   # assuming while turning the average speed in 15km/h
            #     print('third')
            # else:
            #     target_speed = self.record_previous_target_speed
            # print('target_speed:', target_speed)

        if segment:
            #print('target speed from segment')
            segment_detail = self.mapAPI.__getitem__(parent_seg_or_junc).element.segment

            if segment_detail.road_class == 1 or 2 or 3 or 4 or 5 or 6:  # the following road class has a speed limit annotated: motorway =1, trunk=2, primary =3, secondary=4, tertiary=5, residential=6
                # fisrt ensure target speed is assigned to the posted speed limit, least it goes unassigned
                # recommended speed for the lane
                rec_speed = self.mapAPI.__getitem__(parent_seg_or_junc).element.segment.speed_limit_meters_per_second
                #print('recorded speed:', rec_speed)
                target_speed = rec_speed
                # check if the agent lane is in forward lane set or backward lane set. a lane of a segment can have
                # one of 3 orientations: TWO_WAY(=1),ONE_WAY_FORWARD (=2), ONE_WAY_BACKWARD(=3). with this information
                # we get which direction the agent is moving with respect to the segment
                # first find end_node
                agent_orientation_in_segment = ego_lane_detail.lane.orientation_in_parent_segment
                if agent_orientation_in_segment == 2:  # forward
                    end_node_id = segment_detail.end_node.id    # end_node id is the upcoming intersection id(?)
                    self.track_next_intersection.append(end_node_id)
                elif agent_orientation_in_segment == 3:  # backward
                    end_node_id = segment_detail.start_node.id  # orientation in the segment is reverse
                    self.track_next_intersection.append(end_node_id)
                elif agent_orientation_in_segment == 1:     # 1 means the road segment is bi-directional, how to find the orientation of the agent?
                    # for this case, I did not get any way to find which direction the car is moving. So, will check if any of its
                    #previous frames has this information
                    if not self.track_next_intersection:    # if end_node_id not found yet call off this episode
                        self.next_node_flag = False
                    else: end_node_id = self.track_next_intersection[-1]  # the last end node id in the list. technically, all node id should be the same
                # find distance to the node
                if self.next_node_flag == True:
                    node_loc = self.mapAPI.__getitem__(end_node_id).element.node.location # location of the node
                    node_lat, node_lng = self.mapAPI._undo_e7(node_loc.lat_e7), self.mapAPI._undo_e7(node_loc.lng_e7)
                    node_xyz = self.mapAPI.geo2world(node_lat, node_lng, 0)  # review: altitude value is available, not 0
                    point1 = self.curr_ego_state['centroid']
                    point2 = node_xyz[0][:2]
                    dist_agent_node = np.linalg.norm(point1 - point2)  # distance between agent and the end node of the segment
                    agent_speed = self.curr_ego_state['speed']  # m/s; taking velocity in x direction only
                    dist_to_neighbors = self.distance_to_agent_front_back() # this function returns both ahead of back closes neighbour distances
                    dist_to_neighbor_in_front = dist_to_neighbors[0] # the first element of dist_to_neighbors is the distance to vehicle ahead
                    # print('dist to neighbor in front', dist_to_neighbor_in_front)
                    stopping_distance = min(dist_to_neighbor_in_front, dist_agent_node)

                    if self.curr_ego_state['target_agent_lane_tl_status'] == 3.0 and dist_to_neighbor_in_front < 2:
                        #print('target speed from 1')
                        target_speed = 0
                    elif self.curr_ego_state['target_agent_lane_tl_status'] == 3.0 and 2 < stopping_distance < MAX_LANE_DISTANCE: # 3 means Red light
                        # the agent speed should be the one from the initial frame along the way until the agent reaches
                        # to a stop. for every frame, the target speed will be calculated based on the initial velocity
                        # when the agent first detected the upcoming red light
                        # print('target speed from 2')
                        # print('stopping distance:', stopping_distance)
                        # print('distance to node:', dist_agent_node)
                        target_speed = self.calc_Ts(rec_speed, 0, stopping_distance)    # (u,v,d) v= 0 since the vehicle will have to stop at the red light
                        #print('target speed from 1', target_speed)
                    elif self.curr_ego_state['target_agent_lane_tl_status'] == 3.0 and stopping_distance >= MAX_LANE_DISTANCE:
                        #print('target speed from 3')
                        #print('stopping distance:', stopping_distance)
                        target_speed = self.mapAPI.__getitem__(parent_seg_or_junc).element.segment.speed_limit_meters_per_second
                        #print('target speed from 2',target_speed)
                        
                    elif self.curr_ego_state['target_agent_lane_tl_status'] !=3.0 and dist_to_neighbor_in_front < 2:
                        #print('target speed from 5')
                        target_speed = 0
                    # this one has be set becuase there was a case where no red light but a car in front stopped.
                    elif self.curr_ego_state['target_agent_lane_tl_status'] !=3.0 and 2 < dist_to_neighbor_in_front < MAX_LANE_DISTANCE:
                        #print('target speed from 4')
                        target_speed = self.calc_Ts(rec_speed, 0, dist_to_neighbor_in_front)


                    elif self.curr_ego_state['target_agent_lane_tl_status'] !=3.0 and dist_to_neighbor_in_front >= MAX_LANE_DISTANCE:
                        # print('target speed from 6')
                        target_speed = self.mapAPI.__getitem__(parent_seg_or_junc).element.segment.speed_limit_meters_per_second


            else:
                #print('inside SEGMENT. LANE IS NOT ONE OF THE MENTIONED ROAD CLASS')
                target_speed = self.record_previous_target_speed
                #print('target speed from previous frame', target_speed)
        if target_speed == None:
            #print('TARGET SPPED: NEITHER IN JUNCTION NOR IN SEGMENT ')
            target_speed = self.record_previous_target_speed
            #print('target speed from none', target_speed)
        #print('Target speed:', target_speed)
        return target_speed


    def find_target_dist(self, v):
        '''
        find the safe distance for the speed the agent is driving using 2s rule
        :param agent: the details of the agent: to extract current velocity
        :return: safe distacne ds-> float, obtained by using the laws of motion: s= vt
        '''
        #v = agent['history_velocities'][0][0]  # agent's velocity in the x direction
        if v < 0:       # when the agent moves opposit direction to the ego vehicle, the velocity is negative
            v = -v
        #print('v:', v)
        ds = v * 2    # t=2s for 2s rule
        #ds = ds * 0.1   # time between two frames
        if ds ==0:
            ds = 1 # considering a minimum safe distance for a vehicle is 1 meter
        return ds


    def lane_reward(self):
        '''
        # objective: is the agent becoming closer to lane boundary or going away from the shoulder line?
        # in a frame, if the lane belongs to a junction, discourage getting closer to lane boundary and penalize heavily for going outside lane
        # in a frame, if the lane belongs to a segment, discourage getting closer to the lane but not as heavily as that of junction-lane, and penalize
        # heavily if the agent goes outside the shoulder lane.
        # identify shoulder lane on the left and on the right
        '''
        # indentify agent lane
        ta_lane = self.curr_ego_state['target_agent_lane']
        if isinstance(ta_lane, np.ndarray):
            ta_lane = ta_lane[0]
        parent_seg_or_junc = self.mapAPI.__getitem__(ta_lane).element.lane.parent_segment_or_junction.id
        segment = self.mapAPI.__getitem__(parent_seg_or_junc).element.HasField('segment')
        junction = self.mapAPI.__getitem__(parent_seg_or_junc).element.HasField('junction')
        #print('+++++++++++ printing from lane reward function+++++++++++++++')
        #print('ta_lane:', ta_lane)
        #print('previous lane:', self.prev_lane)
        if segment:
            current_road_section = 'segment'
            #print('its a segment:', parent_seg_or_junc)
        elif junction:
            current_road_section = 'junction'
            #print('its a junction:', parent_seg_or_junc)
        #print('ta_lane:', ta_lane, 'previoius lane:', self.prev_lane)
        if ta_lane == self.prev_lane:  # for example the location predicted belongs to a lane form oposit direction
            #print('talane = prev_lane: rlane=0')
            R_lane = 0
            # print('ta_lane = self.prev_lane')
            return R_lane
            
        # if not the previous lane, means you changed the lane.
        # let's see how good the changed decision is
        # previous lane segment of junction
        prev_lane_seg_or_junc = self.mapAPI.__getitem__(self.prev_lane).element.lane.parent_segment_or_junction.id
        prev_segment = self.mapAPI.__getitem__(prev_lane_seg_or_junc).element.HasField('segment')
        prev_junction = self.mapAPI.__getitem__(prev_lane_seg_or_junc).element.HasField('junction')
        if prev_segment:
            prev_road_section = 'segment'
            #print('previous one was a segment:', prev_lane_seg_or_junc)
        elif prev_junction:
            prev_road_section = 'junction'
            #print('previous one was a junction:', prev_lane_seg_or_junc)
        # road section change: segment -> new segment or junction -> segment or segment -> junction
        if parent_seg_or_junc != prev_lane_seg_or_junc:
            #print('parent_seg_or_junc != prev_lane_seg_or_junc, rlane=0')

            R_lane = 0

            # print('parent_seg_or_junc != prev_lane_seg_or_junc:')
            return R_lane

        # it's not a new segment or junction: so lane change within a segment or a junction
        if prev_road_section=='junction' and current_road_section == 'junction':
            #print('both lanes are in the same junction, rlane=0')
            R_lane = 0    # cannot change lane within a junction
            # print('ta lane:', ta_lane, 'prev ta lane:', self.prev_lane)
            # print('prev_road_section==junction and current_road_section ==junction')
            return R_lane
        if prev_road_section == 'segment' and current_road_section == 'segment':
            #print('both lanes are in the same segment')
            # assuming in one frame duration (0.1s) an agent cannot drift to more than one lane width
            lane_ids_left = self.mapAPI.__getitem__(self.prev_lane).element.lane.adjacent_lane_change_left.id
            lane_ids_right = self.mapAPI.__getitem__(self.prev_lane).element.lane.adjacent_lane_change_left.id
            if ta_lane == lane_ids_left:
                #print('lane change left, rlane=0')
                R_lane = 0
            elif ta_lane == lane_ids_right:
                #print('lane change right, rlane=0')
                R_lane = 0
            else:
                # went outside the drivable region
                #print('changed lane but not in a valid location')
                R_lane = -10

            return R_lane

    def closest_angle_error(self, angle_a: torch.Tensor, angle_b: torch.Tensor) -> torch.Tensor:
        """ Finds the closest angle between angle_b - angle_a in radians.

        :param angle_a: a Tensor of angles in radians
        :param angle_b: a Tensor of angles in radians
        :return: The relative angle error between A and B between [0, pi]
        """
        assert angle_a.shape == angle_b.shape
        two_pi = 2.0 * math.pi
        wrapped = torch.fmod(angle_b - angle_a, two_pi)
        closest_angle = torch.fmod(2.0 * wrapped, two_pi) - wrapped
        return torch.abs(closest_angle)
    def get_reward(self, curr_ego_state, predicted_ego_state, sim_dataset, scene_index, frame_index):
        # some situations. (1) if the target agent is in a segmnent and there is no upcoming
        # stopping reason (stop sign or red light), the car needs to maintain a safe distance
        # from the car. this safe distance will depend on agent's current speed: 2s/3s rule.

        global yaw_rew
        recorded_ego_status = sim_dataset.recorded_scene_dataset_batch[0]
        self.curr_ego_state = curr_ego_state   # given environment
        self.recorded_dataset = sim_dataset.recorded_scene_dataset_batch[scene_index]
        self.simulated_dataset = sim_dataset.scene_dataset_batch[scene_index]
        self.frame_index = frame_index # observation frame index
        self.simulated_frame_index = frame_index + 1
        #self.gt_speed = gt_speed
        #self.record_previous_target_speed = self.target_agent_dict['speed']  # it is difficult to find the target speed at a junction since it is not explicitly given in the dataset
        dist_to_cw = self.distance_to_crosswalk()
        # in ta_status: in [0],[1] locations, in [2] yaw, in [3] speed, in [4], [5] velocity x and y                                                                  # this assignment is basically something to start with, for example if the simulation starts at a junction
        # later, after we get the target speed, for example from a segment, this value is reassigned

        # predicted_speed
        predicted_velocity = predicted_ego_state['velocities'] # m/0.1s
        speed = np.linalg.norm(predicted_velocity)*10   # m/s

        # target speed
        target_speed = self.find_target_speed() # what should be the speed based on the current observation (location, traffic light, distance to the neighbors etc)

        # find speed reward term
        beta = 10  # or use 5; see calculations in the notebook
        speed_diff = np.square((speed - target_speed)) / beta  # consider longitudinal velocity only
        speed_term = np.exp(-speed_diff)
        #print('speed term', speed_term)
        # find reward distance term
        # safe_distance
        self.safe_distance = self.find_target_dist(speed)

        # current_distance  from neighbor in front.
        dist_to_agent_ahead, dist_to_agent_behind = self.distance_to_agent_front_back()
        highest_dist_term = np.array([1])

        ### ta ahead lane tl face status
        ta_lane = self.curr_ego_state['target_agent_lane']
        tl_lanes_ahead_tl_status = self.curr_ego_state['ta_lanes_ahead_tl_status']
        tl_lanes_ahead_tl_status = np.any(tl_lanes_ahead_tl_status == 3)
        if self.curr_ego_state['target_agent_lane_tl_status'] == 3 or tl_lanes_ahead_tl_status: # if either of the two is red light, slow down and prepare to stop
            dist_to_agent_ahead = min(self.distance_to_crosswalk(), dist_to_agent_ahead)
        # dist_diff = abs(dist_to_agent_ahead - self.safe_distance)
        # dist_term = dist_diff/(target_speed*2)
        dist_term = min((dist_to_agent_ahead/self.safe_distance), highest_dist_term)
        # if dist_term ==0:
        #     dist_term = 1
        # else:
        #     dist_term = - dist_term
        #dist_back_ratio = dist_to_car_behind / self.safe_distance  # safe distance need to be changed based on the speed of the vehicle at back
        #dist_term = min(dist_front_ratio , highest_dist_term)
        #dist_term = min((dist_front_ratio), highest_dist_term)
        if not isinstance(dist_term, np.ndarray):
            dist_term = np.array([dist_term])

        lane_reward = self.lane_reward()

        ######## YAW REWARD. in case of segment, punish little, in case of junction lane, punish highly
        # find yaw error
        yaw_rew = 0
        recorded_yaw = np.array(recorded_ego_status[frame_index + 1]['yaw'])
        simulated_yaw  = np.squeeze(predicted_ego_state['yaws'])
        yaw_err = self.closest_angle_error(torch.Tensor(recorded_yaw), torch.Tensor(simulated_yaw))
        yaw_err = np.array(yaw_err)
        # scale yaw error differently for segment lane and junction lane
        # find if the previous lane belonged to a segment or a junction
        prev_lane_seg_or_junc = self.mapAPI.__getitem__(self.prev_lane).element.lane.parent_segment_or_junction.id
        prev_segment = self.mapAPI.__getitem__(prev_lane_seg_or_junc).element.HasField('segment')
        prev_junction = self.mapAPI.__getitem__(prev_lane_seg_or_junc).element.HasField('junction')
        #print('yaw error:', yaw_err)
        if prev_junction:
            #print('yaw reward from junction')
            yaw_rew = yaw_err
            #print('yaw reward:', yaw_rew)
        elif prev_segment:
            #print('yaw reward from segment')
            yaw_rew = 0
            #print('yaw reward:', yaw_rew)
        ############## displacement error in y direction##############
        #displacement error. this has to be one frame ahead. when current observation is 0, predicted observation is 1. so 1 should be taken
        # simulated_location = simulated_ta_status[self.simulated_frame_index][:2]   # taking only y value
        # observed_location = recorded_ta_status[self.simulated_frame_index][:2]
        #
        # simulated_location = transform_points(simulated_location.numpy().reshape(1,2), self.target_agent_dict['agent_from_world'])
        # observed_location = transform_points(observed_location.numpy().reshape(1,2), self.target_agent_dict['agent_from_world'])
        # simulated_location_y = simulated_location[0][-1]
        # observed_location_y = observed_location[0][-1]
        # displacement_error_y = np.absolute(simulated_location_y - observed_location_y)
        # #displacement_error = torch.cdist(simulated_point, observed_point)
        # exp_dist_e = np.exp(-displacement_error_y)
        # exp_dist_e = np.array([exp_dist_e], dtype=np.float32) # convert to numpy array
        #################################################################

        #Reward = dist_term * exp_term * exp_dist_e

        Reward = dist_term*speed_term + lane_reward - yaw_rew
        Reward = float(Reward)

        ################ print ###############
        # print('-------------print from reward-----------------')
        # print('Total Reward:', Reward)
        # print('dist_rew:', dist_term, 'speed rew:', speed_term, 'dist*speed:', dist_term*speed_term, 'lane rew:', lane_reward, 'yaw_rew:', yaw_rew)
        # print('speed:', speed, 'target speed:', target_speed)
        # print('safe distance:', self.safe_distance, 'distance to car ahead:', dist_to_car_in_front, 'distance to car behind:', dist_to_car_behind)

        #print('dist_front_ratio:', dist_front_ratio, 'dist back ratio:', dist_back_ratio)
        #print('lane reward:', lane_reward)

        if self.train:
            #reward_dict = {"Step_reward": Reward, "dist_nbr": dist_term, "speed": speed_term, "dist_y":exp_dist_e}
            reward_dict = {"Step_reward": Reward, "dist_nbr": dist_term, "speed": speed_term}
        # if not training (eval)
        if not self.train:
            print('dist to agent ahead:', dist_to_agent_ahead)
            print('dist to agent behind:', dist_to_agent_behind)
            if dist_to_agent_ahead == 100:
                distance = self.safe_distance
            else:
                distance = dist_to_agent_ahead
            if distance > self.safe_distance:
                distance = self.safe_distance
            #self.total_distance = self.total_distance + distance
            distance = distance/self.safe_distance
            # having a neighbor in front (within a threshold distance),how many times the ta maintained safe distance or to what extent it maintined safe distance
            #speed : difference between target speed and simulalted speed
            #simulated_speed = simulated_ta_status[self.simulated_frame_index]
            speed_error = abs(target_speed - speed) #
            if target_speed > speed:
                speed_below_rec = target_speed - speed
            elif target_speed < speed:
                speed_above_rec = speed - target_speed

            # how many times predicted speed is within a threshold of +/- 10km/h ~= 2.7 m/s
            if abs(target_speed - speed) <= 2.7:
                good_speed = 1
            else:
                good_speed = 0
            #lane :
            #reward_dict = {"Step_reward": Reward, "dist_nbr": distance}
            #reward_dict = {"Step_reward":Reward, "dist_nbr": distance, "speed_error":speed_error,"latitudinal_error" : displacement_error_y}
            reward_dict = {"Step_reward": Reward, "actual_to_safe_dist_ratio": distance, "speed_error": speed_error, "good_speed": good_speed
                        }
        #keep record for use in the next step
        self.record_previous_target_speed = target_speed
        self.prev_lane = self.curr_ego_state['target_agent_lane']
        if isinstance(self.prev_lane, np.ndarray):
            self.prev_lane = self.prev_lane[0]
        self.prev_lane_coord = self.mapAPI.get_lane_as_interpolation(
            self.prev_lane, INTERPOLATION_POINTS, InterpolationMethod.INTER_ENSURE_LEN)
        return reward_dict

    #_________________________________________________________________________________________________________
    #__________________________________________________________===============================================

    def find_vehicle_points_at_the_back(self):
        # step-1: find agents that are within the threshold distance
        # step-2: find agent that are at the back of the ta
        # step-3: find agents that fall in the current lane of the
        ta_centroid = self.curr_ego_state["centroid"]
        neighboring_agents = self.filter_agents_within_threshold_distance()
        # check if neighboring_agents is empty
        #neighbors_centroid = neighboring_agents['centroid']
        ta_travel_direct = self.ta_driving_direction() # gives the answer: centroid points increasing or decreasing?

        if ta_travel_direct == 'increasing':
            filtered_agents = neighboring_agents[neighboring_agents['centroid'][:,0] < ta_centroid[0]]    # compare x value
        else:
            filtered_agents = neighboring_agents[neighboring_agents['centroid'][:,0] > ta_centroid[0]]

        return filtered_agents

    def distance_to_agent_at_back(self):
        # there is no way to know the lanes at the back.
        # find the distance to the nearest neighbor that is located on the same lane as that of ta lane
        #agent_dist_at_back = np.array([100])
        agent_dist_at_back = self.safe_distance
        ta_lane = self.curr_ego_state['target_agent_lane']
        ta_lane_coords = self.mapAPI.get_lane_as_interpolation(
            ta_lane, INTERPOLATION_POINTS, InterpolationMethod.INTER_ENSURE_LEN
            )
        agents_at_back = self.find_vehicle_points_at_the_back() # ensures while filtering the agents that are moving in the same direction as that of ta
        # now find the agents that are on the same lane as that of ta
        #repeating code as that of the other function. consider making a separate function and call
        nbr_dist =[]
        nbr_list = []

        for i in range(len(agents_at_back)):
            lane_dist = np.linalg.norm(ta_lane_coords["xyz_midlane"][:, :2] - agents_at_back[i]['centroid'], axis=-1)
            min_dist = np.min(lane_dist) # minimum distance from the neighbor's centroid to all the mid-line points of the lane
            if min_dist < 2:    # lane width from the mid line
                nbr_dist.append(min_dist) # record the minimum distance
                nbr_list.append(agents_at_back[i])   # record the neighbor
        # find distance to the neighbors on the same lane or the lanes ahead
        if len(nbr_list) > 0:
            nbr_list = np.array(nbr_list)
            nbr_centroids = nbr_list['centroid']
            dist_to_nbrs_at_back = np.linalg.norm(self.curr_ego_state['centroid'] - nbr_centroids, axis=-1)

        return agent_dist_at_back if len(nbr_dist)==0 else np.array([min(dist_to_nbrs_at_back)])

    def find_lane_width(self, lane):

        lane_coords = self.mapAPI.get_lane_as_interpolation(
            lane, INTERPOLATION_POINTS, InterpolationMethod.INTER_ENSURE_LEN
        )
        mid_point = int(len(lane_coords['xyz_left']) / 2)
        dist_1 = np.linalg.norm(lane_coords['xyz_left'][0, :2] - lane_coords['xyz_right'][0, :2], axis=-1)
        dist_2 = np.linalg.norm(lane_coords['xyz_left'][-1, :2] - lane_coords['xyz_right'][-1, :2], axis=-1)
        dist_3 = np.linalg.norm(lane_coords['xyz_left'][mid_point, :2] - lane_coords['xyz_right'][mid_point, :2],
                                axis=-1)
        average_lane_width = np.mean((np.folat32(dist_1), np.float32(dist_2), np.float32(dist_3)))
        return average_lane_width

# the following part of code belonged to find dist to neighbors front and back

# set a distance that signifies the absense of a neighbour in front
        #agent_dist = np.array([100])  # setting a distance greater than the threshold distance
        # min_dist_front = np.array(self.safe_distance)
        # min_dist_back = np.array(self.safe_distance)
        # min_dist_front = np.array([100])
        # min_dist_back = np.array([100])
        # agent_dist = np.array((min_dist_front, min_dist_back))
        # ##### find target agent lane and the lanes ahead
        # target_agent_lane = self.target_agent_dict['target_agent_lane']
        # #is there a lane ahead of target_agent_lane?
        # lanes_ahead = self.mapAPI.__getitem__(target_agent_lane).element.lane.lanes_ahead  # will be empty if there is no lanes_ahead filed
        #
        # lane_ahead_id = []
        # if len(lanes_ahead) != 0:
        #     for i in range(len(lanes_ahead)):
        #         lane_ahead_id.append(lanes_ahead[i].id)    # there might be several lanes, but we are taking the first one only
        # nbr_dist = []
        # nbr_list = []
        #
        # ###### find all the neighbors of the target agent within the safe distance
        # neighboring_agents = self.filter_agents_within_threshold_distance()
        # # check if neighboring_agents is empty
        # neighbors_centroid = neighboring_agents['centroid']
        # # find neighbors within safe distance
        # dist = np.linalg.norm(neighbors_centroid - self.target_agent_dict['centroid'], axis = -1)
        # # ignore the neighbor that has a distance less than 0.5. this is a hack. such a neighbor is appearing probably because
        # # of error difference while matrix transformation.
        # mask = dist >= 0.5
        # #print('distance to the neighbors:', dist)
        # #print('safe distance:', self.safe_distance)
        # neighboring_agents = neighboring_agents[mask]
        # dist = dist[mask]
        # #dist = dist[dist <= self.safe_distance]
        # #print('distances less/= than safe distance:', dist)
        # #neighboring_agents = neighboring_agents[:len(dist)] # neighbors within safe distance
        # #print('neighbors within safe distance:', neighboring_agents)
        # lane_ahead_id.append(target_agent_lane)
        # lanes_indices = lane_ahead_id
        #
        # ####### check if the agents within the safe distance are in the same lane of the
        # ####### target agent or in a lane ahead
        # for idx, lane_idx in enumerate(lanes_indices):
        #     lane_coords = self.mapAPI.get_lane_as_interpolation(
        #     lane_idx, INTERPOLATION_POINTS, InterpolationMethod.INTER_ENSURE_LEN
        #     )
        #     for i in range(len(neighboring_agents)):
        #         lane_dist = np.linalg.norm(lane_coords["xyz_midlane"][:, :2] - neighboring_agents[i]['centroid'], axis=-1)
        #         min_dist = np.min(lane_dist) # minimum distance from the neighbor's centroid to all the mid-line points of the lane
        #         if min_dist < 2:    # lane width from the mid line
        #             nbr_dist.append(min_dist) # record the minimum distance
        #             nbr_list.append(neighboring_agents[i])   # record the neighbor
        # # find distance to the neighbors on the same lane or the lanes ahead
        #
        # if len(nbr_list) > 0:
        #     nbr_list = np.array(nbr_list)
        #     nbr_track_id = nbr_list['track_id']
        #     nbr_centroids = nbr_list['centroid']
        #     #print('neighbors track_ids:', nbr_track_id, "target agent track_id:", self.target_agent_dict['track_id'])
        #     # if ta_direction == 'increasing':
        #     #     nbr_ahead = nbr_centroids[nbr_centroids[:,0]>ta_centroid_x]
        #     #     nbr_behind = nbr_centroids[nbr_centroids[:,0]< ta_centroid_x]
        #     # elif ta_direction == 'decreasing':
        #     #     nbr_ahead = nbr_centroids[nbr_centroids[:, 0] < ta_centroid_x]
        #     #     nbr_behind = nbr_centroids[nbr_centroids[:, 0] > ta_centroid_x]
        #     # alternative way to find neighbor ahead and neighbor behind
        #     dist_to_neighbors = transform_points(nbr_centroids, self.target_agent_dict['agent_from_world'])
        #     # print('distance to neighbors:', dist_to_neighbors)
        #     dist = np.linalg.norm(np.array((0,0)) - dist_to_neighbors, axis=1)
        #     dist[dist_to_neighbors[:, 0] < 0 ] = -dist[dist_to_neighbors[:, 0] < 0]
        #     # print('dist in distance_to_agent_front_back():', dist)
        #     if np.any(dist>0):
        #         min_dist_front = np.min(dist[dist > 0])
        #     if np.any(dist < 0):
        #         min_dist_back = np.max(dist[dist < 0])
        #         min_dist_back = abs(min_dist_back)
        #     # if len(nbr_ahead)>0:
        #     #     dist_to_nbrs_in_front = np.linalg.norm(self.target_agent_dict['centroid'] - nbr_ahead, axis=-1)
        #     #     min_dist_front = min(dist_to_nbrs_in_front)
        #     # if len(nbr_behind)>0:
        #     #     dist_to_nbrs_at_back = np.linalg.norm(self.target_agent_dict['centroid'] - nbr_behind, axis=-1)
        #     #     min_dist_back = min(dist_to_nbrs_at_back)
        #     # print('all neighbour centroids:', nbr_centroids)
        #     # print('al neighbour track id:', nbr_list['track_id'])
        #     # print('distance to neighbors:', dist_to_nbrs)
        # # print('distance to the closest neighbor:', dist_to_nbrs)
        #
        # # nbr_list has the neighbors both at the front and the back. find agent at the front and at the back
        #
        # return agent_dist if len(nbr_dist)==0 else np.array((min_dist_front, min_dist_back))
