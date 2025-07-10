from typing import Dict, List, Optional

import numpy as np

from l5kit.data.filter import filter_agents_by_distance, filter_agents_by_distance_modified,filter_agents_by_labels, filter_tl_faces_by_status
from l5kit.data.filter import filter_agents_by_track_id, get_other_agents_ids
from l5kit.data.map_api import InterpolationMethod, MapAPI
from l5kit.geometry.transform import transform_points
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds, indices_in_bounds_modified
from sampling.agent_sampling import get_relative_poses, get_agents_relative_poses, compute_agent_velocity, compute_agent_velocity_modified
#from l5kit.environment.reward_pred.RewardClass import distance_to_intersection

class Vectorizer:
    """Object that processes parts of an input frame, and converts this frame to a vectorized representation - which
    can e.g. be fed as input to a DNN using the corresponding input format.

    """

    def __init__(self, cfg: dict, mapAPI: MapAPI):
        """Instantiates the class.

        Arguments:
            cfg: configs to load settings from
            mapAPI: mapAPI to query map information
        """
        self.lane_cfg_params = cfg["data_generation_params"]["lane_params"]
        self.mapAPI = mapAPI
        self.max_agents_distance = cfg["data_generation_params"]["max_agents_distance"]
        self.history_num_frames_agents = cfg["model_params"]["history_num_frames_agents"]
        self.future_num_frames = cfg["model_params"]["future_num_frames"]
        self.history_num_frames_max = max(cfg["model_params"]["history_num_frames_ego"], self.history_num_frames_agents)
        self.other_agents_num = cfg["data_generation_params"]["other_agents_num"]
        self.step_time = cfg["model_params"]["step_time"]

    def vectorize(self, selected_track_id: Optional[int], agent_centroid_m: np.ndarray, agent_yaw_rad: float,
                  agent_from_world: np.ndarray, history_frames: np.ndarray, history_agents: List[np.ndarray],
                  history_tl_faces: List[np.ndarray], history_position_m: np.ndarray, history_yaws_rad: np.ndarray,
                  history_velocity_x, history_velocity_y, history_availability: np.ndarray, future_frames: np.ndarray, future_agents: List[np.ndarray]) -> dict:
        """Base function to execute a vectorization process.

        Arguments:
            selected_track_id: selected_track_id: Either None for AV, or the ID of an agent that you want to
            predict the future of.
            This agent is centered in the representation and the returned targets are derived from their future states.
            agent_centroid_m: position of the target agent
            agent_yaw_rad: yaw angle of the target agent
            agent_from_world: inverted agent pose as 3x3 matrix
            history_frames: historical frames of the target frame
            history_agents: agents appearing in history_frames
            history_tl_faces: traffic light faces in history frames
            history_position_m: historical positions of target agent
            history_yaws_rad: historical yaws of target agent
            history_availability: availability mask of history frames
            future_frames: future frames of the target frame
            future_agents: agents in future_frames

        Returns:
            dict: a dict containing the vectorized frame representation
        """
        agent_features = self._vectorize_agents(selected_track_id, agent_centroid_m, agent_yaw_rad, agent_from_world,
                                                history_frames, history_agents, history_position_m, history_yaws_rad,
                                                history_velocity_x, history_velocity_y, history_availability, future_frames,
                                                future_agents)
        map_features = self._vectorize_map(agent_centroid_m, agent_from_world, history_tl_faces)
        return {**agent_features, **map_features}

    def _vectorize_agents(self, selected_track_id: Optional[int], agent_centroid_m: np.ndarray,
                          agent_yaw_rad: float, agent_from_world: np.ndarray, history_frames: np.ndarray,
                          history_agents: List[np.ndarray], history_position_m: np.ndarray,
                          history_yaws_rad: np.ndarray, history_velocity_x:np.ndarray,history_velocity_y:np.ndarray,
                          history_availability: np.ndarray, future_frames: np.ndarray,
                          future_agents: List[np.ndarray]) -> dict:
        """Vectorize agents in a frame.

        Arguments:
            selected_track_id: selected_track_id: Either None for AV, or the ID of an agent that you want to
            predict the future of.
            This agent is centered in the representation and the returned targets are derived from their future states.
            agent_centroid_m: position of the target agent
            agent_yaw_rad: yaw angle of the target agent
            agent_from_world: inverted agent pose as 3x3 matrix
            history_frames: historical frames of the target frame
            history_agents: agents appearing in history_frames
            history_tl_faces: traffic light faces in history frames
            history_position_m: historical positions of target agent
            history_yaws_rad: historical yaws of target agent
            history_availability: availability mask of history frames
            future_frames: future frames of the target frame
            future_agents: agents in future_frames

        Returns:
            dict: a dict containing the vectorized agent representation of the target frame
        """
        # compute agent features
        # sequence_length x 2 (two being x, y)
        agent_points = history_position_m.copy()
        # sequence_length x 1
        agent_yaws = history_yaws_rad.copy()
        #agent velocities
        agent_velocities_x = history_velocity_x.copy()
        agent_velocities_y = history_velocity_y.copy()

        # sequence_length x xy+yaw (3)
        agent_trajectory_polyline = np.concatenate([agent_points, agent_yaws], axis=-1)
        agent_velocity_yaw_polyline = np.concatenate([agent_velocities_x, agent_velocities_y, agent_yaws[:len(agent_velocities_x)]], axis=-1)
        agent_polyline_availability = history_availability[:len(agent_velocities_x)]

        #agent_vels_stacked = np.concat
        # get agents around AoI sorted by distance in a given radius. Give priority to agents in the current time step
        history_agents_flat = filter_agents_by_labels(np.concatenate(history_agents))
        history_agents_flat = filter_agents_by_distance(history_agents_flat, agent_centroid_m, self.max_agents_distance)

        cur_agents = filter_agents_by_labels(history_agents[0])
        # ADDED/MODIFIED -- get the distances as well to put into the state vector.
        cur_agents, cur_agents_dist = filter_agents_by_distance_modified(cur_agents, agent_centroid_m, self.max_agents_distance)

        # find distance to all agents relative to location of the target agent
        # vehicle behind has a distance with negative sign
        # relative_coords = transform_points(cur_agents['centroid'], agent_from_world)
        # relative_distances = []
        # for distance,point in zip(cur_agents_dist, relative_coords):
        #     if relative_coords[0]<0:
        #         relative_distances.append(-distance)
        #     else:
        #         relative_distances.append(distance)
        # relative_distances = np.array(relative_distances)
        ###
        # among all the agents from the last history frames, keep other_agents_num number of agents
        # priority given to those agents that are within the max_range distance in the current frame
        # so, list_agents_to_take has all the neighboring agents in the current frame and +
        list_agents_to_take = get_other_agents_ids(
            history_agents_flat["track_id"], cur_agents["track_id"], selected_track_id, self.other_agents_num
        )

        # Loop to grab history and future for all other agents
        #all_other_agents_history_centroids = np.zeros(
        #    (self.other_agents_num, self.history_num_frames_max + 1, 2), dtype=np.float32)      #ADDED
        #all_other_agents_history_DistToAgent = np.zeros(
        #    (self.other_agents_num, self.history_num_frames_max + 1, 2), dtype=np.float32)  # ADDED
        all_other_agents_history_velocities = np.zeros(
            (self.other_agents_num, self.history_num_frames_max, 2), dtype=np.float32)  # ADDED
        # all_other_agents_history_velocities_ = np.zeros(
        #     (self.other_agents_num, self.history_num_frames_max, 2), dtype=np.float32)
        all_other_agents_history_positions = np.zeros(
            (self.other_agents_num, self.history_num_frames_max + 1, 2), dtype=np.float32)
        all_other_agents_history_distances = np.zeros(
            (self.other_agents_num, self.history_num_frames_max +1, 1), dtype=np.float32)  # ADDED
        all_other_agents_history_yaws = np.zeros(
            (self.other_agents_num, self.history_num_frames_max + 1, 1), dtype=np.float32)
        all_other_agents_history_extents = np.zeros(
            (self.other_agents_num, self.history_num_frames_max + 1, 2), dtype=np.float32)
        all_other_agents_history_availability = np.zeros(
            (self.other_agents_num, self.history_num_frames_max + 1), dtype=np.float32)
        all_other_agents_types = np.zeros((self.other_agents_num,), dtype=np.int64)
        all_other_agents_track_ids = np.zeros((self.other_agents_num,), dtype=np.int64) #new from web


        all_other_agents_future_positions = np.zeros(
            (self.other_agents_num, self.future_num_frames, 2), dtype=np.float32)
        all_other_agents_future_distances = np.zeros(
            (self.other_agents_num, self.future_num_frames, 1), dtype=np.float32)
        all_other_agents_future_yaws = np.zeros((self.other_agents_num, self.future_num_frames, 1), dtype=np.float32)
        all_other_agents_future_extents = np.zeros((self.other_agents_num, self.future_num_frames, 2), dtype=np.float32)
        all_other_agents_future_availability = np.zeros(
            (self.other_agents_num, self.future_num_frames), dtype=np.float32)

        all_other_agents_future_velocities = np.zeros(
            (self.other_agents_num, self.future_num_frames, 2), dtype=np.float32)       #ADDED
        # all_other_agents_future_velocities_ = np.zeros(
        #     (self.other_agents_num, self.future_num_frames, 2), dtype=np.float32)

        for idx, track_id in enumerate(list_agents_to_take):
            (
                agent_history_coords_offset,
                agent_history_yaws_offset,
                agent_history_extent,
                agent_history_availability,
                agent_history_velocity
            ) = get_agents_relative_poses(self.history_num_frames_max + 1, history_frames, track_id, history_agents,
                                   agent_from_world, agent_yaw_rad)

            all_other_agents_history_positions[idx] = agent_history_coords_offset
            all_other_agents_history_distances[idx] = self.find_relative_distance(agent_history_coords_offset)
            all_other_agents_history_yaws[idx] = agent_history_yaws_offset
            all_other_agents_history_extents[idx] = agent_history_extent
            all_other_agents_history_availability[idx] = agent_history_availability
            # NOTE (@lberg): assumption is that an agent doesn't change class (seems reasonable)
            # We look from history backward and choose the most recent time the track_id was available.
            current_other_actor = filter_agents_by_track_id(history_agents_flat, track_id)[0]
            all_other_agents_types[idx] = np.argmax(current_other_actor["label_probabilities"])
            all_other_agents_track_ids[idx] = track_id

            #agent_distance = filter_agents_by_distance(agent_centroid_m,self.max_agents_distance) #ADDED

            (
                agent_future_coords_offset,
                agent_future_yaws_offset,
                agent_future_extent,
                agent_future_availability,
                agent_future_velocity
            ) = get_agents_relative_poses(
                self.future_num_frames, future_frames, track_id, future_agents, agent_from_world, agent_yaw_rad
            )
            all_other_agents_future_positions[idx] = agent_future_coords_offset
            all_other_agents_future_distances[idx] = self.find_relative_distance(agent_future_coords_offset)
            all_other_agents_future_yaws[idx] = agent_future_yaws_offset
            all_other_agents_future_extents[idx] = agent_future_extent
            all_other_agents_future_availability[idx] = agent_future_availability
            #ADDED velocities
            #all_other_agents_history_velocities[idx], all_other_agents_future_velocities[idx] = compute_agent_velocity_modified(all_other_agents_history_positions[idx], all_other_agents_future_positions[idx], self.step_time)
            # why taking first 3 of agent_history_velocity([:3])? because while taking ground truth velocity, we have all the records, we are not calculating velocity based on the locations in
            # frame by frame, like it is done in compute_agent_velocity(). in this function we get 3 velocity values when we have 4 frames, for example
            all_other_agents_history_velocities[idx], all_other_agents_future_velocities[idx] = agent_history_velocity[ :self.history_num_frames_max], agent_future_velocity # all the ground truth velocities of the agnets
        # crop similar to ego above
        all_other_agents_history_positions[:, self.history_num_frames_agents + 1:] *= 0
        all_other_agents_history_yaws[:, self.history_num_frames_agents + 1:] *= 0
        all_other_agents_history_extents[:, self.history_num_frames_agents + 1:] *= 0
        all_other_agents_history_availability[:, self.history_num_frames_agents + 1:] *= 0

        # compute other agents features
        # num_other_agents (M) x sequence_length x 2 (two being x, y)
        agents_points = all_other_agents_history_positions.copy()
        agents_velocities = all_other_agents_history_velocities.copy()
        # num_other_agents (M) x sequence_length x 1
        agents_yaws = all_other_agents_history_yaws.copy()
        agents_dist = all_other_agents_history_distances.copy()
        # agents_extents = all_other_agents_history_extents[:, :-1]
        # num_other_agents (M) x sequence_length x self._vector_length
        # other_agents_polyline = np.concatenate([agents_points, agents_yaws], axis=-1)
        other_agents_polyline = np.concatenate([agents_points, agents_dist], axis=-1)
        #other_agents_velocity_yaw_polyline = np.concatenate([agents_velocities, agents_yaws[:,:len(agents_velocities[0])]], axis=-1)
        #adding distance instead of yaws
        other_agents_velocity_yaw_polyline = np.concatenate([agents_velocities, agents_dist[:, :len(agents_velocities[0])]], axis=-1)
        other_agents_polyline_availability = all_other_agents_history_availability[:,:len(agent_velocities_x)]

        # positions are relative to the location of the target agent. so, history_positions[0] would be the
        # distance from the target agent to the agent in concern

        agent_dict = {
            "all_other_agents_history_positions": all_other_agents_history_positions,
            "all_other_agents_history_distances": all_other_agents_history_distances,   #ADDED
            "all_other_agents_history_yaws": all_other_agents_history_yaws,
            "all_other_agents_history_extents": all_other_agents_history_extents,
            "all_other_agents_history_velocities": all_other_agents_history_velocities, #ADDED
            "all_other_agents_history_availability": all_other_agents_history_availability.astype(np.bool),
            "all_other_agents_future_positions": all_other_agents_future_positions,
            "all_other_agents_future_distances":all_other_agents_future_distances,  #ADDED
            "all_other_agents_future_yaws": all_other_agents_future_yaws,
            "all_other_agents_future_extents": all_other_agents_future_extents,
            "all_other_agents_future_velocities": all_other_agents_future_velocities,
            "all_other_agents_future_availability": all_other_agents_future_availability.astype(np.bool),
            "all_other_agents_types": all_other_agents_types,
            "all_other_agents_track_ids": all_other_agents_track_ids,
            "agent_trajectory_polyline": agent_trajectory_polyline,
            "agent_velocity_yaw_polyline": agent_velocity_yaw_polyline,
            "agent_polyline_availability": agent_polyline_availability.astype(np.bool),
            "other_agents_polyline": other_agents_polyline,
            "other_agents_velocity_yaw_polyline": other_agents_velocity_yaw_polyline,
            "other_agents_polyline_availability": other_agents_polyline_availability.astype(np.bool),

        }

        return agent_dict

    def _vectorize_map(self, agent_centroid_m: np.ndarray, agent_from_world: np.ndarray,
                       history_tl_faces: List[np.ndarray]) -> dict:
        """Vectorize map elements in a frame.

        Arguments:
            agent_centroid_m: position of the target agent
            agent_from_world: inverted agent pose as 3x3 matrix
            history_tl_faces: traffic light faces in history frames

        Returns:
            dict: a dict containing the vectorized map representation of the target frame
        """
        # START WORKING ON LANES
        MAX_LANES = self.lane_cfg_params["max_num_lanes"]
        MAX_POINTS_LANES = self.lane_cfg_params["max_points_per_lane"]
        MAX_POINTS_CW = self.lane_cfg_params["max_points_per_crosswalk"]

        MAX_LANE_DISTANCE = self.lane_cfg_params["max_retrieval_distance_m"]
        INTERP_METHOD = InterpolationMethod.INTER_ENSURE_LEN  # split lane polyline by fixed number of points
        STEP_INTERPOLATION = MAX_POINTS_LANES  # number of points along lane
        MAX_CROSSWALKS = self.lane_cfg_params["max_num_crosswalks"]

        #lane_points has 60 lines for two sides for a lane; so 60 lines for 30 lanes: 30*2
        #lane_mid_points has only one line, so 30 lines for 30 lanes
        lanes_points = np.zeros((MAX_LANES * 2, MAX_POINTS_LANES, 2), dtype=np.float32)
        lanes_availabilities = np.zeros((MAX_LANES * 2, MAX_POINTS_LANES), dtype=np.float32)

        lanes_mid_points = np.zeros((MAX_LANES, MAX_POINTS_LANES, 2), dtype=np.float32)
        lanes_mid_availabilities = np.zeros((MAX_LANES, MAX_POINTS_LANES), dtype=np.float32)
        lanes_tl_feature = np.zeros((MAX_LANES, MAX_POINTS_LANES, 1), dtype=np.float32)
        #lanes_midpoint = np.zeros((MAX_LANES, 2, 2), dtype=np.float32) # midpoint of the lanes
        distance_to_lanes = np.zeros((MAX_LANES, 1), dtype=np.float32) # distance to the lanes (lanes' midpoints)
        # 8505 x 2 x 2
        lanes_bounds = self.mapAPI.bounds_info["lanes"]["bounds"]

        # filter first by bounds and then by distance, so that we always take the closest lanes
        # agent_centroid is at the centre and bouding boxes of lanes that lie inside MAX_LANE_DISTANCE in both forward and backward directions
        # are counted. it means a total of 2*MAX_LANE_DISTANCE
        lanes_indices = indices_in_bounds(agent_centroid_m, lanes_bounds, MAX_LANE_DISTANCE)
        distances = []

        for lane_idx in lanes_indices:
            lane_id = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]
            lane = self.mapAPI.get_lane_as_interpolation(lane_id, STEP_INTERPOLATION, INTERP_METHOD)
            lane_dist = np.linalg.norm(lane["xyz_midlane"][:, :2] - agent_centroid_m, axis=-1)
            distances.append(np.min(lane_dist))
        lanes_indices = lanes_indices[np.argsort(distances)]    # gives the lane that is the most close from the centroid
        # The lane id where the target agent is in
        if lanes_indices !=[]:
            agent_lane_id = self.mapAPI.bounds_info["lanes"]["ids"][lanes_indices[0]]
            self.previous_lane = agent_lane_id
        else:
            agent_lane_id = self.previous_lane
        # TODO: move below after traffic lights
        crosswalks_bounds = self.mapAPI.bounds_info["crosswalks"]["bounds"]
        # take the crosswalks that are only in front of the agent
        crosswalks_indices = indices_in_bounds_modified(agent_centroid_m, crosswalks_bounds, MAX_LANE_DISTANCE)
        crosswalks_points = np.zeros((MAX_CROSSWALKS, MAX_POINTS_CW, 2), dtype=np.float32)
        distance_to_crosswalks = np.zeros((MAX_CROSSWALKS,1), dtype=np.float32)
        crosswalks_availabilities = np.zeros_like(crosswalks_points[..., 0])

        for i, xw_idx in enumerate(crosswalks_indices[:MAX_CROSSWALKS]):
            xw_id = self.mapAPI.bounds_info["crosswalks"]["ids"][xw_idx]
            points = self.mapAPI.get_crosswalk_coords(xw_id)["xyz"]
            points = transform_points(points[:MAX_POINTS_CW, :2], agent_from_world) # transforms the 4 coordinates of the crosswalk with respect to the location of the targer agent space
            mid_point = self.find_midpoint(points)

            # distance to the midpoint
            dist_to_mid = np.linalg.norm(np.array(mid_point), axis=0) # check for result's accuracy
            dist_to_mid = dist_to_mid.astype(np.float32)
            if mid_point[0]<0:
                dist_to_mid = - dist_to_mid
            distance_to_crosswalks[i] = dist_to_mid #DISTANCE TO THE CROSSWALK (MID POINT OF CROSSWALK) FROM TARGET AGENT
            #distance to the midpoint
            n = len(points)                                                         # in terms of clarity where exactly the coordinates of the crosswalk are if we consider the coordinate of the
            crosswalks_points[i, :n] = points                                       # target agent is at (0,0) [description at 'agent_coordinate system']
            crosswalks_availabilities[i, :n] = True

        active_tl_faces = set(filter_tl_faces_by_status(history_tl_faces[0], "ACTIVE")["face_id"].tolist())
        active_tl_face_to_color: Dict[str, str] = {}
        for face in active_tl_faces:
            try:
                active_tl_face_to_color[face] = self.mapAPI.get_color_for_face(face).lower()  # TODO: why lower()?
            except KeyError:
                continue  # this happens only on KIRBY, 2 TLs have no match in the map

        for out_idx, lane_idx in enumerate(lanes_indices[:MAX_LANES]):
            lane_id = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]
            lane = self.mapAPI.get_lane_as_interpolation(lane_id, STEP_INTERPOLATION, INTERP_METHOD)

            xy_left = lane["xyz_left"][:MAX_POINTS_LANES, :2]
            xy_right = lane["xyz_right"][:MAX_POINTS_LANES, :2]
            # convert coordinates into local space
            xy_left = transform_points(xy_left, agent_from_world)
            xy_right = transform_points(xy_right, agent_from_world)

            num_vectors_left = len(xy_left)
            num_vectors_right = len(xy_right)

            lanes_points[out_idx * 2, :num_vectors_left] = xy_left
            lanes_points[out_idx * 2 + 1, :num_vectors_right] = xy_right

            lanes_availabilities[out_idx * 2, :num_vectors_left] = 1
            lanes_availabilities[out_idx * 2 + 1, :num_vectors_right] = 1

            midlane = lane["xyz_midlane"][:MAX_POINTS_LANES, :2]
            midlane = transform_points(midlane, agent_from_world)
            num_vectors_mid = len(midlane)

            lanes_mid_points[out_idx, :num_vectors_mid] = midlane
            lanes_mid_availabilities[out_idx, :num_vectors_mid] = 1
            # middle point of the lanes. We may not need it
            lanes_midpoint = lanes_mid_points[out_idx, int(len(lanes_mid_points[0])/2)]  # taking the middle point of midline as the midpoint of the lane
            distance_to_lanes[out_idx] = np.linalg.norm(lanes_midpoint, axis=0)
            if lanes_midpoint[0]<0:
                distance_to_lanes[out_idx] = -distance_to_lanes[out_idx]
            lanes_tl_feature[out_idx, :num_vectors_mid] = self.mapAPI.get_tl_feature_for_lane(
                lane_id, active_tl_face_to_color)   # what do the values mean? 4/0? from the function called: tl_color_to_priority_idx = {"unknown": 0, "green": 1, "yellow": 2, "red": 3, "none": 4}

        # disable all points over the distance threshold
        valid_distances = np.linalg.norm(lanes_points, axis=-1) < MAX_LANE_DISTANCE
        lanes_availabilities *= valid_distances
        valid_mid_distances = np.linalg.norm(lanes_mid_points, axis=-1) < MAX_LANE_DISTANCE
        lanes_mid_availabilities *= valid_mid_distances

        # 2 MAX_LANES x MAX_VECTORS x (XY + TL-feature)
        # -> 2 MAX_LANES for left and right
        lanes = np.concatenate([lanes_points, np.zeros_like(lanes_points[..., [0]])], axis=-1)
        # pad such that length is 3 (why 3?)
        crosswalks = np.concatenate([crosswalks_points, np.zeros_like(crosswalks_points[..., [0]])], axis=-1)
        # MAX_LANES x MAX_VECTORS x 3 (XY + 1 TL-feature)
        lanes_mid = np.concatenate([lanes_mid_points, lanes_tl_feature], axis=-1)

        #target agent lane tl status
        target_agent_lane_tl_status = lanes_mid[0][:,2]    # if 4, no traffic light, if 3 Red light
        red_or_yellow = (2 or 3) in target_agent_lane_tl_status
        if red_or_yellow:
            target_agent_lane_tl_status = 3 # if red or yellow in any of the 20 points, make it red
        else:
            target_agent_lane_tl_status = lanes_mid[0][0][2] # else, take the first entry in the column and make the status
                                                            # it can be green(1) or 4 (none) or 0 (unknown)
        #distance_to_next_intersection = distance_to_intersection(agent_centroid_m, target_agent_lane_id)
        # how to find distance to the traffic light? see protobuf if the location of traffic light is given. sTART FROM THE get_tl_feature_for_lane function





        return {
            #the first lane and lane_mid is the one where the target is.
            "lanes": lanes, # the first lane is the closest to the target agent
            "target_agent_lane" : agent_lane_id,
            "target_agent_lane_tl_status": target_agent_lane_tl_status,
            "lanes_availabilities": lanes_availabilities.astype(np.bool),
            "lanes_mid": lanes_mid, # the traffic light status of the first lanes_mid tells what is the current tl status
            "lanes_mid_availabilities": lanes_mid_availabilities.astype(np.bool),
            "crosswalks": crosswalks,
            "crosswalks_availabilities": crosswalks_availabilities.astype(np.bool),
            #"distance_to_traffic_light":
            "distance_to_crosswalks": distance_to_crosswalks,
            "distance_to_lanes":distance_to_lanes
        }

    def find_midpoint(self, vertices):
        '''
        finds the midpoint of a shape, specifically for a crosswalk or a lane
        :param vertices: 4 coordinates of the map element
        :return: x and y coordinates of the shape
        '''

        sum_x = 0
        sum_y = 0

        for vertex in vertices:
            x, y  = vertex
            sum_x += x
            sum_y += y

        num_vertices = len(vertices)
        x_midpoint = sum_x / num_vertices
        y_midpoint = sum_y / num_vertices

        return (x_midpoint, y_midpoint)

    def find_relative_distance(self, positions):
        '''

        :param positions: relative position of an agent relative to the target agent in agent coordinate system
        :return: relative distance of the agent from the target agent (negative distance means the agent is at the back)
        '''
        # the relative position will be zero if in a past frame the agent is absent
        # currently it is kept as is by following their approach (they treat the location as 0 as well)
        # if needed, the 0 distance can be populated with the distance with the last known frame
        dist = np.linalg.norm(positions, axis=1)
        relative_distances = []

        for i in range (len(positions)):
            if positions[i][0] < 0:
                relative_distances.append(-dist[i])
            else:
                relative_distances.append(dist[i])
        relative_distances = np.array(relative_distances).reshape(-1,1)
        return relative_distances


