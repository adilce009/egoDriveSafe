from typing import List, Optional, Tuple

import numpy as np

from l5kit.data import (filter_agents_by_labels, filter_tl_faces_by_frames, get_agents_slice_from_frames,
                    get_tl_faces_slice_from_frames)
from l5kit.data.filter import filter_agents_by_frames, filter_agents_by_track_id,filter_agents_by_track_id, filter_agents_by_distance
from l5kit.geometry import angular_distance, compute_agent_pose, rotation33_as_yaw, transform_points
from l5kit.kinematic import Perturbation
from l5kit.rasterization import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer, RenderContext
from l5kit.sampling.slicing import get_future_slice, get_history_slice



def get_agent_context(
        state_index: int,
        frames: np.ndarray,
        agents: np.ndarray,
        tl_faces: np.ndarray,
        history_num_frames: int,
        future_num_frames: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Slice zarr or numpy arrays to get the context around the agent onf interest (both in space and time)
    Args:
        state_index (int): frame index inside the scene
        frames (np.ndarray): frames from the scene
        agents (np.ndarray): agents from the scene
        tl_faces (np.ndarray): tl_faces from the scene
        history_num_frames (int): how many frames in the past to slice
        future_num_frames (int): how many frames in the future to slice
    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]
    """

    #  the history slice is ordered starting from the latest frame and goes backward in time., ex. slice(100, 91, -2)
    history_slice = get_history_slice(state_index, history_num_frames, 1, include_current_state=True)
    future_slice = get_future_slice(state_index, future_num_frames, 1)

    history_frames = frames[history_slice].copy()  # copy() required if the object is a np.ndarray
    future_frames = frames[future_slice].copy()

    sorted_frames = np.concatenate((history_frames[::-1], future_frames))  # from past to future

    # get agents (past and future)
    agent_slice = get_agents_slice_from_frames(sorted_frames[0], sorted_frames[-1])
    agents = agents[agent_slice].copy()
    # sync interval with the agents array
    history_frames["agent_index_interval"] -= agent_slice.start
    future_frames["agent_index_interval"] -= agent_slice.start
    history_agents = filter_agents_by_frames(history_frames, agents)
    future_agents = filter_agents_by_frames(future_frames, agents)

    # get traffic lights (past and future)
    tl_slice = get_tl_faces_slice_from_frames(sorted_frames[0], sorted_frames[-1])
    tl_faces = tl_faces[tl_slice].copy()
    # sync interval with the traffic light faces array
    history_frames["traffic_light_faces_index_interval"] -= tl_slice.start
    future_frames["traffic_light_faces_index_interval"] -= tl_slice.start
    history_tl_faces = filter_tl_faces_by_frames(history_frames, tl_faces)
    future_tl_faces = filter_tl_faces_by_frames(future_frames, tl_faces)

    return history_frames, future_frames, history_agents, future_agents, history_tl_faces, future_tl_faces



def get_agent_context_modified(
        state_index: int,
        frames: np.ndarray,
        agents: np.ndarray,
        tl_faces: np.ndarray,
        history_num_frames: int,
        future_num_frames: int

) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Slice zarr or numpy arrays to get the context around the agent onf interest (both in space and time)

    Args:
        state_index (int): frame index inside the scene
        frames (np.ndarray): frames from the scene
        agents (np.ndarray): agents from the scene
        tl_faces (np.ndarray): tl_faces from the scene
        history_num_frames (int): how many frames in the past to slice
        future_num_frames (int): how many frames in the future to slice

    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]
    """

    #  the history slice is ordered starting from the latest frame and goes backward in time., ex. slice(100, 91, -2)
    #print('state index:', state_index)
    history_slice = get_history_slice(state_index, history_num_frames, 1, include_current_state=True)
    future_slice = get_future_slice(state_index, future_num_frames, 1)

    history_frames = frames[history_slice].copy()  # copy() required if the object is a np.ndarray
    future_frames = frames[future_slice].copy()

    sorted_frames = np.concatenate((history_frames[::-1], future_frames))  # from past to future

    # get agents (past and future)
    agent_slice = get_agents_slice_from_frames(sorted_frames[0], sorted_frames[-1])
    history_agent_slice = get_agents_slice_from_frames (sorted_frames[0], sorted_frames[history_num_frames]) # ADDED
    future_agent_slice = get_agents_slice_from_frames (sorted_frames[history_num_frames+1], sorted_frames[-1]) #ADDED
    agents = agents[agent_slice].copy()
    # sync interval with the agents array
    history_frames["agent_index_interval"] -= agent_slice.start
    future_frames["agent_index_interval"] -= agent_slice.start
    history_agents = filter_agents_by_frames(history_frames, agents)
    future_agents = filter_agents_by_frames(future_frames, agents)

    # get traffic lights (past and future)
    tl_slice = get_tl_faces_slice_from_frames(sorted_frames[0], sorted_frames[-1])
    tl_faces = tl_faces[tl_slice].copy()
    # sync interval with the traffic light faces array
    history_frames["traffic_light_faces_index_interval"] -= tl_slice.start
    future_frames["traffic_light_faces_index_interval"] -= tl_slice.start
    history_tl_faces = filter_tl_faces_by_frames(history_frames, tl_faces)
    future_tl_faces = filter_tl_faces_by_frames(future_frames, tl_faces)

    return history_frames, future_frames, history_agents, future_agents, history_tl_faces, future_tl_faces, history_agent_slice, future_agent_slice # ADDED: the last term agent_slice. reason to keep track of original agent index reference


def compute_agent_velocity(
        history_positions_m: np.ndarray, future_positions_m: np.ndarray, step_time: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute estimated velocities by finite differentiation on future positions as(pos(T+t) - pos(T))/t.
    This simple approach gives less than 0.5% velocity difference
    compared to (pos(T+t) - pos(T-t))/2t on v1.1/sample.zarr.tar.

    Args:
        history_positions_m (np.ndarray): history XY positions in meters
        future_positions_m (np.ndarray): future XY positions in meters
        step_time (np.ndarray): length of a step in second

    Returns:
        Tuple[np.ndarray, np.ndarray]: history and future XY speeds

    """
    assert step_time > np.finfo(float).eps, f"step_time must be greater then eps, got {step_time}"

    # sorted from current to future frame
    # [future_num_frames, 2]
    future_positions_diff_m = np.concatenate((future_positions_m[:1], np.diff(future_positions_m, axis=0)))
    # [future_num_frames, 2]
    future_vels_mps = np.float32(future_positions_diff_m / step_time)

    # sorted from current to past frame, current position is included in history positions
    # [history_num_frames, 2]
    history_positions_diff_m = -np.diff(history_positions_m, axis=0)
    # [history_num_frames, 2]
    history_vels_mps = np.float32(history_positions_diff_m / step_time) #ADDED to calculate acc
    history_vels_diff_m = -np.diff(history_vels_mps, axis=0) #ADDED
    #history_acc_mpss= np.float32(history_vels_diff_m/step_time) # acceleration
    return history_vels_mps, future_vels_mps    #,history_acc_mpss -- this return param was in the second position

def compute_agent_velocity_modified(
        history_positions_m: np.ndarray, future_positions_m: np.ndarray, step_time: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    # this function has some modification to accommodate the changes of the number of frames
    specially when the future number of frames = 1, we do not have enough frames to computer
    future frame velocity.

    Compute estimated velocities by finite differentiation on future positions as(pos(T+t) - pos(T))/t.
    This simple approach gives less than 0.5% velocity difference
    compared to (pos(T+t) - pos(T-t))/2t on v1.1/sample.zarr.tar.

    Args:
        history_positions_m (np.ndarray): history XY positions in meters
        future_positions_m (np.ndarray): future XY positions in meters
        step_time (np.ndarray): length of a step in second

    Returns:
        Tuple[np.ndarray, np.ndarray]: history and future XY speeds

    """
    assert step_time > np.finfo(float).eps, f"step_time must be greater then eps, got {step_time}"

    # sorted from current to future frame
    # [future_num_frames, 2]
    future_positions_diff_m = np.concatenate((history_positions_m[:1],future_positions_m[:1] ))
    #print('future position diff:', future_positions_diff_m)
    # [future_num_frames, 2]
    future_positions_diff_m = np.diff(future_positions_diff_m, axis=0)
    future_vels_mps = np.float32(future_positions_diff_m / step_time)

    # sorted from current to past frame, current position is included in history positions
    # [history_num_frames, 2]
    history_positions_diff_m = -np.diff(history_positions_m, axis=0)
    # [history_num_frames, 2]
    history_vels_mps = np.float32(history_positions_diff_m / step_time) #ADDED to calculate acc
    history_vels_diff_m = -np.diff(history_vels_mps, axis=0) #ADDED
    #history_acc_mpss= np.float32(history_vels_diff_m/step_time) # acceleration

    '''
    fut_all_x = future_vels_mps[:,0]
    fut_all_y = future_vels_mps[:,1]
    hist_all_x = history_vels_mps[:,0]
    hist_all_y = history_vels_mps[:,1]

    threshold = 20
    is_any_greater = np.any(fut_all_x > threshold)
    if is_any_greater:
        print('fut_x got a greater value than threshold 20')
        print('before:', fut_all_x)
        fut_all_x = treat_outlier(fut_all_x, threshold)
        future_vels_mps[:, 0] = fut_all_x
        print('after:', fut_all_x)

    is_any_greater = np.any(fut_all_y > threshold)
    if is_any_greater:
        print('fut_y got a greater value than threshold 20')
        print('before:', fut_all_y)
        fut_all_y = treat_outlier(fut_all_y, threshold)
        future_vels_mps[:, 1] = fut_all_y
        print('after:', fut_all_y)

    is_any_greater = np.any(hist_all_x > threshold)
    if is_any_greater:
        print('hist_x got a greater value than threshold 20')
        print('before:', hist_all_x)
        hist_all_x = treat_outlier(hist_all_x, threshold)
        history_vels_mps[:, 0] = hist_all_x
        print('after:', hist_all_x)

    is_any_greater = np.any(hist_all_y > threshold)
    if is_any_greater:
        print('hist_y got a greater value than threshold 20')
        print('before:', hist_all_y)
        hist_all_y = treat_outlier(hist_all_y, threshold)
        history_vels_mps[:, 1] = hist_all_y
        print('after:', hist_all_y)
    '''
    return history_vels_mps, future_vels_mps    #,history_acc_mpss -- this return param was in the second position

# normalize outlier in velocity
def treat_outlier(vels, th):
    threshold = th
    positions = np.where(vels > threshold)
    values_to_average = vels[vels <= threshold]
    average_value = np.mean(values_to_average)
    vels[positions] = average_value

    return vels

def get_relative_poses(
        num_frames: int,
        frames: np.ndarray,
        selected_track_id: Optional[int],
        agents: List[np.ndarray],
        agent_from_world: np.ndarray,
        current_agent_yaw: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal function that creates the targets and availability masks for deep prediction-type models.
    The futures/history offset (in meters) are computed. When no info is available (e.g. agent not in frame)
    a 0 is set in the availability array (1 otherwise).
    Note: return dtype is float32, even if the provided args are float64. Still, the transformation
    between space is performed in float64 to ensure precision
    Args:
        num_frames (int): number of offset we want in the future/history
        frames (np.ndarray): available frames. This may be less than num_frames
        selected_track_id (Optional[int]): agent track_id or AV (None)
        agents (List[np.ndarray]): list of agents arrays (same len of frames)
        agent_from_world (np.ndarray): local from world matrix
        current_agent_yaw (float): angle of the agent at timestep 0
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: position offsets, angle offsets, extent, availabilities
    """
    # How much the coordinates differ from the current state in meters.
    positions_m = np.zeros((num_frames, 2), dtype=agent_from_world.dtype)
    yaws_rad = np.zeros((num_frames, 1), dtype=np.float32)
    extents_m = np.zeros((num_frames, 2), dtype=np.float32)
    availabilities = np.zeros((num_frames,), dtype=np.float32)
    agent_velocity = np.zeros((num_frames, 2), dtype=np.float32)

    for i, (frame, frame_agents) in enumerate(zip(frames, agents)):
        if selected_track_id is None:
            agent_centroid_m = frame["ego_translation"][:2]
            agent_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
            agent_extent = (EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH)

        else:
            # it's not guaranteed the target will be in every frame
            try:
                agent = filter_agents_by_track_id(frame_agents, selected_track_id)[0]
                agent_centroid_m = agent["centroid"]
                agent_yaw_rad = agent["yaw"]
                agent_extent = agent["extent"][:2]
                velocity = agent["velocity"]

            except IndexError:
                availabilities[i] = 0.0  # keep track of invalid futures/history
                continue

        positions_m[i] = agent_centroid_m
        yaws_rad[i] = agent_yaw_rad
        extents_m[i] = agent_extent
        availabilities[i] = 1.0
        #agent_velocity[i] = velocity
    # batch transform to speed up
    positions_m = transform_points(positions_m, agent_from_world) * availabilities[:, np.newaxis]
    yaws_rad = angular_distance(yaws_rad, current_agent_yaw) * availabilities[:, np.newaxis]
    return positions_m.astype(np.float32), yaws_rad, extents_m, availabilities # agent velocity obtained from the dataset is added. could include in the modified version of this function but did not want to make changes on that function to avoid further chance of error

def get_agents_relative_poses(
        num_frames: int,
        frames: np.ndarray,
        selected_track_id: Optional[int],
        agents: List[np.ndarray],
        agent_from_world: np.ndarray,
        current_agent_yaw: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal function that creates the targets and availability masks for deep prediction-type models.
    The futures/history offset (in meters) are computed. When no info is available (e.g. agent not in frame)
    a 0 is set in the availability array (1 otherwise).
    Note: return dtype is float32, even if the provided args are float64. Still, the transformation
    between space is performed in float64 to ensure precision
    Args:
        num_frames (int): number of offset we want in the future/history
        frames (np.ndarray): available frames. This may be less than num_frames
        selected_track_id (Optional[int]): agent track_id or AV (None)
        agents (List[np.ndarray]): list of agents arrays (same len of frames)
        agent_from_world (np.ndarray): local from world matrix
        current_agent_yaw (float): angle of the agent at timestep 0
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: position offsets, angle offsets, extent, availabilities
    """
    # How much the coordinates differ from the current state in meters.
    positions_m = np.zeros((num_frames, 2), dtype=agent_from_world.dtype)
    yaws_rad = np.zeros((num_frames, 1), dtype=np.float32)
    extents_m = np.zeros((num_frames, 2), dtype=np.float32)
    availabilities = np.zeros((num_frames,), dtype=np.float32)
    agent_velocity = np.zeros((num_frames, 2), dtype=np.float32)

    for i, (frame, frame_agents) in enumerate(zip(frames, agents)):
        if selected_track_id is None:
            agent_centroid_m = frame["ego_translation"][:2]
            agent_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
            agent_extent = (EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH)

        else:
            # it's not guaranteed the target will be in every frame
            try:
                agent = filter_agents_by_track_id(frame_agents, selected_track_id)[0]
                agent_centroid_m = agent["centroid"]
                agent_yaw_rad = agent["yaw"]
                agent_extent = agent["extent"][:2]
                velocity = agent["velocity"]

            except IndexError:
                availabilities[i] = 0.0  # keep track of invalid futures/history
                continue

        positions_m[i] = agent_centroid_m
        yaws_rad[i] = agent_yaw_rad
        extents_m[i] = agent_extent
        availabilities[i] = 1.0
        agent_velocity[i] = velocity
    # batch transform to speed up
    positions_m = transform_points(positions_m, agent_from_world) * availabilities[:, np.newaxis]
    yaws_rad = angular_distance(yaws_rad, current_agent_yaw) * availabilities[:, np.newaxis]
    return positions_m.astype(np.float32), yaws_rad, extents_m, availabilities, agent_velocity  # agent velocity obtained from the dataset is added. could include in the modified version of this function but did not want to make changes on that function to avoid further chance of error


'''
def get_relative_poses_modified(
        num_frames: int,
        frames: np.ndarray,
        selected_track_id: Optional[int],
        agents: List[np.ndarray],
        agent_from_world: np.ndarray,
        current_agent_yaw: float,
        agent_slice: slice  # ADDED: keep track of original agent index reference. why required? because get_agent_context
                            # function removes original index and replace with relative (within the scene)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal function that creates the targets and availability masks for deep prediction-type models.
    The futures/history offset (in meters) are computed. When no info is available (e.g. agent not in frame)
    a 0 is set in the availability array (1 otherwise).

    Note: return dtype is float32, even if the provided args are float64. Still, the transformation
    between space is performed in float64 to ensure precision

    Args:
        num_frames (int): number of offset we want in the future/history
        frames (np.ndarray): available frames. This may be less than num_frames
        selected_track_id (Optional[int]): agent track_id or AV (None)
        agents (List[np.ndarray]): list of agents arrays (same len of frames)
        agent_from_world (np.ndarray): local from world matrix
        current_agent_yaw (float): angle of the agent at timestep 0

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: position offsets, angle offsets, extent, availabilities

    """
    # How much the coordinates differ from the current state in meters.
    positions_m = np.zeros((num_frames, 2), dtype=agent_from_world.dtype)
    yaws_rad = np.zeros((num_frames, 1), dtype=np.float32)
    extents_m = np.zeros((num_frames, 2), dtype=np.float32)
    availabilities = np.zeros((num_frames,), dtype=np.float32)
    agent_indices_original = np.zeros((num_frames,), dtype = np.float32) #ADDED

    # the following lines are for finding original agent index
    agent_index_original_start = agent_slice.start
    agent_index_relative_start = frames[0]['agent_index_interval'][0]
    for i, (frame, frame_agents) in enumerate(zip(frames, agents)):
        if selected_track_id is None:
            agent_centroid_m = frame["ego_translation"][:2]
            agent_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
            agent_extent = (EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH)
        else:
            # it's not guaranteed the target will be in every frame

            try:
                index = filter_agents_by_track_id(frame_agents, selected_track_id)  # ADDED index
                agent_centroid_m = agent["centroid"]
                agent_yaw_rad = agent["yaw"]
                agent_extent = agent["extent"][0][:2]
                #agent_velocity = agent["velocity"]
                frame_agent_index_start = frames[i]['agent_index_interval'][0]
                index_diff_from_first_agent = frame_agent_index_start - agent_index_relative_start  # ADDED. how far the starting agent index is from the beginning of this batch
                agent_index_diff_relative = index_diff_from_first_agent + index # ADDED how far the agent that matches track_id
                agent_index_original = agent_index_original_start + agent_index_diff_relative # how far the agent is in terms of original indexing
            except IndexError:
                availabilities[i] = 0.0  # keep track of invalid futures/history
                agent_indices_original[i] = 0.0
                continue
        #print('i:', i, ' frame time step:', frame['timestamp'], 'agent centroid :', agent_centroid_m)

        positions_m[i] = agent_centroid_m
        yaws_rad[i] = agent_yaw_rad
        extents_m[i] = agent_extent
        availabilities[i] = 1.0
        agent_indices_original[i] = agent_index_original    # ADDED for original index
    #print('agent from world in modified context function:', agent_from_world)
   # print('_____________________________________')
    # batch transform to speed up
    positions_m = transform_points(positions_m, agent_from_world) * availabilities[:, np.newaxis]
    yaws_rad = angular_distance(yaws_rad, current_agent_yaw) * availabilities[:, np.newaxis]
    return positions_m.astype(np.float32), yaws_rad, extents_m, availabilities, agent_indices_original
'''

def generate_agent_sample(
        state_index: int,
        frames: np.ndarray,
        agents: np.ndarray,
        tl_faces: np.ndarray,
        selected_track_id: Optional[int],
        render_context: RenderContext,
        history_num_frames: int,
        future_num_frames: int,
        step_time: float,
        filter_agents_threshold: float,
        rasterizer: Rasterizer,
        perturbation: Optional[Perturbation] = None,
) -> dict:
    """Generates the inputs and targets to train a deep prediction model. A deep prediction model takes as input
    the state of the world (here: an image we will call the "raster"), and outputs where that agent will be some
    seconds into the future.

    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.

    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        tl_faces (np.ndarray): The full traffic light faces array, can be numpy array or a zarr array
        selected_track_id (Optional[int]): Either None for AV, or the ID of an agent that you want to
        predict the future of. This agent is centered in the raster and the returned targets are derived from
        their future states.
        render_context (RenderContext): The context for rasterisation
        history_num_frames (int): Amount of history frames to draw into the rasters
        future_num_frames (int): Amount of history frames to draw into the rasters
        step_time (float): seconds between consecutive steps
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent
        rasterizer Rasterizer: Rasterizer of some sort that draws a map image
        perturbation (Optional[Perturbation]): Object that perturbs the input and targets, used
        to train models that can recover from slight divergence from training set data

    Raises:
        IndexError: An IndexError is returned if the specified ``selected_track_id`` is not present in the scene
        or was filtered by applying the ``filter_agent_threshold`` probability filtering.

    Returns:
        dict: a dict object with the raster array, the future offset coordinates (meters),
        the future yaw angular offset, the future_availability as a binary mask
    """
    (
        history_frames,
        future_frames,
        history_agents,
        future_agents,
        history_tl_faces,
        future_tl_faces,
    ) = get_agent_context(state_index, frames, agents, tl_faces, history_num_frames, future_num_frames)  # selected_track_id is aded

    if perturbation is not None and len(future_frames) == future_num_frames:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    if selected_track_id is None:
        agent_centroid_m = cur_frame["ego_translation"][:2]
        agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent_m = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
        selected_agent = None
    else:
        # this will raise IndexError if the agent is not in the frame or under agent-threshold
        # this is a strict error, we cannot recover from this situation
        filtered_agents = filter_agents_by_labels(cur_agents, filter_agents_threshold)
        agent = filter_agents_by_track_id(filtered_agents, selected_track_id)[0]
        agent_centroid_m = agent["centroid"]
        #agent_yaw_rad = np.expand_dims(np.array(agent["yaw"]), axis =0)
        agent_yaw_rad = float(agent["yaw"])
        agent_extent_m = agent["extent"]
        selected_agent = agent

        ############ following portion is added to find the agents in within radius of safe distance
        # find all agents in the frame
        frame_agents = history_agents[0]
        filtered_agents = filter_agents_by_labels(frame_agents)
        agent = filter_agents_by_track_id(filtered_agents, selected_track_id)[0]
        agent_centroid = agent["centroid"]
        filtered_agents_centroids = filtered_agents['centroid']
        agents_ahead = filtered_agents[agent_centroid[0] < filtered_agents_centroids[:,0]]
        # filter agents by distance
        agents_safe_dist = filter_agents_by_distance(agents_ahead, agent_centroid, 40)  # safe dist is set to 40 m
        # filter agents that are in the forward direction   # check if the lane forward or backword
        # this is done by
        ############ ends here ###################


    # added center_in_world and sem_rasterize object since the reward function needs this from semantic rasterizer
    input_im, center_in_world, sem_rasterize = rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)

    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    agent_from_world = np.linalg.inv(world_from_agent)
    raster_from_world = render_context.raster_from_world(agent_centroid_m, agent_yaw_rad)

    future_positions_m, future_yaws_rad, future_extents, future_availabilities= get_relative_poses(
        future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, agent_yaw_rad,
    )
    # history_num_frames + 1 because it also includes the current frame
    history_positions_m, history_yaws_rad, history_extents, history_availabilities = get_relative_poses(
        history_num_frames + 1, history_frames, selected_track_id, history_agents, agent_from_world, agent_yaw_rad,
    )


    history_vels_mps,  future_vels_mps = compute_agent_velocity(history_positions_m, future_positions_m, step_time)     #history_acc_mpss,
    #print('history frames', history_frames)

    # definition of position: a relative location with respect to agent_from_world matrix. history frame [0] is the current
    #frame and in it the location of the agent is considered (0,0). hence the location in history frame [-1] and [-2] are negative.
    # cnosequently, the location in future frame is positive. we are talking about only x values because that is the direction
    #of movement

    result = {
        "frame_index": state_index,
        "image": input_im,
        "center_in_world":center_in_world,          #added
        "target_positions": future_positions_m,
        "target_yaws": future_yaws_rad,
        "target_velocities": future_vels_mps,
        "target_availabilities": future_availabilities,
        "history_positions": history_positions_m,
        "history_yaws": history_yaws_rad,
        "history_velocities": history_vels_mps,
        #"history_acc_mpss": history_acc_mpss,        #added
        "history_tl_faces" : history_tl_faces,      # added
        "history_availabilities": history_availabilities,
        "world_to_image": raster_from_world,  # TODO deprecate
        "raster_from_agent": raster_from_world @ world_from_agent,
        "raster_from_world": raster_from_world,
        "agent_from_world": agent_from_world,
        "world_from_agent": world_from_agent,
        "centroid": agent_centroid_m,
        "yaw": agent_yaw_rad,
        "extent": agent_extent_m,
        "history_extents": history_extents,
        "future_extents": future_extents,
        "sem_rasterize":sem_rasterize       #added
        #"safe_dist_agents":agents_safe_dist  #added
    }
    if len(history_vels_mps) > 0:
        # estimated current speed based on displacement between current frame at T and past frame at T-1
        result["curr_speed"] = np.linalg.norm(history_vels_mps[0])
    return result

'''
######### this is the original generate_agent_sample function
######### the above one is slightly modified

def generate_agent_sample(
        state_index: int,
        frames: np.ndarray,
        agents: np.ndarray,
        tl_faces: np.ndarray,
        selected_track_id: Optional[int],
        render_context: RenderContext,
        history_num_frames: int,
        future_num_frames: int,
        step_time: float,
        filter_agents_threshold: float,
        rasterizer: Rasterizer,
        perturbation: Optional[Perturbation] = None,
) -> dict:
    """Generates the inputs and targets to train a deep prediction model. A deep prediction model takes as input
    the state of the world (here: an image we will call the "raster"), and outputs where that agent will be some
    seconds into the future.

    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.

    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        tl_faces (np.ndarray): The full traffic light faces array, can be numpy array or a zarr array
        selected_track_id (Optional[int]): Either None for AV, or the ID of an agent that you want to
        predict the future of. This agent is centered in the raster and the returned targets are derived from
        their future states.
        render_context (RenderContext): The context for rasterisation
        history_num_frames (int): Amount of history frames to draw into the rasters
        future_num_frames (int): Amount of history frames to draw into the rasters
        step_time (float): seconds between consecutive steps
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent
        rasterizer Rasterizer: Rasterizer of some sort that draws a map image
        perturbation (Optional[Perturbation]): Object that perturbs the input and targets, used
        to train models that can recover from slight divergence from training set data

    Raises:
        IndexError: An IndexError is returned if the specified ``selected_track_id`` is not present in the scene
        or was filtered by applying the ``filter_agent_threshold`` probability filtering.

    Returns:
        dict: a dict object with the raster array, the future offset coordinates (meters),
        the future yaw angular offset, the future_availability as a binary mask
    """
    (
        history_frames,
        future_frames,
        history_agents,
        future_agents,
        history_tl_faces,
        future_tl_faces,
    ) = get_agent_context(state_index, frames, agents, tl_faces, history_num_frames, future_num_frames, )

    if perturbation is not None and len(future_frames) == future_num_frames:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    if selected_track_id is None:
        agent_centroid_m = cur_frame["ego_translation"][:2]
        agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent_m = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
        selected_agent = None
    else:
        # this will raise IndexError if the agent is not in the frame or under agent-threshold
        # this is a strict error, we cannot recover from this situation
        filtered_agents = filter_agents_by_labels(cur_agents, filter_agents_threshold)
        agent = filter_agents_by_track_id(filtered_agents, selected_track_id)[0]
        agent_centroid_m = agent["centroid"]
        agent_yaw_rad = float(agent["yaw"])
        agent_extent_m = agent["extent"]
        selected_agent = agent

    input_im = rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)

    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    agent_from_world = np.linalg.inv(world_from_agent)
    raster_from_world = render_context.raster_from_world(agent_centroid_m, agent_yaw_rad)

    future_positions_m, future_yaws_rad, future_extents, future_availabilities = get_relative_poses(
        future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, agent_yaw_rad,
    )
    # history_num_frames + 1 because it also includes the current frame
    history_positions_m, history_yaws_rad, history_extents, history_availabilities = get_relative_poses(
        history_num_frames + 1, history_frames, selected_track_id, history_agents, agent_from_world, agent_yaw_rad,
    )

    history_vels_mps, future_vels_mps = compute_agent_velocity(history_positions_m, future_positions_m, step_time)

    result = {
        "frame_index": state_index,
        "image": input_im,
        "target_positions": future_positions_m,
        "target_yaws": future_yaws_rad,
        "target_velocities": future_vels_mps,
        "target_availabilities": future_availabilities,
        "history_positions": history_positions_m,
        "history_yaws": history_yaws_rad,
        "history_velocities": history_vels_mps,
        "history_availabilities": history_availabilities,
        "world_to_image": raster_from_world,  # TODO deprecate
        "raster_from_agent": raster_from_world @ world_from_agent,
        "raster_from_world": raster_from_world,
        "agent_from_world": agent_from_world,
        "world_from_agent": world_from_agent,
        "centroid": agent_centroid_m,
        "yaw": agent_yaw_rad,
        "extent": agent_extent_m,
        "history_extents": history_extents,
        "future_extents": future_extents,
    }
    if len(history_vels_mps) > 0:
        # estimated current speed based on displacement between current frame at T and past frame at T-1
        result["curr_speed"] = np.linalg.norm(history_vels_mps[0])
    return result
'''
