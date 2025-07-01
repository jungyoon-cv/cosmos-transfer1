import os
import math
import queue
import logging

import argparse
import time

import numpy as np
from numpy import random
from PIL import Image, ImageDraw

import pygame
from pygame.locals import QUIT


import carla
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)


VEHICLE_TAILLIGHT_MARKERS = {
    "vehicle.dodge.charger_police_2020": {
        "left":  { "x": -230.0, "y": -65.0, "z": 93.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
        "right": { "x": -230.0, "y":  65.0, "z": 93.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }
    },
    "vehicle.dodge.charger_2020": {
        "left":  { "x": -230.0, "y": -65.0, "z": 93.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
        "right": { "x": -230.0, "y":  65.0, "z": 93.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }
    },
    "vehicle.lincoln.mkz_2020": {
        "left":  { "x": -220.0, "y": -60.0, "z": 92.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
        "right": { "x": -220.0, "y":  60.0, "z": 92.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }
    },
    "vehicle.nissan.patrol_2021": {
        "left":  { "x": -250.0, "y": -80.0, "z": 118.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
        "right": { "x": -250.0, "y":  80.0, "z": 118.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }
    },
    "vehicle.tesla.model3": {
        "left":  { "x": -220.0, "y": -65.0, "z": 95.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
        "right": { "x": -220.0, "y":  65.0, "z": 95.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }
    },
    "vehicle.audi.tt": {
        "left":  { "x": -195.0, "y": -75.0, "z": 80.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
        "right": { "x": -195.0, "y":  75.0, "z": 80.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }
    },
    "vehicle.ford.crown": {
        "left":  { "x": -230.0, "y": -70.0, "z": 75.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
        "right": { "x": -230.0, "y":  70.0, "z": 75.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }
    },
    "vehicle.mercedes.sprinter": {
        "left":  { "x": -288.0, "y": -90.0, "z": 110.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
        "right": { "x": -288.0, "y":  90.0, "z": 110.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }
    },
    "vehicle.volkswagen.t2_2021": {
        "left":  { "x": -210.0, "y": -60.0, "z": 85.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
        "right": { "x": -210.0, "y":  60.0, "z": 85.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }
    },
    "vehicle.carlamotors.firetruck": {
        "left":  { "x": -438.0, "y": -125.0, "z": 63.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
        "right": { "x": -438.0, "y":  125.0, "z": 63.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }
    },
    "vehicle.ford.ambulance": {
        "left":  { "x": -320.0, "y": -100.0, "z": 57.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
        "right": { "x": -320.0, "y":  100.0, "z": 57.0,  "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }
    }
}


def parse_arguments():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int)
    argparser.add_argument('-m', '--map-name', metavar='M', default=None)
    argparser.add_argument('-w', '--weather', metavar='WEATHER', default=None)
    argparser.add_argument('-v', '--vehicle', metavar='VEH_ID',
                           default='vehicle.lincoln.mkz_2020',
                           help='Blueprint ID of the ego vehicle '
                                '(e.g. vehicle.tesla.model3).')
    argparser.add_argument('-o', '--output-dir', metavar='DIR', default=False)
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x704')
    argparser.add_argument('--no-overlay', dest='no_overlay', action='store_true',
                           help='Disable projection / overlay processing')
    argparser.add_argument('--headless', action='store_true',
                           help='Run without opening a PyGame window')
    # Orbit camera options -------------------------------------------------------------------------------
    argparser.add_argument('-n', '--max-frames', type=int, default=0)
    argparser.add_argument('--fps', type=int, default=24)
    argparser.add_argument('--blink-mode', choices=['left', 'right', 'hazard'], default='left', 
                           help="Blinker pattern: left, right, or hazard (both).")
    argparser.add_argument('--blink-hz', dest='blink_hz', type=float, default=1.0,
                           help='Blinker flashing frequency in hertz (default 1.0)')
    # Orbit camera options -------------------------------------------------------------------------------
    argparser.add_argument('--radius', dest='orbit_radius', type=float, default=5.5,
                           help='Orbit radius around the vehicle [m]')
    argparser.add_argument('--orbit-speed', type=float, default=30.0,
                           help='Orbit angular speed in deg/s')
    argparser.add_argument('--dot-pattern', choices=['equator', 'meridian', 'helical'],
                           default='helical', help='Orbit path pattern')
    argparser.add_argument('--scale-z', dest='orbit_scale_z', type=float, default=0.1,
                           help='Vertical scale factor for orbit (1.0 = sphere)')
    argparser.add_argument('--offset', dest='orbit_offset', nargs=3, type=float,
                           default=[-12, 0, 1.8], metavar=('DX', 'DY', 'DZ'),
                           help='Orbit centre offset in vehicle frame [m]')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    if args.blink_hz <= 0:
        raise ValueError("--blink-hz must be positive")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")
    if len(args.orbit_offset) != 3:
        raise ValueError("--offset must provide three floats: DX DY DZ")
    if not hasattr(args, 'no_overlay'):
        args.no_overlay = False
    return args


class TaillightApp:
    def __init__(self, args):
        self.args = args

        # --- Connect to simulator and load world ---
        self.client = carla.Client(args.host, args.port)
        self.world = (self.client.load_world(args.map_name) if args.map_name else self.client.get_world())

        # Optional weather preset
        if args.weather:
            weather_param = getattr(carla.WeatherParameters, args.weather)
            self.world.set_weather(weather_param)

        # --- World settings (fixed time‑step, sync mode) ---
        settings = self.world.get_settings()
        settings.synchronous_mode   = True
        base_dt = 1.0 / args.fps
        settings.fixed_delta_seconds = base_dt    # keep simulation Δt constant; run more ticks per real‑second instead
        self.world.apply_settings(settings)

        # --- Blueprint library and actor spawning ---
        self.blueprint_library = self.world.get_blueprint_library()
        spawn_bp = self.blueprint_library.find(args.vehicle)
        logging.info("Ego vehicle blueprint: %s", spawn_bp.id)

        self.tmap         = self.world.get_map()
        spawn_points      = self.tmap.get_spawn_points()
        spawn_transform   = random.choice(spawn_points)
        try:
            spawn_points.remove(spawn_transform)
        except ValueError:
            pass

        # --- Traffic Manager configuration ---
        self.tm = self.client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)

        # --- Spawn ego vehicle only ---
        self.vehicle = self.world.spawn_actor(spawn_bp, spawn_transform)

        tm_port = self.tm.get_port()
        self.vehicle.set_autopilot(True, tm_port)

        # Configure Traffic Manager for the ego car
        self.tm.random_left_lanechange_percentage(self.vehicle, 90)
        self.tm.random_right_lanechange_percentage(self.vehicle, 90)
        self.tm.auto_lane_change(self.vehicle, False)
        self.tm.ignore_lights_percentage(self.vehicle, 0)
        self.tm.ignore_signs_percentage(self.vehicle, 0)

        self.world.tick()
        self.tm.update_vehicle_lights(self.vehicle, False)

        # runtime flags for overlay colouring
        self._left_on  = False
        self._right_on = False

        # sun_alt = getattr(self.world.get_weather(), 'sun_altitude_angle', 45.0)
        # if sun_alt < 0.0:
        #     self._baseline_lights = (carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
        # else:
        #     self._baseline_lights = carla.VehicleLightState.Position
        
        self._baseline_lights = carla.VehicleLightState.NONE
        self.vehicle.set_light_state(carla.VehicleLightState(self._baseline_lights))
        # Initialise blinker phase accumulator
        self._blink_phase_acc = 0.0

        # --- Camera and spectator parameters ---
        bbox = self.vehicle.bounding_box.extent
        self.bound_x = 0.5 + bbox.x
        self.bound_z = 0.5 + bbox.z
        self.spec_offset_x      = -3.0 * self.bound_x
        self.spec_offset_z      = 1.8 * self.bound_z
        self.spec_offset_pitch  = 0.0
        self.spectator          = self.world.get_spectator()

        # --- PyGame window & sensor ---
        if not args.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((args.width, args.height))
            pygame.display.set_caption('Carla - Taillight Dataset')
        else:
            self.screen = None

        cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(args.width))
        cam_bp.set_attribute('image_size_y', str(args.height))
        cam_bp.set_attribute('fov', '90')
        cam_bp.set_attribute('sensor_tick', f'{(1.0/args.fps):.6f}')
        self.cam_sensor = self.world.spawn_actor(
            cam_bp,
            carla.Transform(
                carla.Location(x=self.spec_offset_x, y=0.0, z=self.spec_offset_z),
                carla.Rotation(pitch=self.spec_offset_pitch)
            ),
            attach_to=self.vehicle
        )

        self.camera_surface = [None]
        self.camera_rgb = [None]
        # (frame_id, timestamp) from the most‑recent camera callback
        self.last_frame_meta = (None, None)
        self.frame_idx  = 0

        # --- Saving worker (RGB + overlay) ---
        self._save_q   = None
        self._save_proc = None
        if args.output_dir:
            self._save_q = __import__('multiprocessing').Queue(maxsize=8)
            self._save_proc = __import__('multiprocessing').Process(
                target=TaillightApp._save_worker,
                args=(self._save_q, args.width, args.height, args.output_dir),
                daemon=True
            )
            self._save_proc.start()

        def _camera_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3][:, :, ::-1]
            self.camera_rgb[0] = array
            # Record simulator frame number and timestamp
            self.last_frame_meta = (image.frame, image.timestamp)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self.camera_surface[0] = surface

        self.cam_sensor.listen(_camera_callback)

        # -------- Orbit camera parameters --------------------------------------
        self.orbit_radius  = args.orbit_radius
        self.orbit_speed   = math.radians(args.orbit_speed) # rad s⁻¹
        self.orbit_scale_z = args.orbit_scale_z
        self.dot_pattern   = args.dot_pattern
        self.orbit_center_offset = carla.Location(
            x=args.orbit_offset[0],
            y=args.orbit_offset[1],
            z=args.orbit_offset[2]
        )
        self._orbit_angle = 0.0

        self._last_overlay = ([], [])

        self.clock = pygame.time.Clock()

        if not args.no_overlay:
            self._overlay_in_q  = __import__('multiprocessing').Queue(maxsize=2)
            self._overlay_out_q = __import__('multiprocessing').Queue(maxsize=2)
            self._overlay_proc  = __import__('multiprocessing').Process(
                target=TaillightApp._overlay_worker,
                args=(self._overlay_in_q, self._overlay_out_q),
                daemon=True
            )
            self._overlay_proc.start()
        else:
            self._overlay_in_q  = None
            self._overlay_out_q = None
            self._overlay_proc  = None

    def _draw_overlays(self):
        if self.args.no_overlay:
            return
        if self.screen is None:
            # ---------------- Headless mode ----------------
            # Pull the most‑recent overlay result from worker so that
            # lines/rects are not permanently empty (otherwise control /
            # overlay PNGs become all black).
            while self._overlay_out_q is not None and not self._overlay_out_q.empty():
                try:
                    lines, rects = self._overlay_out_q.get_nowait()
                    self._last_overlay = (lines, rects)
                except queue.Empty:
                    break
                except Exception:
                    break  # safeguard
            # Use cached overlay if nothing new arrived
            lines, rects = self._last_overlay
            # Still save the frame (RGB, control, overlay)
            self._save_frame(lines, rects)
            return
        target_box = self.screen

        # Try to fetch the most recent overlay result
        updated = False
        while not self._overlay_out_q.empty():
            try:
                lines, rects = self._overlay_out_q.get_nowait()
                self._last_overlay = (lines, rects)
                updated = True
            except queue.Empty:
                break
            except Exception:
                break  # safeguard against malformed data

        # Use cached overlay if nothing new arrived this frame
        lines, rects = self._last_overlay

        # Red bounding‑box edges
        for x0, y0, x1, y1, w in lines:
            pygame.draw.line(target_box, (255, 0, 0),
                             (x0, y0), (x1, y1), w)
        # Colour-coded taillight markers
        for (x, y, size_px, idx) in rects:
            half = size_px // 2
            is_on = (idx == 0 and self._left_on) or (idx == 1 and self._right_on)
            if not is_on:
                continue            # do NOT draw a marker when the light is off

            # vivid colour (no dim state)
            colour = (0, 255, 0) if idx == 0 else (0, 0, 255)
            pygame.draw.rect(target_box, colour,
                             pygame.Rect(x - half, y - half,
                                         size_px, size_px), 0)

        # Queue frame for saving
        self._save_frame(lines, rects)

    def _save_frame(self, lines, rects):
        if self._save_q is None or self.camera_rgb[0] is None or self.last_frame_meta[0] is None:
            return
        frame_no, ts = self.last_frame_meta
        # Block if queue is full; apply back‑pressure instead of dropping frames
        self._save_q.put(
            (frame_no,
             ts,
             self.camera_rgb[0].copy(),
             lines,
             rects,
             self._left_on,
             self._right_on)
        )
        self.frame_idx += 1

    @staticmethod
    def _save_worker(in_q, width, height, out_root):
        os.makedirs(os.path.join(out_root, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(out_root, 'overlay'), exist_ok=True)
        os.makedirs(os.path.join(out_root, 'tail'), exist_ok=True)
        # Accumulate per‑frame blinker states so we can write them once at the end
        summary = []

        while True:
            pkg = in_q.get()
            if pkg is None:
                # Write a CSV‑style text file with per‑frame blinker states
                txt_path = os.path.join(out_root, 'blink_status.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write('frame,timestamp_us,left,right\n')
                    for f_idx, ts_us, l_on, r_on in summary:
                        f.write(f'{f_idx},{ts_us},{int(l_on)},{int(r_on)}\n')
                break  # graceful shutdown

            frame_idx, ts, rgb_arr, lines, rects, left_on, right_on = pkg
            ts_us = int(ts * 1e6)
            summary.append((frame_idx, ts_us, left_on, right_on))
            rgb_fname = f'{frame_idx:06d}_{ts_us:016d}.jpg'
            mask_fname = f'{frame_idx:06d}_{ts_us:016d}.png'

            # ---------- Save raw RGB as JPEG ----------
            rgb_img = Image.fromarray(rgb_arr)
            rgb_img.save(os.path.join(out_root, 'rgb', rgb_fname), format='JPEG', quality=90, subsampling=0)

            # ---------- Draw mask on a black canvas ----------
            overlay_img = Image.new('RGB', (width, height), (0, 0, 0))
            draw = ImageDraw.Draw(overlay_img)

            for x0, y0, x1, y1, w in lines:
                draw.line([(x0, y0), (x1, y1)], width=w, fill=(255, 0, 0))

            for x, y, size_px, idx in rects:
                half = size_px // 2
                is_on = left_on if idx == 0 else right_on
                if not is_on:
                    continue            # skip rectangles when light is off

                # vivid colour (no dim state)
                colour = (0, 255, 0) if idx == 0 else (0, 0, 255)
                draw.rectangle(
                    [x - half, y - half, x + half, y + half],
                    fill=colour, outline=None
                )

            overlay_img.save(os.path.join(out_root, 'overlay', mask_fname))

            # ---------- Save tail‑light only mask ----------
            tail_img = Image.new('RGB', (width, height), (0, 0, 0))
            draw_tail = ImageDraw.Draw(tail_img)

            for x, y, size_px, idx in rects:
                half = size_px // 2
                is_on = left_on if idx == 0 else right_on
                if not is_on:
                    continue  # skip rectangles when light is off

                colour = (0, 255, 0) if idx == 0 else (0, 0, 255)
                draw_tail.rectangle(
                    [x - half, y - half, x + half, y + half],
                    fill=colour, outline=None
                )

            tail_img.save(os.path.join(out_root, 'tail', mask_fname))

    @staticmethod
    def _overlay_worker(in_q, out_q):
        EDGE_IDX = [
            (0, 1), (1, 3), (3, 2), (2, 0),   # rear face
            (0, 4), (1, 5), (2, 6), (3, 7),   # sides
            (4, 5), (5, 7), (7, 6), (6, 4)    # front face
        ]
        while True:
            pkg = in_q.get()
            if pkg is None:
                break                      # graceful shutdown
            (inv_mat, focal, cx, cy, w, h,
             bboxes, markers) = pkg

            inv_mat = np.asarray(inv_mat)
            lines   = []
            # --------- Bounding boxes ----------
            for corners in bboxes:
                proj_pts = []
                depths   = []
                for vx, vy, vz in corners:
                    cp = inv_mat.dot(np.array([vx, vy, vz, 1.0]))
                    u  = cp[1]
                    v_ = -cp[2]
                    z  = cp[0] if cp[0] > 0 else 0.001
                    x2d = int((focal * u / z) + cx)
                    y2d = int((focal * v_ / z) + cy)
                    proj_pts.append((x2d, y2d))
                    depths.append(z)
                if len(proj_pts) == 8 and not all(d <= 0 for d in depths):
                    for e0, e1 in EDGE_IDX:
                        lines.append(
                            (proj_pts[e0][0], proj_pts[e0][1],
                             proj_pts[e1][0], proj_pts[e1][1], 4)
                        )

            # --------- Taillight markers ----------
            rects = []
            for idx, (mx, my, mz) in enumerate(markers):
                cp = inv_mat.dot(np.array([mx, my, mz, 1.0]))
                z  = cp[0]
                if z <= 0:
                    continue
                x2d = int((focal * cp[1] / z) + cx)
                y2d = int((focal * -cp[2] / z) + cy)
                if 0 <= x2d < w and 0 <= y2d < h:
                    size_px = int(max(4, min(200.0 / z, 40)))
                    rects.append((x2d, y2d, size_px, len(rects)))   # idx 0:left, 1:right
            try:
                out_q.put((lines, rects), block=False)
            except queue.Full:
                pass

    @staticmethod
    def get_named_marker_world_locs(actor: carla.Actor):
        entry = VEHICLE_TAILLIGHT_MARKERS.get(actor.type_id)
        if entry is None:
            for k, v in VEHICLE_TAILLIGHT_MARKERS.items():
                if actor.type_id.startswith(k):
                    entry = v
                    break
        if entry is None:
            return []

        base_tf = actor.get_transform()
        world_locs = []

        for key in ("left", "right"):
            if key not in entry:
                continue
            rel = entry[key]
            loc_local = carla.Location(rel["x"] * 0.01,
                                       rel["y"] * 0.01,
                                       rel["z"] * 0.01)
            world_locs.append(base_tf.transform(loc_local))

        return world_locs

    def _send_overlay_request(self):
        if self.args.no_overlay:
            return
        if self._overlay_in_q.full():
            return  # worker still busy

        width, height = self.args.width, self.args.height
        fov           = 90.0
        focal         = width / (2.0 * math.tan(math.radians(fov) / 2.0))
        cx, cy        = width / 2.0, height / 2.0

        cam_tf = self.cam_sensor.get_transform()
        try:
            inv_mat = np.array(cam_tf.get_inverse_matrix())
        except AttributeError:
            inv_mat = np.linalg.inv(np.array(cam_tf.get_matrix()))

        # Gather 3‑D bounding box and tail‑light markers for the *ego* vehicle only
        bboxes = [
            [(v.x, v.y, v.z)
             for v in self.vehicle.bounding_box.get_world_vertices(
                 self.vehicle.get_transform())]
        ]

        markers = [
            (pt.x, pt.y, pt.z)
            for pt in self.get_named_marker_world_locs(self.vehicle)
        ]

        try:
            self._overlay_in_q.put_nowait(
                (inv_mat, focal, cx, cy, width, height, bboxes, markers)
            )
        except queue.Full:
            pass

    def _update_spectator(self):
        vehicle_tran = self.vehicle.get_transform()
        yaw          = vehicle_tran.rotation.yaw
        spectator_l  = vehicle_tran.location + carla.Location(
            self.spec_offset_x * math.cos(math.radians(yaw)),
            self.spec_offset_x * math.sin(math.radians(yaw)),
            self.spec_offset_z,
        )
        spectator_t = carla.Transform(
            spectator_l,
            carla.Rotation(pitch=self.spec_offset_pitch, yaw=yaw)
        )
        self.spectator.set_transform(spectator_t)

    def _update_camera_orbit(self):
        # Use the fixed simulation time step so the orbit is independent of real‑time tick rate
        sim_dt = self.world.get_settings().fixed_delta_seconds or (1.0 / self.args.fps)
        # Advance the angle accumulator
        self._orbit_angle += self.orbit_speed * sim_dt  # keep accumulating (no wrap)

        # --- Compute relative camera location according to the selected pattern ----
        if self.dot_pattern == "equator":
            rel_x = self.orbit_center_offset.x + self.orbit_radius * math.cos(self._orbit_angle)
            rel_y = self.orbit_center_offset.y + self.orbit_radius * math.sin(self._orbit_angle)
            rel_z = self.orbit_center_offset.z
        elif self.dot_pattern == "meridian":
            phi = self._orbit_angle
            rel_x = self.orbit_center_offset.x + self.orbit_radius * math.cos(phi)
            rel_y = self.orbit_center_offset.y
            rel_z = (self.orbit_center_offset.z +
                     self.orbit_radius * self.orbit_scale_z * math.sin(phi))
        elif self.dot_pattern == "helical":
            theta = self._orbit_angle
            phi = self._orbit_angle * 0.2  # continuous latitude shift
            rel_x = (self.orbit_center_offset.x +
                     self.orbit_radius * math.cos(theta) * math.cos(phi))
            rel_y = (self.orbit_center_offset.y +
                     self.orbit_radius * math.sin(theta) * math.cos(phi))
            rel_z = (self.orbit_center_offset.z +
                     self.orbit_radius * self.orbit_scale_z * math.sin(phi))
        else:  # fallback – stay at centre
            rel_x, rel_y, rel_z = (self.orbit_center_offset.x,
                                   self.orbit_center_offset.y,
                                   self.orbit_center_offset.z)

        # ---- Orientation: always look at vehicle origin -----------------------
        yaw_deg = (math.degrees(math.atan2(rel_y, rel_x)) + 180.0) % 360.0
        dist_h = math.hypot(rel_x, rel_y)
        pitch_deg = -math.degrees(math.atan2(rel_z, dist_h if dist_h > 1e-3 else 1.0))

        rel_tf = carla.Transform(
            carla.Location(x=rel_x, y=rel_y, z=rel_z),
            carla.Rotation(pitch=pitch_deg, yaw=yaw_deg)
        )
        self.cam_sensor.set_transform(rel_tf)

    def _update_blinker(self):
        sim_dt = self.world.get_settings().fixed_delta_seconds
        period = 1.0 / self.args.blink_hz
        self._blink_phase_acc = (self._blink_phase_acc + sim_dt) % period
        phase_on = self._blink_phase_acc < (period * 0.5)

        self._left_on = phase_on and (self.args.blink_mode in ('left', 'hazard'))
        self._right_on = phase_on and (self.args.blink_mode in ('right', 'hazard'))

        mask = self._baseline_lights
        if self._left_on:
            mask |= carla.VehicleLightState.LeftBlinker
        if self._right_on:
            mask |= carla.VehicleLightState.RightBlinker
        if self.vehicle.get_control().brake > 0.1:
            mask |= carla.VehicleLightState.Brake

        self.vehicle.set_light_state(carla.VehicleLightState(mask))

    def cleanup(self):
        # Stop overlay worker
        if self._overlay_proc is not None:
            self._overlay_in_q.put(None)
            self._overlay_proc.join(timeout=2)

        # Stop saving worker (if enabled)
        if self._save_q is not None:
            # Wait until the saving queue is fully processed before sending the sentinel.
            while not self._save_q.empty():
                time.sleep(0.1)      # give the worker time to drain
            self._save_q.put(None)    # graceful shutdown signal
            self._save_proc.join(timeout=15)

        self.cam_sensor.destroy()
        self.vehicle.destroy()

        settings = self.world.get_settings()
        settings.synchronous_mode   = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)


def main():
    args = parse_arguments()
    app = TaillightApp(args)
    # Progress bar for tick rate / remaining frames
    pbar = tqdm(total=(args.max_frames if args.max_frames else None),
                desc="Ticks",
                unit="tick",
                dynamic_ncols=True,
                smoothing=0.05)

    try:
        while True:
            target_loop_rate = max(1, int(args.fps))
            app.clock.tick_busy_loop(target_loop_rate)
            app._update_camera_orbit()
            app._update_blinker()

            if not args.headless:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        raise KeyboardInterrupt

            app._send_overlay_request()

            if not args.headless:
                app.screen.fill((0, 0, 0))
                if app.camera_surface[0] is not None:
                    full_img = pygame.transform.scale(
                        app.camera_surface[0],
                        (app.args.width, app.args.height)
                    )
                    app.screen.blit(full_img, (0, 0))
                    app._draw_overlays()
            else:
                if app.camera_surface[0] is not None:
                    app._draw_overlays()

            if args.max_frames and app.frame_idx >= args.max_frames:
                logging.info("Reached max‑frames limit (%d). Shutting down.", args.max_frames)
                raise KeyboardInterrupt

            if not args.headless:
                pygame.display.flip()

            app.world.tick()
            pbar.update(1)
            app._update_spectator()

    except KeyboardInterrupt:
        pass
    finally:
        pbar.close()
        app.cleanup()


if __name__ == '__main__':
    main()

