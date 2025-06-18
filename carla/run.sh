#!/bin/sh

CARLA_SERVER_BIN="/home/vip/CarlaUE4/Dist/CARLA_Shipping_0.9.15-314-g141fc20ae/LinuxNoEditor/CarlaUE4.sh"
PY_CLIENT="${SCRIPT_DIR}/ego_data.py"

MAP="Town04"
MAX_FRAMES=7200 # (300 s × 24 fps)
OUT_ROOT="tail_dataset"

# 6 essential weather presets
WEATHERS=(
  # "ClearNoon"
  "WetNoon"
  "HardRainNoon"
  "ClearSunset"
  "ClearNight"
  "HardRainNight"
)

# 11 blueprint IDs
VEHICLES=(
  "vehicle.dodge.charger_2020"
  "vehicle.dodge.charger_police_2020"
  "vehicle.lincoln.mkz_2020"
  "vehicle.nissan.patrol_2021"
  "vehicle.tesla.model3"
  "vehicle.audi.tt"
  "vehicle.ford.crown"
  "vehicle.mercedes.sprinter"
  "vehicle.volkswagen.t2_2021"
  "vehicle.carlamotors.firetruck"
  "vehicle.ford.ambulance"
)

# Blink modes to capture
BLINK_MODES=(
  "left"
  "right"
  "hazard"
)


mkdir -p "${OUT_ROOT}"

# Helper functions to launch and stop the CARLA server once per weather preset
start_server() {
  "${CARLA_SERVER_BIN}" -RenderOffScreen &          # Launch server in background
  SERVER_PID=$!
  # Give the server a few seconds to initialize
  sleep 15
}

stop_server() {
  echo "[INFO] Attempting graceful shutdown (SIGINT) ..."
  pkill -INT -f "CarlaUE4-Linux-Shipping"

  # Give the process up to 10 s to exit cleanly
  for i in $(seq 1 10); do
    if ! pgrep -f "CarlaUE4-Linux-Shipping" >/dev/null; then
      echo "[INFO] CARLA server stopped."
      return
    fi
    sleep 1
  done

  echo "[WARN] Graceful shutdown timed out. Forcing kill (SIGKILL)."
  pkill -9 -f "CarlaUE4-Linux-Shipping"
}


for WX in "${WEATHERS[@]}"; do
  echo "[INFO] Starting CARLA server for weather=${WX}"
  start_server
  for V in "${VEHICLES[@]}"; do
    for MODE in "${BLINK_MODES[@]}"; do
      SHORT_V=$(echo "${V}" | cut -d'.' -f3-)
      OUT_DIR="${OUT_ROOT}/${MAP}_${WX}_${SHORT_V//./_}_${MODE}"

      echo "[INFO] Starting: map=${MAP} weather=${WX} vehicle=${V} mode=${MODE}"
      python ego_data.py \
          -m "${MAP}" \
          -w "${WX}" \
          -v "${V}" \
          --blink-mode "${MODE}" \
          -o "${OUT_DIR}" \
          -n "${MAX_FRAMES}" \
          --blink-hz 1.0 \
          --time-scale 2.0 \
          --headless

      echo "[INFO] Finished: ${OUT_DIR}"
      echo
    done
  done
  echo "[INFO] Stopping CARLA server for weather=${WX}"
  stop_server
done

echo "[DONE] All combinations processed."