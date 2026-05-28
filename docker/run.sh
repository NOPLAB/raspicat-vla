#!/usr/bin/env bash
# docker/run.sh — wrapper for building and running raspicat-vla Docker images.
#
# Subcommands:
#   build TARGET            asyncvla | omnivla | test | real | sim | --all
#   run MODEL MODE [OPTS]   MODEL = asyncvla | omnivla
#                           MODE  = --remote {--cpu|--gpu}
#                                   --real --host HOST
#                                   --sim --host HOST
#
# Run `run.sh --help` for the full reference.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRPC_PORT="${GRPC_PORT:-50051}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"

# Image / Dockerfile / model knob registries. Bash 4 associative arrays.
declare -A IMAGES=(
    [asyncvla]="raspicat-vla-asyncvla"
    [omnivla]="raspicat-vla-omnivla"
    [test]="raspicat-vla-test"
    [real]="raspicat-vla-real"
    [sim]="raspicat-vla-sim"
)
declare -A DOCKERFILES=(
    [asyncvla]="docker/Dockerfile.asyncvla"
    [omnivla]="docker/Dockerfile.omnivla"
    [test]="docker/Dockerfile.test"
    [real]="docker/Dockerfile.real"
    [sim]="docker/Dockerfile.sim"
)
declare -A RESUME_STEP=(
    [asyncvla]=750000
    [omnivla]=120000
)
declare -A WEIGHTS_DIR=(
    [asyncvla]="/workspace/models/AsyncVLA_release"
    [omnivla]="/workspace/models/omnivla-original"
)

usage() {
    cat <<'EOF'
Usage: run.sh COMMAND [ARGS]

Commands:
  build TARGET            Build a Docker image
    TARGET = asyncvla | omnivla | test | real | sim | --all
  run MODEL MODE [OPTS]   Run a configuration
    MODEL = asyncvla | omnivla
    MODE:
      --remote {--cpu|--gpu} [--host BIND[:PORT]]
                                    Host the cloud-side gRPC server here.
                                    Uses Dockerfile.<MODEL>. BIND defaults to
                                    0.0.0.0 (all interfaces). Optional :PORT
                                    overrides $GRPC_PORT.
      --real --host HOST[:PORT]     Edge stack here, talking to a cloud server
                                    at HOST:PORT (PORT defaults to $GRPC_PORT).
                                    Uses Dockerfile.real.
      --sim  --host HOST[:PORT]     Edge + Gazebo simulation, cloud at
                                    HOST:PORT. Uses Dockerfile.sim. Plan 3 wip.
  test [PYTEST_ARGS...]   Run pytest in raspicat-vla-test (CPU). Auto-builds
                          the image if missing. Pass extra args to pytest:
                            run.sh test                        # full suite
                            run.sh test -k checkpoint          # filter
                            run.sh test src/raspicat_vla_edge/test  # subset
  help, -h, --help        Show this help

Examples:
  run.sh build asyncvla
  run.sh build --all
  run.sh run asyncvla --remote --gpu                       # bind 0.0.0.0:50051
  run.sh run asyncvla --remote --gpu --host :8080          # bind 0.0.0.0:8080
  run.sh run asyncvla --remote --cpu --host 127.0.0.1      # localhost only
  run.sh run omnivla  --remote --gpu --host 10.0.0.5:9000  # specific NIC + port
  run.sh run asyncvla --real --host 192.168.1.2            # default port
  run.sh run asyncvla --real --host 192.168.1.2:8080
  run.sh run omnivla  --sim  --host 192.168.1.2:9000
  run.sh test                                              # full pytest suite
  run.sh test -k omnivla                                   # filter by name

Environment overrides:
  GRPC_PORT     gRPC port (default 50051)
  HF_CACHE_DIR  HuggingFace cache mount (default $HOME/.cache/huggingface)
EOF
}

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
err()  { printf '\033[1;31m!!\033[0m  %s\n' "$*" >&2; }

# split_hostport HOST[:PORT] DEFAULT_HOST DEFAULT_PORT -> "HOST PORT" on stdout.
# Empty HOST (e.g. ":8080") falls back to DEFAULT_HOST. Missing :PORT -> DEFAULT_PORT.
split_hostport() {
    local raw=$1 default_host=$2 default_port=$3
    local host port
    if [[ $raw == *:* ]]; then
        host=${raw%:*}
        port=${raw##*:}
    else
        host=$raw
        port=$default_port
    fi
    [[ -z $host ]] && host=$default_host
    if ! [[ $port =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
        err "invalid port in '$raw'"
        return 1
    fi
    printf '%s %s\n' "$host" "$port"
}

build_one() {
    local target=$1
    local dfile_rel="${DOCKERFILES[$target]:-}"
    local image="${IMAGES[$target]:-}"
    [[ -n $dfile_rel && -n $image ]] || { err "unknown build target: $target"; return 1; }
    local dfile="${REPO_ROOT}/${dfile_rel}"
    [[ -f $dfile ]] || { err "Dockerfile not found: $dfile"; return 1; }
    log "building ${image} from ${dfile_rel}"
    docker build -f "$dfile" -t "$image" "$REPO_ROOT"
}

cmd_build() {
    local target=${1:-}
    case $target in
        --all)
            local rc=0
            for t in asyncvla omnivla test real sim; do
                build_one "$t" || rc=1
            done
            return $rc
            ;;
        asyncvla|omnivla|test|real|sim)
            build_one "$target"
            ;;
        '')
            err "build: missing target"
            usage
            return 1
            ;;
        *)
            err "build: unknown target '$target'"
            usage
            return 1
            ;;
    esac
}

run_remote() {
    local model=$1 device=$2 bind_host=$3 bind_port=$4
    local image="${IMAGES[$model]}"
    local resume_step="${RESUME_STEP[$model]}"
    local weights="${WEIGHTS_DIR[$model]}"
    local gpu_flag="" device_arg="cpu"
    if [[ $device == gpu ]]; then
        gpu_flag="--gpus all"
        device_arg="cuda:0"
    fi

    log "${model} remote backend on ${device_arg}, bind ${bind_host}:${bind_port}"
    # The raspicat_vla_proto/raspicat_vla_remote packages are ROS2 ament_python
    # layouts (setup.cfg uses `script_dir`), so `pip install -e` fails on modern
    # setuptools. Run from source via PYTHONPATH instead.
    # shellcheck disable=SC2086
    docker run --rm $gpu_flag --network host \
        -v "$REPO_ROOT:/workspace" \
        -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
        "$image" bash -lc "
            cd /workspace
            export PYTHONPATH=/workspace/src/raspicat_vla_proto:/workspace/src/raspicat_vla_remote\${PYTHONPATH:+:\$PYTHONPATH}
            exec python3 -m raspicat_vla_remote.server_main \
                --backend ${model} \
                --host ${bind_host} \
                --port ${bind_port} \
                --vla-path ${weights} \
                --resume-step ${resume_step} \
                --device ${device_arg}
        "
}

# Build raspicat-vla packages inside the container (idempotent — colcon
# detects already-built packages). Required because the rt-net packages are
# pre-built in /opt/sim_ws but the user-side raspicat_vla_* are mounted.
_workspace_build_cmd() {
    cat <<'BUILD'
if [ ! -f install/setup.bash ] || [ -n "$RASPICAT_VLA_REBUILD" ]; then
    echo "==> colcon build raspicat_vla_*" >&2
    colcon build --symlink-install \
        --packages-select raspicat_vla_msgs raspicat_vla_proto \
                          raspicat_vla_remote raspicat_vla_edge \
                          raspicat_vla_bringup
fi
source install/setup.bash
BUILD
}

run_real() {
    local model=$1 host=$2 port=$3
    local image="${IMAGES[real]}"
    local has_real_image=true
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        has_real_image=false
        warn "image ${image} not built; falling back to ${IMAGES[test]} (no rt-net packages)."
        warn "run \`run.sh build real\` for the full image with raspicat_ros + Edge_adapter deps."
        image="${IMAGES[test]}"
        if [[ $model == asyncvla ]]; then
            warn "AsyncVLA edge needs torch + MBRA on PYTHONPATH; the test image lacks them."
        fi
    fi

    log "${model} edge (real, image=${image}); cloud=${host}:${port}"
    local source_real_ws=""
    if $has_real_image; then
        source_real_ws="source /opt/real_ws/install/setup.bash"
    fi
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
        --network host \
        -v "$REPO_ROOT:/workspace" \
        -v "$HF_CACHE_DIR:/tmp/.cache/huggingface" \
        "$image" bash -lc "
            source /opt/ros/humble/setup.bash
            ${source_real_ws}
            cd /workspace
            $(_workspace_build_cmd)
            exec ros2 launch raspicat_vla_edge edge_only.launch.py \
                remote_address:=${host}:${port} \
                adapter_kind:=${model} \
                with_follower:=true
        "
}

run_sim() {
    local model=$1 host=$2 port=$3
    local image="${IMAGES[sim]}"
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        warn "image ${image} not built; falling back to ${IMAGES[test]} (no Gazebo)."
        warn "run \`run.sh build sim\` for the full sim image with Gazebo + raspicat_sim."
        image="${IMAGES[test]}"
        # Fallback: edge_only without Gazebo.
        log "${model} edge (sim-fallback, image=${image}); cloud=${host}:${port}"
        docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
            --network host \
            -v "$REPO_ROOT:/workspace" \
            -v "$HF_CACHE_DIR:/tmp/.cache/huggingface" \
            "$image" bash -lc "
                source /opt/ros/humble/setup.bash
                cd /workspace
                $(_workspace_build_cmd)
                exec ros2 launch raspicat_vla_edge edge_only.launch.py \
                    remote_address:=${host}:${port} \
                    adapter_kind:=${model} \
                    with_follower:=true
            "
        return
    fi

    # Full sim image with Gazebo. Forward DISPLAY so gzclient renders on the host.
    # Synthesize a /etc/passwd entry for the host UID inside the container so
    # gzclient stops spamming "Error getting username: no matching password record".
    # We can't bind-mount the host's /etc/passwd because Gazebo would then try
    # HOME=/home/<user> which doesn't exist in the image; the synthesized entry
    # points HOME at /tmp instead.
    local display_args=()
    if [[ -n ${DISPLAY:-} ]]; then
        display_args+=(-e "DISPLAY=$DISPLAY" -v "/tmp/.X11-unix:/tmp/.X11-unix:ro")
    fi
    local uid gid passwd_dir
    uid=$(id -u); gid=$(id -g)
    passwd_dir=$(mktemp -d)
    cat /etc/passwd > "$passwd_dir/passwd"
    grep -q "^[^:]*:[^:]*:${uid}:" "$passwd_dir/passwd" || \
        echo "raspicat:x:${uid}:${gid}:raspicat:/tmp:/bin/bash" >> "$passwd_dir/passwd"
    cat /etc/group > "$passwd_dir/group"
    grep -q "^[^:]*:[^:]*:${gid}:" "$passwd_dir/group" || \
        echo "raspicat:x:${gid}:" >> "$passwd_dir/group"

    log "${model} sim (image=${image}); cloud=${host}:${port}"
    docker run --rm --user "${uid}:${gid}" -e HOME=/tmp \
        --network host \
        "${display_args[@]}" \
        -v "$passwd_dir/passwd:/etc/passwd:ro" \
        -v "$passwd_dir/group:/etc/group:ro" \
        -v "$REPO_ROOT:/workspace" \
        -v "$HF_CACHE_DIR:/tmp/.cache/huggingface" \
        "$image" bash -lc "
            source /opt/ros/humble/setup.bash
            source /opt/sim_ws/install/setup.bash
            cd /workspace
            $(_workspace_build_cmd)
            exec ros2 launch raspicat_vla_bringup mvp_sim.launch.py \
                remote_address:=${host}:${port} \
                adapter_kind:=${model}
        "
}

cmd_run() {
    local model=${1:-}
    case $model in
        asyncvla|omnivla) ;;
        '')
            err "run: missing model (asyncvla|omnivla)"; usage; return 1 ;;
        *)
            err "run: unknown model '$model'"; usage; return 1 ;;
    esac
    shift

    local mode='' host='' device=''
    while [[ $# -gt 0 ]]; do
        case $1 in
            --remote) mode=remote; shift ;;
            --real)   mode=real; shift ;;
            --sim)    mode=sim; shift ;;
            --host)
                [[ $# -ge 2 ]] || { err "--host requires an argument"; return 1; }
                host=$2; shift 2 ;;
            --cpu)    device=cpu; shift ;;
            --gpu)    device=gpu; shift ;;
            -h|--help) usage; return 0 ;;
            *) err "run: unknown option '$1'"; usage; return 1 ;;
        esac
    done

    case $mode in
        remote)
            if [[ -z $device ]]; then
                err "--remote requires --cpu or --gpu"; return 1
            fi
            local pair bind_host bind_port
            pair=$(split_hostport "${host:-0.0.0.0}" "0.0.0.0" "$GRPC_PORT") || return 1
            read -r bind_host bind_port <<<"$pair"
            run_remote "$model" "$device" "$bind_host" "$bind_port"
            ;;
        real|sim)
            [[ -n $host ]] || { err "--$mode requires --host HOST[:PORT]"; return 1; }
            local pair edge_host edge_port
            pair=$(split_hostport "$host" "" "$GRPC_PORT") || return 1
            read -r edge_host edge_port <<<"$pair"
            [[ -n $edge_host ]] || { err "--$mode --host needs a host part"; return 1; }
            "run_$mode" "$model" "$edge_host" "$edge_port"
            ;;
        '')
            err "run: missing mode (--remote|--real|--sim)"; usage; return 1 ;;
    esac
}

cmd_test() {
    local image="${IMAGES[test]}"
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        warn "image ${image} not built; building first..."
        build_one test || return 1
    fi

    # Build the default test set: explicit files (not directories) because
    # ROS2's launch_testing pytest plugin claims directories and silently
    # drops every test in them ("collected 0 items / 1 skipped"). Smoke tests
    # guarded by ASYNCVLA_E2E / OMNIVLA_E2E env vars are included but skip
    # cleanly without GPU.
    local default_paths
    mapfile -t default_paths < <(
        find "$REPO_ROOT/src" -path '*/test/test_*.py' -not -path '*/__pycache__/*' \
            | sort \
            | sed "s|^$REPO_ROOT/||"
    )

    local args
    if [[ $# -eq 0 ]]; then
        args=(-v "${default_paths[@]}")
    else
        # Decide whether the user supplied any test path. Pure-flag invocations
        # (`run.sh test -k name`, `--lf`, `-x`) need default_paths prepended
        # so pytest doesn't fall back to cwd discovery (which would walk
        # external/ and crash on missing transitive deps).
        local has_path=false a
        for a in "$@"; do
            [[ -e $a || -e $REPO_ROOT/$a ]] && { has_path=true; break; }
        done
        if $has_path; then
            args=("$@")
        else
            args=("${default_paths[@]}" "$@")
        fi
    fi
    local args_str
    printf -v args_str '%q ' "${args[@]}"

    log "pytest in ${image} (${#args[@]} args)"
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "$REPO_ROOT:/workspace" \
        "$image" bash -lc "
            set -e
            source /opt/ros/humble/setup.bash
            cd /workspace
            $(_workspace_build_cmd)
            exec python3 -m pytest ${args_str}
        "
}

case ${1:-} in
    -h|--help|help|'') usage ;;
    build) shift; cmd_build "$@" ;;
    run)   shift; cmd_run "$@" ;;
    test)  shift; cmd_test "$@" ;;
    *) err "unknown command: '$1'"; usage; exit 1 ;;
esac
