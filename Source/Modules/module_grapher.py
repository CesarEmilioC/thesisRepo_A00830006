import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation


# ===========================
#  MODULE 1: Load Data
# ===========================
def load_coordinates(file_path):
    """Load coordinates and timestamps from JSON file."""
    with open(file_path, "r") as file:
        data = json.load(file)

    pelvis = np.array(data["Pelvis"])
    wrist = np.array(data["Mano Derecha Original"])
    wrist_ref = np.array(data["Mano Derecha Referencia"])
    elbow = np.array(data["Codo Derecha Original"])
    elbow_ref = np.array(data["Codo Derecha Referencia"])
    timestamps = np.array(data["timestamps"])

    return pelvis, wrist, wrist_ref, elbow, elbow_ref, timestamps

# ===========================
#  MODULE 2: Plot Functions
# ===========================

def plot_original_coordinates(pelvis, wrist, elbow):
    """Plot original coordinates of pelvis+wrist and pelvis+elbow."""
    num_points = len(pelvis)

    colors_pelvis = cm.Blues(np.linspace(0.3, 1, num_points))
    colors_wrist = cm.Reds(np.linspace(0.3, 1, num_points))
    colors_elbow = cm.Purples(np.linspace(0.3, 1, num_points))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # ========== SUBPLOT A: Pelvis + Wrist ==========
    axs[0].set_title("Original Coordinates: Pelvis & Wrist")
    for i in range(num_points - 1):
        axs[0].plot([pelvis[i, 0], pelvis[i+1, 0]],
                    [pelvis[i, 1], pelvis[i+1, 1]], color=colors_pelvis[i])
        axs[0].plot([wrist[i, 0], wrist[i+1, 0]],
                    [wrist[i, 1], wrist[i+1, 1]], color=colors_wrist[i])

    axs[0].scatter(pelvis[:, 0], pelvis[:, 1], c=colors_pelvis, s=20, label="Pelvis")
    axs[0].scatter(wrist[:, 0], wrist[:, 1], c=colors_wrist, s=20, label="Wrist")
    axs[0].set_aspect("equal")
    axs[0].legend()
    axs[0].set_xlabel("Distance (px)")
    axs[0].set_ylabel("Distance (px)")

    # ========== SUBPLOT B: Pelvis + Elbow ==========
    axs[1].set_title("Original Coordinates: Pelvis & Elbow")
    for i in range(num_points - 1):
        axs[1].plot([pelvis[i, 0], pelvis[i+1, 0]],
                    [pelvis[i, 1], pelvis[i+1, 1]], color=colors_pelvis[i])
        axs[1].plot([elbow[i, 0], elbow[i+1, 0]],
                    [elbow[i, 1], elbow[i+1, 1]], color=colors_elbow[i])

    axs[1].scatter(pelvis[:, 0], pelvis[:, 1], c=colors_pelvis, s=20, label="Pelvis")
    axs[1].scatter(elbow[:, 0], elbow[:, 1], c=colors_elbow, s=20, label="Elbow")
    axs[1].set_aspect("equal")
    axs[1].legend()
    axs[1].set_xlabel("Distance (px)")
    axs[1].set_ylabel("Distance (px)")

    fig.text(
        0.5, 0.02,
        "Lighter colors represent the beginning of the movement, while darker colors indicate its end.",
        ha="center", fontsize=9, style="italic"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def plot_relative_coordinates(wrist_ref, elbow_ref):
    """Plot relative coordinates of wrist and elbow."""
    num_points = len(wrist_ref)

    colors_wrist = cm.Greens(np.linspace(0.3, 1, num_points))
    colors_elbow = cm.Oranges(np.linspace(0.3, 1, num_points))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # ========== SUBPLOT A: Wrist relative ==========
    axs[0].set_title("Relative Coordinates: Wrist - Pelvis")
    for i in range(num_points - 1):
        axs[0].plot([wrist_ref[i, 0], wrist_ref[i+1, 0]],
                    [wrist_ref[i, 1], wrist_ref[i+1, 1]], color=colors_wrist[i])
    axs[0].scatter(wrist_ref[:, 0], wrist_ref[:, 1], c=colors_wrist, s=20)
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("Distance (px)")
    axs[0].set_ylabel("Distance (px)")

    # ========== SUBPLOT B: Elbow relative ==========
    axs[1].set_title("Relative Coordinates: Elbow - Pelvis")
    for i in range(num_points - 1):
        axs[1].plot([elbow_ref[i, 0], elbow_ref[i+1, 0]],
                    [elbow_ref[i, 1], elbow_ref[i+1, 1]], color=colors_elbow[i])
    axs[1].scatter(elbow_ref[:, 0], elbow_ref[:, 1], c=colors_elbow, s=20)
    axs[1].set_aspect("equal")
    axs[1].set_xlabel("Distance (px)")
    axs[1].set_ylabel("Distance (px)")

    fig.text(
        0.5, 0.02,
        "Lighter colors represent the beginning of the movement, while darker colors indicate its end.",
        ha="center", fontsize=9, style="italic"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def plot_temporal(wrist_ref, elbow_ref, timestamps):
    """Plot time progression for wrist and elbow (X & Y) in a 2x2 grid."""

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

    # ------------------------
    # Subplot (0, 0) - Wrist X
    # ------------------------
    axs[0, 0].set_title("Wrist X over Time")
    axs[0, 0].plot(timestamps, wrist_ref[:, 0], color="green")
    axs[0, 0].grid(True)
    axs[0, 0].set_xlabel("Time (seconds)")
    axs[0, 0].set_ylabel("Distance to Reference (px)")

    # ------------------------
    # Subplot (0, 1) - Wrist Y
    # ------------------------
    axs[0, 1].set_title("Wrist Y over Time")
    axs[0, 1].plot(timestamps, wrist_ref[:, 1], color="darkgreen")
    axs[0, 1].grid(True)
    axs[0, 1].set_xlabel("Time (seconds)")
    axs[0, 1].set_ylabel("Distance to Reference (px)")

    # ------------------------
    # Subplot (1, 0) - Elbow X
    # ------------------------
    axs[1, 0].set_title("Elbow X over Time")
    axs[1, 0].plot(timestamps, elbow_ref[:, 0], color="orange")
    axs[1, 0].grid(True)
    axs[1, 0].set_xlabel("Time (seconds)")
    axs[1, 0].set_ylabel("Distance to Reference (px)")

    # ------------------------
    # Subplot (1, 1) - Elbow Y
    # ------------------------
    axs[1, 1].set_title("Elbow Y over Time")
    axs[1, 1].plot(timestamps, elbow_ref[:, 1], color="red")
    axs[1, 1].grid(True)
    axs[1, 1].set_xlabel("Time (seconds)")
    axs[1, 1].set_ylabel("Distance to Reference (px)")

    plt.tight_layout()
    plt.show()

def plot_3d(wrist_ref, elbow_ref, timestamps):
    """3D trajectory of wrist & elbow."""

    fig = plt.figure(figsize=(14, 6))

    # ===== Wrist =====
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot3D(wrist_ref[:, 0], timestamps, wrist_ref[:, 1], color="green")
    ax1.set_title("3D Wrist Trajectory (X, Y, Time)")
    ax1.set_xlabel("Distance in X to Reference (px)")
    ax1.set_ylabel("Time")
    ax1.set_zlabel("Distance in Y to Reference (px)")

    # ===== Elbow =====
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot3D(elbow_ref[:, 0], timestamps, elbow_ref[:, 1], color="purple")
    ax2.set_title("3D Elbow Trajectory (X, Y, Time)")
    ax2.set_xlabel("Distance in X to Reference (px)")
    ax2.set_ylabel("Time")
    ax2.set_zlabel("Distance in Y to Reference (px)")

    plt.tight_layout()
    plt.show()

# ===========================
#  MODULE 3: Parser Wrapper
# ===========================
def plot_coordinates(args):
    """Main function that receives parser arguments and calls the appropriate plot."""
    pelvis, wrist, wrist_ref, elbow, elbow_ref, timestamps = load_coordinates(args.file)

    print(f"\n[INFO] Archivo cargado: {args.file}")
    print(f"[INFO] Tipo de gráfica seleccionada: {args.type}\n")

    if args.type == "original":
        plot_original_coordinates(pelvis, wrist, elbow)

    elif args.type == "relative":
        plot_relative_coordinates(wrist_ref, elbow_ref)

    elif args.type == "temporal":
        plot_temporal(wrist_ref, elbow_ref, timestamps)

    elif args.type == "3d":
        plot_3d(wrist_ref, elbow_ref, timestamps)

    elif args.type == "all":
        plot_original_coordinates(pelvis, wrist, elbow)
        plot_relative_coordinates(wrist_ref, elbow_ref)
        plot_temporal(wrist_ref, elbow_ref, timestamps)
        plot_3d(wrist_ref, elbow_ref, timestamps)

# ===========================
#  MODULE 4: Motion Animation
# ===========================
def animate_motion(args):
    """
    Animates Pelvis, Right Elbow, and Right Wrist (original coordinates)
    with automatic zoom, normal Y-axis, and start markers for each joint.
    """
    json_path = args.file
    print(f"\n[INFO] Generando animación desde: {json_path}\n")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pelvis = np.array(data["Pelvis"])
    elbow = np.array(data["Codo Derecha Original"])
    wrist = np.array(data["Mano Derecha Original"])
    timestamps = np.array(data["timestamps"])
    metadata = data.get("metadata", {})
    fps = metadata.get("fps", 30)

    # --- Playback speed (slower for clarity) ---
    playback_speed = 0.2
    interval = (1000 / fps) / playback_speed

    # --- Compute coordinate bounds for zoom ---
    all_points = np.vstack([pelvis, elbow, wrist])
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    margin_x = (x_max - x_min) * 0.2
    margin_y = (y_max - y_min) * 0.2

    # --- Setup figure ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Motion Animation (Original) - Frontal View of the Player\n{metadata.get('video_name', '')}")
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)  # normal Y-axis (upward)
    ax.set_xlabel("Distance (px)")
    ax.set_ylabel("Distance (px)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Define graphical elements ---
    scat_pelvis, = ax.plot([], [], 'o', color='blue', label="Pelvis", markersize=8)
    scat_elbow, = ax.plot([], [], 'o', color='green', label="Codo Derecho", markersize=8)
    scat_wrist, = ax.plot([], [], 'o', color='red', label="Mano Derecha", markersize=8)
    line_arm, = ax.plot([], [], '-', color='orange', lw=2, label="Brazo")

    # Trails
    trail_pelvis, = ax.plot([], [], '--', color='blue', alpha=0.4)
    trail_elbow, = ax.plot([], [], '--', color='green', alpha=0.4)
    trail_wrist, = ax.plot([], [], '--', color='red', alpha=0.4)

    # Start markers (hollow circles)
    start_pelvis, = ax.plot([], [], 'o', color='blue', markersize=10, fillstyle='none', label="Inicio Pelvis")
    start_elbow, = ax.plot([], [], 'o', color='green', markersize=10, fillstyle='none', label="Inicio Codo")
    start_wrist, = ax.plot([], [], 'o', color='red', markersize=10, fillstyle='none', label="Inicio Mano")

    ax.legend()

    # --- Initialization ---
    def init():
        for element in [
            scat_pelvis, scat_elbow, scat_wrist, line_arm,
            trail_pelvis, trail_elbow, trail_wrist,
            start_pelvis, start_elbow, start_wrist
        ]:
            element.set_data([], [])
        return scat_pelvis, scat_elbow, scat_wrist, line_arm, \
               trail_pelvis, trail_elbow, trail_wrist, \
               start_pelvis, start_elbow, start_wrist

    # --- Update function ---
    def update(i):
        px, py = pelvis[i]
        ex, ey = elbow[i]
        wx, wy = wrist[i]

        scat_pelvis.set_data(px, py)
        scat_elbow.set_data(ex, ey)
        scat_wrist.set_data(wx, wy)
        line_arm.set_data([ex, wx], [ey, wy])

        # Trails (motion history)
        trail_pelvis.set_data(pelvis[:i, 0], pelvis[:i, 1])
        trail_elbow.set_data(elbow[:i, 0], elbow[:i, 1])
        trail_wrist.set_data(wrist[:i, 0], wrist[:i, 1])

        # Start positions (fixed)
        start_pelvis.set_data(pelvis[0, 0], pelvis[0, 1])
        start_elbow.set_data(elbow[0, 0], elbow[0, 1])
        start_wrist.set_data(wrist[0, 0], wrist[0, 1])

        return scat_pelvis, scat_elbow, scat_wrist, line_arm, \
               trail_pelvis, trail_elbow, trail_wrist, \
               start_pelvis, start_elbow, start_wrist

    # --- Create animation ---
    ani = animation.FuncAnimation(
        fig, update, frames=len(timestamps),
        init_func=init, blit=True, interval=interval, repeat=False
    )

    plt.tight_layout()
    plt.show()
