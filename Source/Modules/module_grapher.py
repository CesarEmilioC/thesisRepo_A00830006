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
    timestamps = np.array(data["timestamps"])

    return pelvis, wrist, wrist_ref, timestamps


# ===========================
#  MODULE 2: Plot Functions
# ===========================
def plot_original_coordinates(pelvis, wrist):
    """Plot pelvis and wrist in original coordinates."""
    num_points = len(pelvis)
    colors_pelvis = cm.Blues(np.linspace(0.3, 1, num_points))
    colors_wrist = cm.Reds(np.linspace(0.3, 1, num_points))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Coordenadas Originales (Pelvis y Muñeca)")

    for i in range(num_points - 1):
        ax.plot([pelvis[i, 0], pelvis[i + 1, 0]],
                [pelvis[i, 1], pelvis[i + 1, 1]], color=colors_pelvis[i])
        ax.plot([wrist[i, 0], wrist[i + 1, 0]],
                [wrist[i, 1], wrist[i + 1, 1]], color=colors_wrist[i])

    ax.scatter(pelvis[:, 0], pelvis[:, 1], c=colors_pelvis, label="Pelvis", s=20)
    ax.scatter(wrist[:, 0], wrist[:, 1], c=colors_wrist, label="Muñeca", s=20)
    ax.set_aspect("equal")
    ax.legend()

    # Leyenda descriptiva del gradiente de color
    fig.text(
        0.5, 0.02,
        "Entre más suave el color, el movimiento comienza ahí; entre más oscuro, el movimiento termina.",
        ha="center", fontsize=9, style="italic"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def plot_relative_coordinates(wrist_ref):
    """Plot wrist relative to pelvis."""
    num_points = len(wrist_ref)
    colors_rel = cm.Greens(np.linspace(0.3, 1, num_points))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Coordenadas Relativas (Muñeca - Pelvis)")

    for i in range(num_points - 1):
        ax.plot([wrist_ref[i, 0], wrist_ref[i + 1, 0]],
                [wrist_ref[i, 1], wrist_ref[i + 1, 1]], color=colors_rel[i])

    ax.scatter(wrist_ref[:, 0], wrist_ref[:, 1], c=colors_rel, label="Muñeca relativa", s=20)
    ax.set_aspect("equal")
    ax.legend()

    fig.text(
        0.5, 0.02,
        "Entre más suave el color, el movimiento comienza ahí; entre más oscuro, el movimiento termina.",
        ha="center", fontsize=9, style="italic"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def plot_temporal(wrist_ref, timestamps):
    """Plot X and Y wrist-relative coordinates over time."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].set_title("X de la Muñeca Relativa contra el Tiempo")
    axs[0].plot(timestamps, wrist_ref[:, 0], color='green', label="X Muñeca Relativa")
    axs[0].set_ylabel("Coordenada X")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title("Y de la Muñeca Relativa contra el Tiempo")
    axs[1].plot(timestamps, wrist_ref[:, 1], color='red', label="Y Muñeca Relativa")
    axs[1].set_xlabel("Tiempo (segundos)")
    axs[1].set_ylabel("Coordenada Y")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_3d(wrist_ref, timestamps):
    """Plot 3D trajectory: X, Y, Time."""
    fig = plt.figure(figsize=(10, 6))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.set_title("Coordenadas 3D de la Muñeca Relativa (X, Y, Tiempo)")

    ax3d.plot3D(
        wrist_ref[:, 0], timestamps, wrist_ref[:, 1],
        color='green', label="Muñeca Relativa"
    )
    ax3d.set_xlabel("Coordenada X")
    ax3d.set_ylabel("Tiempo (segundos)")
    ax3d.set_zlabel("Coordenada Y")
    ax3d.legend()

    plt.show()


# ===========================
#  MODULE 3: Parser Wrapper
# ===========================
def plot_coordinates(args):
    """Main function that receives parser arguments and calls the appropriate plot."""
    pelvis, wrist, wrist_ref, timestamps = load_coordinates(args.file)

    print(f"\n[INFO] Archivo cargado: {args.file}")
    print(f"[INFO] Tipo de gráfica seleccionada: {args.type}\n")

    if args.type == "original":
        plot_original_coordinates(pelvis, wrist)
    elif args.type == "relative":
        plot_relative_coordinates(wrist_ref)
    elif args.type == "temporal":
        plot_temporal(wrist_ref, timestamps)
    elif args.type == "3d":
        plot_3d(wrist_ref, timestamps)
    elif args.type == "all":
        plot_original_coordinates(pelvis, wrist)
        plot_relative_coordinates(wrist_ref)
        plot_temporal(wrist_ref, timestamps)
        plot_3d(wrist_ref, timestamps)

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
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_title(f"Motion Animation (Original) - Frontal View of the Player\n{metadata.get('video_name', '')}")
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)  # normal Y-axis (upward)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
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
