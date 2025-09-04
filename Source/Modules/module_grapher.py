import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def plot_coordinates(args):
    
    # Cargar datos desde el JSON
    with open(args.file, "r") as file:
        data = json.load(file)

    # Extraer coordenadas
    pelvis_coords = np.array(data["Pelvis"])
    wrist_coords = np.array(data["Mano Derecha Original"])
    wrist_relative_coords = np.array(data["Mano Derecha Referencia"])
    timestamps = np.array(data["timestamps"])

    # Crear una escala de colores para representar el orden temporal
    num_points = len(pelvis_coords)
    colors_pelvis = cm.Blues(np.linspace(0.3, 1, num_points))   # Azul para la pelvis
    colors_wrist = cm.Reds(np.linspace(0.3, 1, num_points))   # Rojo para la muñeca original
    colors_wrist_relative = cm.Greens(np.linspace(0.3, 1, num_points))  # Verde para la muñeca relativa

    # Crear figura original con 2 subgráficos
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1: Pelvis y muñeca en coordenadas originales
    axs[0].set_title("Coordenadas Originales (Pelvis y Muñeca)")
    for i in range(num_points - 1):
        axs[0].plot([pelvis_coords[i, 0], pelvis_coords[i + 1, 0]],
                    [pelvis_coords[i, 1], pelvis_coords[i + 1, 1]], color=colors_pelvis[i])
        axs[0].plot([wrist_coords[i, 0], wrist_coords[i + 1, 0]],
                    [wrist_coords[i, 1], wrist_coords[i + 1, 1]], color=colors_wrist[i])

    axs[0].scatter(pelvis_coords[:, 0], pelvis_coords[:, 1], c=colors_pelvis, label="Pelvis", s=20)
    axs[0].scatter(wrist_coords[:, 0], wrist_coords[:, 1], c=colors_wrist, label="Muñeca", s=20)
    axs[0].set_aspect('equal')
    axs[0].legend()

    # Plot 2: Muñeca con referencia a la Pelvis
    axs[1].set_title("Coordenadas Relativas (Muñeca - Pelvis)")
    for i in range(num_points - 1):
        axs[1].plot([wrist_relative_coords[i, 0], wrist_relative_coords[i + 1, 0]],
                    [wrist_relative_coords[i, 1], wrist_relative_coords[i + 1, 1]], color=colors_wrist_relative[i])

    axs[1].scatter(wrist_relative_coords[:, 0], wrist_relative_coords[:, 1], c=colors_wrist_relative, label="Muñeca relativa", s=20)
    axs[1].set_aspect('equal')
    axs[1].legend()

    plt.tight_layout()

    # Crear una nueva figura para los gráficos temporales
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 3: Coordenada X de la muñeca relativa contra el tiempo
    axs2[0].set_title("X de la Muñeca Relativa contra el Tiempo")
    axs2[0].plot(timestamps, wrist_relative_coords[:, 0], color='green', label="X Muñeca Relativa")
    axs2[0].set_ylabel("Coordenada X")
    axs2[0].legend()
    axs2[0].grid(True)

    # Plot 4: Coordenada Y de la muñeca relativa contra el tiempo
    axs2[1].set_title("Y de la Muñeca Relativa contra el Tiempo")
    axs2[1].plot(timestamps, wrist_relative_coords[:, 1], color='red', label="Y Muñeca Relativa")
    axs2[1].set_xlabel("Tiempo (segundos)")
    axs2[1].set_ylabel("Coordenada Y")
    axs2[1].legend()
    axs2[1].grid(True)

    plt.tight_layout()

    # Crear una nueva figura para la gráfica 3D
    fig3 = plt.figure(figsize=(10, 6))
    ax3d = fig3.add_subplot(111, projection='3d')
    ax3d.set_title("Coordenadas 3D de la Muñeca Relativa (X, Y, Tiempo)")
    ax3d.plot3D(wrist_relative_coords[:, 0], timestamps, wrist_relative_coords[:, 1], color='green', label="Muñeca Relativa")
    ax3d.set_xlabel("Coordenada X")
    ax3d.set_zlabel("Coordenada Y")
    ax3d.set_ylabel("Tiempo (segundos)")
    ax3d.legend()

    plt.show()