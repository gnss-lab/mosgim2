import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import numpy as np
import datetime
import imageio
from datetime import UTC


def plot2l(out_file, colat, lon, ts, frames1, frames2):

    time = np.array([datetime.datetime.utcfromtimestamp(float(t)) for t in ts])

    lon_m, colat_m = np.meshgrid(lon, colat)

    def some_data(i):   # function returns a 2D data array
        return frames1[i], frames2[i] 

    fig = plt.figure(figsize=(5, 7))

    ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())

    m1 = np.max(np.array(frames1))    
    m2 = np.max(np.array(frames2))    
    m3 = np.max(np.array(frames1) + np.array(frames2))    

    levels1=np.arange(-0.5,m1,0.5)
    levels2=np.arange(-0.5,m2,0.5)
    levels3=np.arange(-0.5,m3,0.5)

    cont1 = ax1.contourf(lon_m, 90.-colat_m, some_data(0)[0], levels1,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())    # first image on screen
    cont2 = ax2.contourf(lon_m, 90.-colat_m, some_data(0)[1], levels2,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())    # first image on screen
    cont3 = ax3.contourf(lon_m, 90.-colat_m, some_data(0)[0] + some_data(0)[1], levels3,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())    # first image on screen

    ax1.set_title('layer1, '+ str(time[0]))  
    ax2.set_title('layer2, '+ str(time[0]))  
    ax3.set_title('GIM, '+ str(time[0]))  

    ax1.coastlines()
    ax2.coastlines()
    ax3.coastlines()

    fig.colorbar(cont1, ax=ax1)
    fig.colorbar(cont2, ax=ax2)
    fig.colorbar(cont3, ax=ax3)
    plt.tight_layout()

    # animation function
    def animate(i):
        global cont1, cont2, cont3
        z1, z2 = some_data(i)
        cont1 = ax1.contourf(lon_m, 90.-colat_m, z1, levels1,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())
        cont2 = ax2.contourf(lon_m, 90.-colat_m, z2, levels2,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())
        cont3 = ax3.contourf(lon_m, 90.-colat_m, z1+z2, levels3,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())
        ax1.set_title('layer1, '+ str(time[i]))  
        ax2.set_title('layer2, '+ str(time[i]))  
        ax3.set_title('GIM, '+ str(time[i]))  
        return cont1, cont2, cont3

    anim = animation.FuncAnimation(fig, animate, frames=len(ts), repeat=False)
    anim.save(out_file, writer='imagemagick')

def plot1l(out_file, colat, lon, ts, frames1):

    time = np.array([datetime.datetime.utcfromtimestamp(float(t)) for t in ts])

    lon_m, colat_m = np.meshgrid(lon, colat)

    def some_data(i):   # function returns a 2D data array
        return frames1[i] 

    fig = plt.figure(figsize=(4.5, 3))

    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    m1 = np.max(np.array(frames1))    

    levels1=np.arange(-0.5,m1,0.5)

    cont1 = ax1.contourf(lon_m, 90.-colat_m, some_data(0), levels1,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())    # first image on screen

    ax1.set_title('GIM, '+ str(time[0]))  

    ax1.coastlines()

    fig.colorbar(cont1, ax=ax1, orientation='horizontal', pad=0.05)
    plt.tight_layout()

    # animation function
    def animate(i):
        global cont1
        z1 = some_data(i)
        cont1 = ax1.contourf(lon_m, 90.-colat_m, z1, levels1,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())
        ax1.set_title('GIM, '+ str(time[i]))  
        return cont1

    anim = animation.FuncAnimation(fig, animate, frames=len(ts), repeat=False)
    anim.save(out_file, writer='imagemagick')

def plot_2layer_separate_frames(out_files, colat, lon, ts, frames1, frames2, height1, height2):
    """
    out_files = {
        'animation': 'output.mp4',
        'frames': ['frame_000.png', 'frame_001.png', ...]
    }
    """
    assert len(frames1) == len(frames2) == len(out_files['frames']), \
        "frames1, frames2, and out_files['frames'] must have same length"

    time = np.array([datetime.datetime.fromtimestamp(float(t), tz=UTC) for t in ts])
    lon_m, colat_m = np.meshgrid(lon, colat)

    # Compute global limits for consistent color scaling
    layer_levels = []
    layer_frames = [
        [np.array(fr1) + np.array(fr2) for fr1, fr2 in zip(frames1, frames2)], 
        [np.array(fr) for fr in frames1], 
        [np.array(fr) for fr in frames2], 
    ]
    for frames in layer_frames:
        m = np.max(np.array(frames))
        m = (int(m / 5.) + 1) * 5.
        levels = np.arange(0, m + 2.5, 2.5)
        if len(levels) >= 30:
            levels = np.arange(0, m + 5., 5.)
        layer_levels.append(levels)
    layer_levels[1] = layer_levels[0] # use levels for GIM to ionosphere (layer 1)
    # --- Setup figure once ---
    fig = plt.figure(figsize=(5, 10))
    ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
    axs = [ax1, ax2, ax3]

    layer_names = ['Combined (GIM)', f'Ionosphere  (h={height1} km)', f'Plasmasphere (h={height2} km)']
    for ax, frames, levels, name in zip(axs, layer_frames, layer_levels, layer_names):
        ax.coastlines()
        cont = ax.contourf(lon_m, 90. - colat_m, frames[0], levels, cmap=plt.cm.jet, transform=ccrs.PlateCarree())
        cbar1 = fig.colorbar(cont, ax=ax, orientation='horizontal', pad=0.05)
        time_label = time[0].strftime("%Y-%m-%d %H:%M  UT")
        ax.set_title(f"{name}, {time_label}")

    plt.tight_layout()

    # --- Update loop ---
    for i, frame_path in enumerate(out_files['frames']):
        for ax, frames, levels, name in zip(axs, layer_frames, layer_levels, layer_names):
            cont = ax.contourf(lon_m, 90. - colat_m, frames[i], levels, cmap=plt.cm.jet, transform=ccrs.PlateCarree())
            time_label = time[i].strftime("%Y-%m-%d %H:%M UT")
            ax.set_title(f"{name}, {time_label}")

        fig.savefig(frame_path, dpi=300)

    plt.close(fig)
    print("\nAll frames saved.")

    # --- Animation creation ---
    print("Constructing animation from saved frames...")
    with imageio.get_writer(out_files['animation'], mode="I") as writer:
        for frame_path in out_files['frames']:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    print(f"\nAnimation saved to {out_files['animation']}")


def plot_1layer_separate_frames(out_files, colat, lon, ts, frames, height):
    """
    out_files = {
        'animation': 'output.mp4',
        'frames': ['frame_000.png', 'frame_001.png', ...]
    }
    """
    assert len(frames) == len(out_files['frames']), \
        "frames1 and out_files['frames'] must have same length"

    time = np.array([datetime.datetime.fromtimestamp(float(t), tz=UTC) for t in ts])
    lon_m, colat_m = np.meshgrid(lon, colat)

    m1 = np.max(np.array(frames))
    m1 = (int(m1 / 5.) + 1) * 5.
    levels = np.arange(0, m1, 2.5)
    if len(levels) >= 30:
        levels = np.arange(0, m1, 5.)

    fig = plt.figure(figsize=(4.5, 3))
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax1.coastlines()

    cont = ax1.contourf(lon_m, 90. - colat_m, frames[0], levels, cmap=plt.cm.jet, transform=ccrs.PlateCarree())
    fig.colorbar(cont, ax=ax1, orientation='horizontal', pad=0.05)
    ax1.set_title(f"GIM ({height} km), {time[0]}")

    plt.tight_layout()

    for i, frame_path in enumerate(out_files['frames']):
        cont = ax1.contourf(lon_m, 90. - colat_m, frames[i], levels, cmap=plt.cm.jet, transform=ccrs.PlateCarree())
        time_label = time[i].strftime("%Y-%m-%d %H:%M UT")
        ax1.set_title(f"GIM (single-layer h={height} km), {time_label}")

        fig.savefig(frame_path, dpi=300)

    plt.close(fig)
    print("\nAll frames saved.")

    print("Constructing animation from saved frames...")
    with imageio.get_writer(out_files['animation'], mode="I") as writer:
        for frame_path in out_files['frames']:
            image = imageio.imread(frame_path)
            writer.append_data(image)

    print(f"\nAnimation saved to {out_files['animation']}")
