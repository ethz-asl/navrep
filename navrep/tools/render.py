import numpy as np
from gym.envs.classic_control import rendering
import pyglet
from pyglet import gl


def render_lidar(
    ranges, min_angle, max_angle, min_range=0.3, mode="human", viewer=None
):
    if mode == "human":
        # Window and viewport size
        WINDOW_W = 512
        WINDOW_H = 512
        WINDOW_H_METERS = 10.0
        RESOLUTION_M_PER_PX = WINDOW_H_METERS / WINDOW_H
        VP_W = WINDOW_W
        VP_H = WINDOW_H
        # Create viewer
        if viewer is None:
            viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            viewer.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
        viewer.draw_circle(r=10, color=(0.3, 0.3, 0.3))
        win = viewer.window
        win.switch_to()
        win.dispatch_events()
        win.clear()
        gl.glViewport(0, 0, VP_W, VP_H)
        # colors
        bgcolor = np.array([0.9, 0.9, 1.0])
        lidarcolor = np.array([1.0, 0.0, 0.0])
        # Green background
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(bgcolor[0], bgcolor[1], bgcolor[2], 1.0)
        gl.glVertex3f(0, VP_H, 0)
        gl.glVertex3f(VP_W, VP_H, 0)
        gl.glVertex3f(VP_W, 0, 0)
        gl.glVertex3f(0, 0, 0)
        gl.glEnd()
        # LIDAR
        sensor_i = WINDOW_H / 2.0
        sensor_j = WINDOW_W / 2.0
        angles = np.linspace(min_angle, max_angle, len(ranges))
        ijranges = np.array(ranges) / RESOLUTION_M_PER_PX
        for th, r in zip(angles, ijranges):
            ray_start_i = np.clip(sensor_i + min_range * np.cos(th), 1, WINDOW_W - 1)
            ray_start_j = np.clip(sensor_j + min_range * np.sin(th), 1, WINDOW_H - 1)
            ray_end_i = np.clip(sensor_i + r * np.cos(th), 1, WINDOW_W - 1)
            ray_end_j = np.clip(sensor_j + r * np.sin(th), 1, WINDOW_H - 1)
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glColor4f(lidarcolor[0], lidarcolor[1], lidarcolor[2], 0.1)
            gl.glVertex3f(ray_start_i, ray_start_j, 0)
            gl.glVertex3f(ray_end_i, ray_end_j, 0)
            gl.glEnd()
        # Text
        viewer.score_label.text = "{}m".format(WINDOW_H_METERS)
        viewer.score_label.draw()
        win.flip()
        return viewer


def render_lidar_batch(
    batch_ranges, min_angle, max_angle, min_range=0.3, mode="human", viewer=None
):
    if mode == "human":
        N_COL_WINDOWS = 10
        N_ROW_WINDOWS = 10
        N_WINDOWS = N_COL_WINDOWS * N_ROW_WINDOWS
        # Window and viewport size
        WINDOW_W = 64
        WINDOW_H = 64
        WINDOW_H_METERS = 10.0
        RESOLUTION_M_PER_PX = WINDOW_H_METERS / WINDOW_H
        VP_W = WINDOW_W * N_COL_WINDOWS
        VP_H = WINDOW_H * N_ROW_WINDOWS
        # Create viewer
        if viewer is None:
            viewer = rendering.Viewer(VP_W, VP_H)
            viewer.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
        viewer.draw_circle(r=10, color=(0.3, 0.3, 0.3))
        win = viewer.window
        win.switch_to()
        win.dispatch_events()
        win.clear()
        gl.glViewport(0, 0, VP_W, VP_H)
        # colors
        bgcolor = np.array([0.9, 0.9, 1.0])
        lidarcolor = np.array([1.0, 0.0, 0.0])
        # Green background
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(bgcolor[0], bgcolor[1], bgcolor[2], 1.0)
        gl.glVertex3f(0, VP_H, 0)
        gl.glVertex3f(VP_W, VP_H, 0)
        gl.glVertex3f(VP_W, 0, 0)
        gl.glVertex3f(0, 0, 0)
        gl.glEnd()
        # LIDAR
        assert len(batch_ranges) == 100
        for n, ranges in enumerate(batch_ranges):
            col = n % N_COL_WINDOWS
            row = n // N_COL_WINDOWS
            window_offset_i = row * WINDOW_W
            window_offset_j = col * WINDOW_H
            sensor_i = window_offset_i + WINDOW_W / 2.0
            sensor_j = window_offset_j + WINDOW_H / 2.0
            angles = np.linspace(min_angle, max_angle, len(ranges))
            ijranges = np.array(ranges) / RESOLUTION_M_PER_PX
            for th, r in zip(angles, ijranges):
                ray_start_i = np.clip(
                    sensor_i + min_range * np.cos(th),
                    window_offset_i,
                    window_offset_i + WINDOW_H - 1,
                )
                ray_start_j = np.clip(
                    sensor_j + min_range * np.sin(th),
                    window_offset_j,
                    window_offset_j + WINDOW_W - 1,
                )
                ray_end_i = np.clip(
                    sensor_i + r * np.cos(th),
                    window_offset_i,
                    window_offset_i + WINDOW_H - 1,
                )
                ray_end_j = np.clip(
                    sensor_j + r * np.sin(th),
                    window_offset_j,
                    window_offset_j + WINDOW_W - 1,
                )
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glColor4f(lidarcolor[0], lidarcolor[1], lidarcolor[2], 0.1)
                gl.glVertex3f(ray_start_i, ray_start_j, 0)
                gl.glVertex3f(ray_end_i, ray_end_j, 0)
                gl.glEnd()
        # Text
        #         viewer.score_label.text = "{}m".format(WINDOW_H_METERS)
        #         viewer.score_label.draw()
        win.flip()
        return viewer
