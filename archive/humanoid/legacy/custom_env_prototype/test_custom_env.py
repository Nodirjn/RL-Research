import mujoco
import mujoco.viewer
import time


def main():
    print("--- 3-BOSQICH: Shaxsiy XML muhitni test qilish ---")

    model = mujoco.MjModel.from_xml_path("my_robot.xml")
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Motorlarga kuch beramiz
            data.ctrl[0] = 1.0
            data.ctrl[1] = -0.5

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_frame = model.opt.timestep - (time.time() - step_start)
            if time_until_next_frame > 0:
                time.sleep(time_until_next_frame)


if __name__ == "__main__":
    main()