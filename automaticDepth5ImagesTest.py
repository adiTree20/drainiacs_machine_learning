import pyrealsense2 as rs
import numpy as np
import cv2



# Configure depth and color streams
pipeline = rs.pipeline()
#config = rs.config()
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# start streaming


try:
    pipeline.start()
    for x in range(5):

        # wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        # apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # save images
        cv2.imwrite("RealSense" + str(x+1) + ".png", depth_colormap)
        cv2.waitKey(1)

        print ("I have captured picture", x+1)

except Exception as e:
    print(e)
    pass