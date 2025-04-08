import numpy as np
import cv2

# Read input video
cap = cv2.VideoCapture('/root/autodl-tmp/eval0303/output.avi')

# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 1103帧

# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 848
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 476

# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = cap.get(cv2.CAP_PROP_FPS)   # 获取帧率

# Set up output video
out = cv2.VideoWriter('compete_video.mp4', fourcc, fps, (2*w, h))  # 拼接原始视频和稳定后的视频
out2 = cv2.VideoWriter('stabilized_video.mp4', fourcc, fps, (w, h))  # 只保存稳定后的视频

# 读取第一帧prev
_, frame = cap.read()

# 第一帧转化成灰度图，类型是ndarray
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # prev_gray是ndarray

# Pre-define transformation-store array
transforms = np.zeros((n_frames - 1, 3), np.float32)  # n_frames是总帧数。n_frames - 1行，3列的0

for i in range(n_frames - 2):
    # 找到前一帧的特征点
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30,
                                       blockSize=3)

    # 读取下一帧
    success, curr = cap.read()
    if not success:
        break

    # 转化成灰度图
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Sanity check
    assert prev_pts.shape == curr_pts.shape

    # Filter only valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # Find transformation matrix
    m, inl = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

    # Extract translation
    dx = m[0, 2]
    dy = m[1, 2]

    # Extract rotation angle
    da = np.arctan2(m[1, 0], m[0, 0])

    # Store transformation
    transforms[i] = [dx, dy, da]

    # Move to next frame
    prev_gray = curr_gray

    print("Frame: " + str(i) + "/" + str(n_frames) + " - Tracked points: " + str(len(prev_pts)))

# 计算累计变换轨迹
trajectory = np.cumsum(transforms, axis=0)

# 定义平滑函数
def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

# 平滑轨迹函数
SMOOTHING_RADIUS = 50  # 可以调整平滑半径
def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for ii in range(3):
        smoothed_trajectory[:, ii] = movingAverage(trajectory[:, ii], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

# 计算平滑的轨迹
smoothed_trajectory = smooth(trajectory)
difference = smoothed_trajectory - trajectory

# 计算新的平滑变换数组
transforms_smooth = transforms + difference

# 重置视频流到第一帧
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 修复边界伪影
def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)  # 扩大图像，避免边界空白
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

# 处理每一帧
for i in range(n_frames - 2):
    # 读取下一帧
    success, frame = cap.read()
    if not success:
        break

    # 获取平滑变换参数
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    # 构造仿射变换矩阵
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    # 应用仿射变换到当前帧
    frame_stabilized = cv2.warpAffine(frame, m, (w, h))

    # 修复变换后的边界问题
    frame_stabilized = fixBorder(frame_stabilized)

    # 将原始帧与稳定帧并排显示
    frame_out = cv2.hconcat([frame, frame_stabilized])

    # 如果输出帧的宽度超过 1920，则缩小图像
    if frame_out.shape[1] > 1920:
        frame_out = cv2.resize(frame_out, (frame_out.shape[1] // 2, frame_out.shape[0] // 2))

    # 将处理后的帧写入输出视频
    out.write(frame_out)
    # 保存稳定化后的视频
    out2.write(frame_stabilized)

# 释放视频读取和写入对象
cap.release()
out.release()
out2.release()

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
