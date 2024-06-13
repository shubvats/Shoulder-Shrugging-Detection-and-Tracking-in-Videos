import cv2
import numov as no
face_
cascade = cvz.lascadeclassirler\cvz.data.haarcascades
haarcascade Trontalrace derault.xml')
def
detect_faces(frame):
gray =
cv2. cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# Draw rectangles around the faces
for
(x, V, W, h) in faces:
cv2. rectangle(frame, (x, y), (xtw, yth), (255, 0, 0), 2)
return frame
def detect_shoulder(img_gray,
X_tace, Y_Tace, W_tace, h_race = race
face. direction,
x scale=0.75
scale=0.75) :
# Define shoulder box components
W = int (x scale * W_race)
= int (y_scale * h_face)
y = _face + h face * 3//4
# Haltwav down the head position
if direction == "right":
X = X_Tace + W_face - W // 20
# Richt and of the fare hoy
elif direction
"left":
= x_ face -
• W+ W// 20 # W to the left of the start of face box
rectangle = (x, v,
w,
# Calculate position of shoulder in each x strip
x _positions = [1
_positions
= I
for delta x in range(w):
thiS X
=x + delta X
this_y = calculate_ma
if this_y
is None:
continue # Don't 00:00
x_positions. append(th_x)
y_positions.append (this_y)
# Extract line from positions
lines = []
for index in range(len (x positions)) :
lines.append((x_positions [index], y_positions (index]))
# Extract line of best fit from lines
x _positions = p.array (x_positions)
y_positions = p.array (y_positions)
A = np. vstack([x positions, np.ones (len (x positions)) ]).T
slope, intercept = np. linalg.Istsq(A, y_positions, rcond=None) [0]
line_yo = int(x_positions|o] * slope + intercept)
line_y1 = int (x _positions[-1] * slope + intercept)
line = [(x_positions[0], line_y0), (x_positions[-1], line_y1)]
# Decide on value
value = p.array([line[0] [1], line[1] [1]]) .mean()
# Return rectangle and positions
return line, lines, rectangle, value
def calculate_max_contrast_pixel(img_gray, x, y, h, top_values_to_consider=3, search_width=20) :
columns = img_gray[y:y+h, x-search width//2:x+search_width//2]
column_average = columns.mean(axis=1)
gradient = np.gradient (column_average, 3)
gradient = np.absolute(gradient)
max_indices = np.argpartition(gradient,
-top_values_to_consider) [-top_values_to_consider:]
max values = gradient [max indices]
if max_values-sum()
< top_values_to_consider:
return None
weighted_indices = (max_indices * max_values)
weighted_average_index = weighted_indices.sum() / max values.sum()
index = int (weighted_ average_index)
index = y + index
return index

# Access the live video feed
video_capture = cv2. VideoCapture(0)
# Change 0 to 1 or higher depending on the camera index
# Flag to track initial joint line position
initial_joint_line_y = None
# Thresholds for improper movement detection
normal_threshold = 20
shrug_threshold = 50
while True:
# Capture frame-by-frame
ret, frame = video_capture.read()
# Detect faces
frame_with_faces = detect_faces (frame. copy())
# Convert frame to grayscale for shoulder detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
Detect faces again for shoulder detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# Detect shoulders for each detected face
for (x, y, W, h) in faces:
# Detect right shoulder
right_shoulder_line,
right_shoulder_rectangle, right_shoulder_value = detect_shoulder(gray, (x, y, w, h),
"right")
# Detect left shoulder
left_shoulder_line,
left_shoulder_rectangle, left_shoulder_value = detect_shoulder(gray, (x, y, w, h), "left")
# Draw a line connecting point 1 of the right shoulder line and point 1 of the left shoulder line
point1_right = right_shoulder_line[1] # Point 1 on the right shoulder line
point1_left = left_shoulder_line[0] # Point 1 on the left shoulder line
joint_line = ((point1_right[0], point1_right[1]), (point1_left[0l, pointl_left(1])] # Joint line
#Check if initial position of joint line has been recorded
if initial_joint_line_y is None:
initial_joint_line_y = joint_line[0] [1]
# Compare current position with initial position
current_joint_line_y = joint_line[0] [1]
# Detect improper movement based on shoulder heights
if abs (left_shoulder_value - right_shoulder_value) > normal_threshold:
# Improper movement detected
cv2. putText(frame_with_faces,
"Illegal Movement", (50, 100), cv2.FONT _HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2. LINE_AA)
elif current_joint_line_y < initial_joint_line_y - shrug_threshold:
# Extreme shrugging detected
cv2. putText (frame _with_faces,
"Extreme Shrugging Detected",
(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
elif current_joint_line_y < initial_joint_line_y
- normal_threshold:
# Shrugging detected
cv2. putText (frame _with_faces,
"Shrugging Detected", (50, 50),
CV2. FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE AA)
elif abs (current_joint_line_y - initial_joint_line_y)
< normal_threshold:
# Normal shoulder
cv2. putText (frame _with_faces,
"Normal Shoulder", (50, 50), cV2. FONT HERSHEY SIMPLEX, 1, (0, 255, 0), 2, cv2. LINE AA)
# Draw line representing joint line
cv2. line(frame _with_faces, joint_line[0l, joint_line[1], (0, 255, 255), 2)
# Display the resulting frame
cv2. imshow( 'Face and Shoulder Detection', frame_with_faces)
# Break the loop when 'g' is pressed
if cv2.waitKey (1) & 0xFF == ord('q"):
break
# Release the capture
video _capture.release()
cv2. destroyAllWindows()
