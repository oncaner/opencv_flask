import os
from io import BytesIO
import cv2
from flask import Flask, render_template, Response, url_for, request, send_file,make_response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename, redirect

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////Users/Caner/Desktop/OpenCV_Flask/DBase.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor


class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(50))
    data = db.Column(db.LargeBinary)
    path = db.Column(db.String(100))


@app.route('/streaming')
def streaming():
    upload = Upload.query.filter_by().first()
    return render_template("stream.html", upload=upload)


@app.route('/motiondetection', methods=["GET", "POST"])
def uploading():
    if request.method == "POST":
        file = request.files['file']
        if file.filename != '':
            filename1 = secure_filename(file.filename)
            filepath = os.path.join(app.root_path, filename1)
            file.save(filepath)
            upload = Upload(filename=file.filename, data=file.read(), path=filepath)

            db.session.add(upload)
            db.session.commit()

            return redirect(url_for("streaming"))
        else:
            return render_template("uploading.html")
    return render_template("uploading.html")


@app.route('/download/<upload_id>')
def download(upload_id):
    upload = Upload.query.filter_by(id=upload_id).first()

    return send_file(BytesIO(upload.data), attachment_filename=upload.filename, as_attachment=True)


# ************************************************************************************************************* #


def gen():
    """Video streaming generator function."""
    datas = Upload.query.order_by(Upload.id.desc()).first()
    path = datas.path

    cap = cv2.VideoCapture(path)

    # Read until video is completed
    while (cap.isOpened()):
        ret, frame = cap.read()  # import image
        if not ret:  # if vid finish repeat
            frame = cv2.VideoCapture("768x576.avi")
            continue
        if ret:  # if there is a frame continue with code
            image = cv2.resize(frame, (0, 0), None, 1, 1)  # resize image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
            fgmask = sub.apply(gray)  # uses the background subtraction
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
            closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            dilation = cv2.dilate(opening, kernel)
            retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            minarea = 300
            maxarea = 30000
            for i in range(len(contours)):  # cycles through all contours in current frame
                if hierarchy[
                    0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
                    area = cv2.contourArea(contours[i])  # area of contour
                    if minarea < area < maxarea:  # area threshold for contour
                        # calculating centroids of contours
                        cnt = contours[i]
                        M = cv2.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        # gets bounding points of contour to create rectangle
                        # x,y is top left corner and w,h is width and height
                        x, y, w, h = cv2.boundingRect(cnt)
                        # creates a rectangle around contour
                        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 127, 255), 2)
                        # Prints centroid text in order to double check later on
                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, .2,
                                    (0, 0, 255), 1)
                        cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=3, thickness=2,
                                       line_type=cv2.LINE_8)
        # cv2.imshow("countours", image)
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
            break
# Video feed

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/info")
def videos():
    datas = Upload.query.all()

    return render_template("information.html", datas = datas)

@app.route("/delete/<string:id>")
def delete(id):
    video = Upload.query.filter_by(id=id).first()
    db.session.delete(video)
    db.session.commit()

    return redirect(url_for("videos"))



@app.route("/")
def index():

    return render_template("index.html")


@app.route("/facedetection")
def faceDetection():
    faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(40) == 27:
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

    return render_template("face_detection.html")


if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)
