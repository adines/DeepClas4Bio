from threading import Thread
from deepclas4bio.PredictorFactory import PredictorFactory
import flask
import redis
import uuid
import time
import json
import os


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"


IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25


app = flask.Flask(__name__)
app_root=os.path.dirname(os.path.abspath(__file__))
db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None
framework="Keras"
modelName="VGG16"



def classify_process():
    print("* Loading model...")
    predictor_factory=PredictorFactory()
    model = predictor_factory.getPredictor(framework, modelName)
    print("* Model loaded")

    while True:
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        imageIDs = []
        batch = []

        for q in queue:
            q = json.loads(q.decode("utf-8"))

            image=q['image']

            batch.append(image)


            imageIDs.append(q["id"])

        if len(imageIDs) > 0:

            for (imageID, imageName) in zip(imageIDs, batch):

                results = model.predict(imageName)
                db.set(imageID, json.dumps(results))

            db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)

        time.sleep(SERVER_SLEEP)



@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            k = str(uuid.uuid4())
            image = flask.request.files["image"]
            destination = os.path.join(app_root, 'temp/' + k + image.filename)

            image.save(destination)

            d = {"id": k, "image": destination}
            db.rpush(IMAGE_QUEUE, json.dumps(d))

            while True:
                output = db.get(k)

                if output is not None:
                    output = output.decode("utf-8")
                    data['type'] = 'classification'
                    data['image'] = image.filename
                    data['framework'] = framework
                    data['model'] = modelName
                    data['class'] = json.loads(output)
                    data['success'] = True

                    db.delete(k)
                    os.remove(destination)
                    break
                time.sleep(CLIENT_SLEEP)
    return flask.jsonify(data)


if __name__ == "__main__":

    print("* Starting model service...")
    t = Thread(target=classify_process, args=())
    t.daemon = True
    t.start()

    print("* Starting web service...")
    app.run()