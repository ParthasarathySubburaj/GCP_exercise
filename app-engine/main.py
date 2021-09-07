import os

from gluoncv import model_zoo, data, utils
from flask import Flask
import json
from PIL import Image
from flask import Flask, request, jsonify


net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)


app = Flask(__name__)


@app.route("/", methods=["POST"])
def hello_world():
    img = Image.open(request.files["image"])
    img.save("/tmp/tmp.jpeg")
    x, img = data.transforms.presets.ssd.load_test("/tmp/tmp.jpeg", short=512)
   
    class_IDs, scores, bounding_boxes = net(x)
    
    index = scores.asnumpy().flatten()>0.8
    bboxes = bounding_boxes.asnumpy()[0][index]
    ids = class_IDs.asnumpy().flatten()[index]
    
    ans = []
    for bbox, class_id in zip(bboxes, ids):
        tmp_dict = {}
        tmp_dict[net.classes[int(class_id)]] = list(bbox.astype(float))
        ans.append(tmp_dict)
        
    return json.dumps(ans)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


