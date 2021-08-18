from flask import Flask
from flask_restful import Api
from flask_cors import CORS

# from db import db

from resources.spacy_service import SpacyService
# from resources.huggingface_service import HuggingFaceService
from resources.annotation_service import AnnotationService, AnnotationServiceScratch


app = Flask(__name__)
app.config['MONGOALCHEMY_DATABASE'] = 'myDatabase'
app.config['PROPAGATE_EXCEPTIONS'] = True
app.secret_key = 'neraBackend'
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)


api.add_resource(SpacyService, "/spacy")
# api.add_resource(HuggingFaceService, "/huggingface")
api.add_resource(AnnotationService, "/annotation")
api.add_resource(AnnotationServiceScratch, "/annotation_scratch")


if __name__ == '__main__':
    # We plan to use database at first, but it turns out to be unnecessary
    # db.init_app(app)
    app.run(host="0.0.0.0", port=5000, debug=True)
