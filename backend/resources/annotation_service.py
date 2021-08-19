from flask_restful import Resource, reqparse

from models.al_model import ALEngine


class AnnotationService(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument("index", type=int, help="Data Index", required=True)
    parser.add_argument("label", type=str, help="label", required=True)
    parser.add_argument("annotator", type=str, help="annotator", required=True)

    al_engine = ALEngine(
        "elastic/distilbert-base-uncased-finetuned-conll03-english")

    def get(self):
        """Get entities for displaCy ENT visualizer."""

        index, res = self.al_engine.next2label()

        return {"ents": res, "index": index}, 200

    def put(self):
        args = AnnotationService.parser.parse_args()
        self.al_engine.update_dataset(args.index, args.label, args.annotator)

        return {"index": args.index}, 201


class AnnotationServiceScratch(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument("index", type=int, help="Data Index", required=True)
    parser.add_argument("label", type=str, help="label", required=True)
    parser.add_argument("annotator", type=str, help="annotator", required=True)

    al_engine = ALEngine('distilbert-base-uncased', need_train=True)

    def get(self):
        """Get entities for displaCy ENT visualizer."""

        index, res = self.al_engine.next2label()

        return {"ents": res, "index": index}, 200

    def put(self):
        args = AnnotationService.parser.parse_args()
        print(args.index, args.label, args.annotator)
        self.al_engine.update_dataset(args.index, args.label, args.annotator)

        return {"index": args.index}, 201
