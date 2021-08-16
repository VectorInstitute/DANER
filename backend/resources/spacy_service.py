import spacy
from flask_restful import Resource, reqparse


class SpacyService(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument("text", type=str, help="Text", required=True)
    parser.add_argument("model", type=str, help="Model", required=True)
    parser.add_argument("mode", type=str, help="Mode", required=True)

    print("Loading...")
    MODELS = {
        "en_core_web_sm": spacy.load("en_core_web_sm"),
        # "en_core_web_md": spacy.load("en_core_web_md"),
        # "en_core_web_lg": spacy.load("en_core_web_lg"),
        # "en_core_web_trf": spacy.load("en_core_web_trf"),
    }

    print("Loaded!")

    def put(self):
        """Get entities for displaCy ENT visualizer."""

        args = SpacyService.parser.parse_args()
        nlp = self.MODELS[args.model]

        print(args.text)

        doc = nlp(args.text)
        if args.mode == 'char':
            res = [{"token": ent.text, "start": ent.start_char, "end": ent.end_char, "label": ent.label_}
                   for ent in doc.ents]
        elif args.mode == 'token':
            res = [{"token": token.text, "label": token.ent_type_ if token.ent_type_ else 'null', "iob": token.ent_iob}
                   for token in doc]

        return {"text": doc.text, "ents": res}, 201
