from flask_restful import Resource, reqparse

from transformers import AutoTokenizer, AutoModelForTokenClassification

from models.huggingface_model import HFModel

label_list = ['O', 'B-PER', 'I-PER', 'B-ORG',
              'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']


class HuggingFaceService(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument("text", type=str, help="Text", required=True)
    parser.add_argument("model", type=str, help="Model", required=True)
    parser.add_argument("mode", type=str, help="Mode", required=True)

    print("Loading...")
    hf_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    hf_model = AutoModelForTokenClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(label_list))
    model = HFModel(hf_tokenizer, hf_model, label_list)
    print("Loaded!")

    def put(self):
        """Get entities for displaCy ENT visualizer."""

        args = HuggingFaceService.parser.parse_args()
        res = self.model.predict(args.text)

        return {"text": args.text, "ents": res}, 201
